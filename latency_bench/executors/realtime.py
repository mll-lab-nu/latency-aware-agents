from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import (
    Action,
    ActionEvent,
    ActionHoldPolicy,
    EpisodeMetrics,
    LatencyRecord,
    Observation,
    OrderingPolicy,
    PolicyOutput,
    StepRecord,
)
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors.env_step_backend import _resolve_start_method, _set_thread_env_defaults
from latency_bench.executors import realtime_warmup
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
    profile_metadata_from_source,
)
from latency_bench.logging.records import step_record_info
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy


REALTIME_INFERENCE_START_METHOD = "spawn"


@dataclass
class RealtimeObservationTask:
    """
    env -> worker
    """
    obs_id: int
    observation: Observation
    submit_env_step: int
    submit_time_ms: float


@dataclass
class RealtimePolicyCompletion:
    """
    results returned by workers
    """
    task: RealtimeObservationTask
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    worker_id: int
    device: str
    error: BaseException | None = None


@dataclass
class RealtimePolicyResult:
    """
    slim worker -> parent result; parent reattaches the observation task.
    """
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    error: BaseException | None = None


@dataclass
class RealtimeActionDispatch:
    action_id: int
    action: Action
    obs_env_step: int
    ready_time_ms: float
    ready_step_float: float
    latency_raw_frames: int | None = None
    ready_raw_frame: int | None = None
    apply_raw_frame: int | None = None
    applied: bool = False
    dropped: bool = False


class ProcessInferencePool:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        inference_devices: list[str],
    ):
        _set_thread_env_defaults()
        self.config = config
        self.inference_devices = list(inference_devices)
        self.start_method = REALTIME_INFERENCE_START_METHOD
        self._ctx = mp.get_context(self.start_method)
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self._busy_worker_tasks: dict[int, RealtimeObservationTask] = {}
        self.closed = False
        self._start_workers()

    @property
    def in_flight_count(self) -> int:
        return len(self._busy_worker_tasks)

    @property
    def idle_worker_count(self) -> int:
        return len(self.inference_devices) - len(self._busy_worker_tasks)

    def reset_episode(self) -> None:
        self.drain_until_idle()

    def drain_until_idle(self) -> dict[str, float | int]:
        self._raise_if_closed()
        busy_before = self.in_flight_count
        wait_start_s = time.perf_counter()
        drained_completion_count = 0
        while self.in_flight_count:
            completions = self._drain_all_ready()
            drained_completion_count += len(completions)
            if self.in_flight_count:
                time.sleep(0.001)
        return {
            "pre_episode_in_flight_count": int(busy_before),
            "pre_episode_drained_completion_count": int(drained_completion_count),
            "pre_episode_drain_wait_ms": float((time.perf_counter() - wait_start_s) * 1000.0),
            "pre_episode_in_flight_after_drain": int(self.in_flight_count),
        }

    def submit(self, task: RealtimeObservationTask) -> bool:
        self._raise_if_closed()
        for worker_id in range(len(self.inference_devices)):
            if worker_id not in self._busy_worker_tasks:
                self._connections[worker_id].send(("predict", {"task": task}))
                self._busy_worker_tasks[worker_id] = task
                return True
        return False

    def drain_completed(self, now_ms: float) -> list[RealtimePolicyCompletion]:
        del now_ms
        self._raise_if_closed()
        return self._drain_all_ready()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", {}))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self._processes:
            process.join(timeout=0.2)
        terminated_processes = []
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                terminated_processes.append(process)
        for process in terminated_processes:
            process.join(timeout=1.0)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass

    def _start_workers(self) -> None:
        try:
            for worker_id, device in enumerate(self.inference_devices):
                parent_conn, child_conn = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=_inference_worker,
                    args=(child_conn, self.config, worker_id, device, self.start_method),
                    name=f"latency-bench-realtime-inference-{worker_id}",
                )
                process.start()
                child_conn.close()
                self._connections.append(parent_conn)
                self._processes.append(process)

            for worker_id, connection in enumerate(self._connections):
                tag, payload = connection.recv()
                if tag == "error":
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))
                if tag != "ready":
                    raise RuntimeError(f"inference worker {worker_id} returned {tag!r} during startup")
        except Exception:
            self.close()
            raise

    def _drain_all_ready(self) -> list[RealtimePolicyCompletion]:
        completed: list[RealtimePolicyCompletion] = []
        for worker_id, connection in enumerate(self._connections):
            while connection.poll():
                tag, payload = connection.recv()
                if tag == "result":
                    task = self._busy_worker_tasks.pop(worker_id)
                    result = payload["result"]
                    completed.append(
                        RealtimePolicyCompletion(
                            task=task,
                            policy_output=result.policy_output,
                            completion_time_ms=result.completion_time_ms,
                            latency_record=result.latency_record,
                            worker_id=worker_id,
                            device=self.inference_devices[worker_id],
                            error=result.error,
                        )
                    )
                elif tag == "error":
                    self._busy_worker_tasks.pop(worker_id, None)
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))
        return completed

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("inference pool is closed")


class _RealtimeResultCoordinator:
    def __init__(self, *, ordering_policy: OrderingPolicy):
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.issue_order: list[int] = []
        self.result_buffer: dict[int, RealtimePolicyCompletion] = {}
        self.ignored_obs_ids: set[int] = set()
        self.ignored_late_result_count = 0
        self.discarded_ready_result_count = 0
        self.fifo_blocked_count = 0

    def reset(self) -> None:
        self.issue_order.clear()
        self.result_buffer.clear()
        self.ignored_obs_ids.clear()
        self.ignored_late_result_count = 0
        self.discarded_ready_result_count = 0
        self.fifo_blocked_count = 0

    def mark_submitted(self, task: RealtimeObservationTask) -> None:
        self.issue_order.append(int(task.obs_id))

    def add_completion(self, completion: RealtimePolicyCompletion) -> list[RealtimePolicyCompletion]:
        obs_id = int(completion.task.obs_id)
        if obs_id in self.ignored_obs_ids:
            self.ignored_obs_ids.remove(obs_id)
            self.ignored_late_result_count += 1
            return [completion]
        if obs_id in self.issue_order:
            self.result_buffer[obs_id] = completion
        return []

    def release_ready(self) -> tuple[list[RealtimePolicyCompletion], list[RealtimePolicyCompletion]]:
        if self.ordering_policy == OrderingPolicy.LATEST_READY:
            return self._release_latest_ready()
        return self._release_fifo()

    def _release_latest_ready(self) -> tuple[list[RealtimePolicyCompletion], list[RealtimePolicyCompletion]]:
        if not self.result_buffer:
            return [], []
        selected_obs_id = max(self.result_buffer)
        selected = self.result_buffer.pop(selected_obs_id)
        discarded: list[RealtimePolicyCompletion] = []
        for obs_id in sorted(list(self.result_buffer)):
            if obs_id < selected_obs_id:
                discarded.append(self.result_buffer.pop(obs_id))
        for obs_id in list(self.issue_order):
            if obs_id < selected_obs_id:
                self.ignored_obs_ids.add(obs_id)
        self.issue_order = [obs_id for obs_id in self.issue_order if obs_id > selected_obs_id]
        self.discarded_ready_result_count += len(discarded)
        return [selected], discarded

    def _release_fifo(self) -> tuple[list[RealtimePolicyCompletion], list[RealtimePolicyCompletion]]:
        released: list[RealtimePolicyCompletion] = []
        while self.issue_order and self.issue_order[0] in self.result_buffer:
            obs_id = self.issue_order.pop(0)
            released.append(self.result_buffer.pop(obs_id))
        if not released and self.issue_order and self.result_buffer:
            self.fifo_blocked_count += 1
        return released, []


class RealtimeLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner | None,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
        inference_pool: Any | None = None,
        config: dict[str, Any] | None = None,
        inference_devices: list[str] | None = None,
        use_env_process: bool = False,
        video_recorder: Any | None = None,
        video_output_dir: str | None = None,
        video_fps: float | None = None,
    ):
        self.env = env
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.latency_source = latency_source
        self.action_scheduler = action_scheduler
        self.clock = clock
        self.logger = logger
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.inference_pool = inference_pool
        self.config = config
        self.inference_devices = list(inference_devices or [])
        self.use_env_process = bool(use_env_process)
        self.video_recorder = video_recorder
        self.video_output_dir = video_output_dir
        self.video_fps = video_fps
        self._owns_inference_pool = inference_pool is None
        self._episode_start_s = 0.0
        self._next_obs_id = 0
        self._next_action_id = 0

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        pool = self._ensure_inference_pool()
        if self.use_env_process:
            return self._run_episode_with_env_process(
                pool=pool,
                episode_id=episode_id,
                seed=seed,
                eval_max_steps=eval_max_steps,
            )
        return self._run_episode_local_env(
            pool=pool,
            episode_id=episode_id,
            seed=seed,
            eval_max_steps=eval_max_steps,
        )

    def run_warmup(self, *, eval_max_steps: int) -> dict[str, Any]:
        collector = realtime_warmup.RealtimeWarmupCollector(
            save_step_records=False,
            save_action_records=True,
            save_latency_records=True,
        )
        saved_logger = self.logger
        saved_video_recorder = self.video_recorder
        saved_video_output_dir = self.video_output_dir
        try:
            self.logger = collector
            self.video_recorder = None
            self.video_output_dir = None
            warmup_episode_id = -1
            while len(collector.e2e_latency_values_ms) < realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                self.run_episode(
                    episode_id=warmup_episode_id,
                    seed=None,
                    eval_max_steps=min(eval_max_steps, realtime_warmup.WARMUP_EPISODE_MAX_STEPS),
                )
                if len(collector.latency_records) > realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                    collector.latency_records = collector.latency_records[
                        -realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                    ]
                if len(collector.action_records) > realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                    collector.action_records = collector.action_records[
                        -realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                    ]
                if realtime_warmup.is_realtime_warmup_stable(collector.e2e_latency_values_ms):
                    break
                if len(collector.e2e_latency_values_ms) >= realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES:
                    break
                warmup_episode_id -= 1
        finally:
            self.logger = saved_logger
            self.video_recorder = saved_video_recorder
            self.video_output_dir = saved_video_output_dir

        stable = realtime_warmup.is_realtime_warmup_stable(collector.e2e_latency_values_ms)
        summary = realtime_warmup.write_realtime_warmup_artifacts(
            self.config["logging"]["output_dir"],
            collector,
            stable=stable,
        )
        if not stable:
            print(
                "WARNING: realtime warmup did not stabilize; proceeding with the measured "
                "episode anyway (inspect warmup_stability.png and warmup_summary.json). "
                f"reason={summary.get('warmup_stability_reason')} "
                f"trimmed_mean_span_ms={summary.get('warmup_trimmed_mean_span_ms')} "
                f"median_span_ms={summary.get('warmup_median_span_ms')}",
                file=sys.stderr,
                flush=True,
            )
        return summary

    def close(self) -> None:
        if self._owns_inference_pool and self.inference_pool is not None:
            self.inference_pool.close()
        self.env.close()

    def _ensure_inference_pool(self) -> Any:
        if self.inference_pool is None:
            self.inference_pool = ProcessInferencePool(
                config=self.config or {},
                inference_devices=self.inference_devices,
            )
        return self.inference_pool

    def _run_episode_local_env(
        self,
        *,
        pool: Any,
        episode_id: int,
        seed: int | None,
        eval_max_steps: int,
    ) -> EpisodeMetrics:
        pre_episode_drain = pool.drain_until_idle()
        self.action_scheduler.reset()
        self.env.reset(seed=seed)
        if self.video_recorder is not None:
            self.video_recorder.start_episode(episode_id=episode_id, metadata={})
            self.video_recorder.record_frame(self.env.render_game_frame())
        coordinator = _RealtimeResultCoordinator(ordering_policy=self.action_scheduler.ordering_policy)
        self._episode_start_s = self.time_fn()

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        submitted_observation_frames = 0
        dropped_observation_count = 0
        env_deadline_miss_count = 0
        max_wall_clock_lag_ms = 0.0

        for env_step in range(int(eval_max_steps)):
            scheduled_time_ms = self.clock.step_to_time_ms(env_step)
            target_wall_s = self._episode_start_s + scheduled_time_ms / 1000.0
            now_s = self.time_fn()
            if now_s < target_wall_s:
                self.sleep_fn(target_wall_s - now_s)
            current_wall_elapsed_ms = self._wall_elapsed_ms()
            wall_clock_lag_ms = max(0.0, current_wall_elapsed_ms - scheduled_time_ms)
            if wall_clock_lag_ms > self.clock.frame_ms:
                env_deadline_miss_count += 1
            max_wall_clock_lag_ms = max(max_wall_clock_lag_ms, wall_clock_lag_ms)

            self._drain_and_release_results(
                pool=pool,
                coordinator=coordinator,
                now_ms=current_wall_elapsed_ms,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
            )

            observation_submitted = False
            observation_dropped = False
            if self.decision_scheduler.should_observe(env_step, scheduled_time_ms):
                task = self._make_observation_task(env_step=env_step, submit_time_ms=current_wall_elapsed_ms)
                if pool.submit(task):
                    coordinator.mark_submitted(task)
                    submitted_observation_frames += 1
                    observation_submitted = True
                else:
                    dropped_observation_count += 1
                    observation_dropped = True

            active_action = self.action_scheduler.update(env_step, scheduled_time_ms)
            result = self.env.step(active_action)
            episode_done = result.done or result.truncated
            if self.video_recorder is not None and not episode_done:
                self.video_recorder.record_frame(self.env.render_game_frame())
            step_info = step_record_info(result.info)
            step_info.update(
                {
                    "scheduled_time_ms": scheduled_time_ms,
                    "wall_elapsed_ms": current_wall_elapsed_ms,
                    "frame_time_ms": self.clock.frame_ms,
                    "wall_clock_lag_ms": wall_clock_lag_ms,
                    "observation_submitted": observation_submitted,
                    "observation_dropped": observation_dropped,
                    "in_flight_count": pool.in_flight_count,
                    "idle_worker_count": pool.idle_worker_count,
                }
            )
            step_info.update(
                active_action_step_info(
                    applied_events=self.action_scheduler.get_applied_events(),
                    env_step=env_step,
                    scheduled_time_ms=scheduled_time_ms,
                    frame_ms=self.clock.frame_ms,
                    latency_type="measured",
                    profile_metadata=profile_metadata_from_source(self.latency_source),
                )
            )
            record = StepRecord(
                episode_id=episode_id,
                env_step=env_step,
                sim_time_ms=scheduled_time_ms,
                active_action=active_action,
                reward=float(result.reward),
                done=episode_done,
                info=step_info,
            )
            step_records.append(record)
            if self.logger is not None:
                self.logger.log_step(record)
            if episode_done:
                break

        self._drain_and_release_results(
            pool=pool,
            coordinator=coordinator,
            now_ms=self._wall_elapsed_ms(),
            action_events=action_events,
            latency_values_ms=latency_values_ms,
        )
        metadata = self._episode_metadata(
            seed=seed,
            step_records=step_records,
            submitted_observation_frames=submitted_observation_frames,
            dropped_observation_count=dropped_observation_count,
            env_deadline_miss_count=env_deadline_miss_count,
            max_wall_clock_lag_ms=max_wall_clock_lag_ms,
            coordinator=coordinator,
            pool=pool,
            pre_episode_drain=pre_episode_drain,
        )
        metrics = compute_episode_metrics(
            episode_id=episode_id,
            step_records=step_records,
            action_events=action_events,
            latency_values_ms=latency_values_ms,
            metadata=metadata,
            frame_ms=self.clock.frame_ms,
        )
        if self.logger is not None:
            self.logger.log_episode_metrics(metrics)
        if self.video_recorder is not None:
            self.video_recorder.close_episode({"episode_return": metrics.episode_return})
        return metrics

    def _run_episode_with_env_process(
        self,
        *,
        pool: Any,
        episode_id: int,
        seed: int | None,
        eval_max_steps: int,
    ) -> EpisodeMetrics:
        pre_episode_drain = pool.drain_until_idle()
        ctx = mp.get_context(_resolve_start_method())
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_env_loop_worker,
            args=(
                child_conn,
                self.config,
                episode_id,
                seed,
                eval_max_steps,
                self.video_output_dir,
                self.video_fps,
            ),
            name="latency-bench-realtime-env",
        )
        process.start()
        child_conn.close()

        coordinator = _RealtimeResultCoordinator(ordering_policy=self.action_scheduler.ordering_policy)
        self._episode_start_s = self.time_fn()
        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        action_event_index: dict[int, int] = {}
        latency_values_ms: list[float] = []
        submitted_observation_frames = 0
        dropped_observation_count = 0
        env_deadline_miss_count = 0
        max_wall_clock_lag_ms = 0.0
        pending_step_info: dict[int, dict[str, Any]] = {}
        done = False

        try:
            def release_ready_actions(now_ms: float) -> None:
                nonlocal done
                released = self._drain_and_release_results(
                    pool=pool,
                    coordinator=coordinator,
                    now_ms=now_ms,
                    action_events=action_events,
                    latency_values_ms=latency_values_ms,
                )
                for event in released:
                    try:
                        parent_conn.send(("action_dispatch", _action_event_dispatch_payload(event)))
                    except BrokenPipeError:
                        done = True
                        break
                    action_event_index[event.action_id] = action_events.index(event)

            def handle_env_message(tag: str, payload: dict[str, Any]) -> None:
                nonlocal submitted_observation_frames
                nonlocal dropped_observation_count
                nonlocal max_wall_clock_lag_ms
                nonlocal env_deadline_miss_count
                nonlocal done
                if tag == "observation":
                    release_ready_actions(self._wall_elapsed_ms())
                    if done:
                        return
                    incoming_task = payload["task"]
                    task = RealtimeObservationTask(
                        obs_id=self._next_obs_id,
                        observation=incoming_task.observation,
                        submit_env_step=incoming_task.submit_env_step,
                        submit_time_ms=incoming_task.submit_time_ms,
                    )
                    self._next_obs_id += 1
                    if pool.submit(task):
                        coordinator.mark_submitted(task)
                        submitted_observation_frames += 1
                        pending_step_info[int(task.submit_env_step)] = {
                            "observation_submitted": True,
                            "observation_dropped": False,
                            "in_flight_count": pool.in_flight_count,
                            "idle_worker_count": pool.idle_worker_count,
                        }
                    else:
                        dropped_observation_count += 1
                        pending_step_info[int(task.submit_env_step)] = {
                            "observation_submitted": False,
                            "observation_dropped": True,
                            "in_flight_count": pool.in_flight_count,
                            "idle_worker_count": pool.idle_worker_count,
                        }
                elif tag == "action_ack":
                    index = action_event_index.get(payload["action_id"])
                    if index is not None:
                        target = action_events[index]
                        target.applied = payload["applied"]
                        target.apply_raw_frame = payload["apply_raw_frame"]
                elif tag == "step":
                    record = payload["record"]
                    annotation = pending_step_info.pop(int(record.env_step), None)
                    if annotation is not None:
                        record.info.update(annotation)
                    max_wall_clock_lag_ms = max(max_wall_clock_lag_ms, float(record.info.get("wall_clock_lag_ms", 0.0)))
                    if float(record.info.get("wall_clock_lag_ms", 0.0)) > self.clock.frame_ms:
                        env_deadline_miss_count += 1
                    step_records.append(record)
                    if self.logger is not None:
                        self.logger.log_step(record)
                elif tag == "episode_done":
                    done = True
                elif tag == "error":
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))

            while not done:
                while parent_conn.poll():
                    tag, payload = parent_conn.recv()
                    handle_env_message(tag, payload)
                    if done:
                        break
                if done:
                    break
                now_ms = self._wall_elapsed_ms()
                release_ready_actions(now_ms)
                if done:
                    continue

                if not parent_conn.poll(0.001):
                    if not process.is_alive():
                        done = True
                    continue

                tag, payload = parent_conn.recv()
                handle_env_message(tag, payload)

            self._drain_and_release_results(
                pool=pool,
                coordinator=coordinator,
                now_ms=self._wall_elapsed_ms(),
                action_events=action_events,
                latency_values_ms=latency_values_ms,
            )
            metadata = self._episode_metadata(
                seed=seed,
                step_records=step_records,
                submitted_observation_frames=submitted_observation_frames,
                dropped_observation_count=dropped_observation_count,
                env_deadline_miss_count=env_deadline_miss_count,
                max_wall_clock_lag_ms=max_wall_clock_lag_ms,
                coordinator=coordinator,
                pool=pool,
                pre_episode_drain=pre_episode_drain,
            )
            metrics = compute_episode_metrics(
                episode_id=episode_id,
                step_records=step_records,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
                metadata=metadata,
                frame_ms=self.clock.frame_ms,
            )
            if self.logger is not None:
                self.logger.log_episode_metrics(metrics)
            return metrics
        finally:
            try:
                parent_conn.send(("close", {}))
            except (BrokenPipeError, EOFError, OSError):
                pass
            parent_conn.close()
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def _drain_and_release_results(
        self,
        *,
        pool: Any,
        coordinator: _RealtimeResultCoordinator,
        now_ms: float,
        action_events: list[ActionEvent],
        latency_values_ms: list[float],
    ) -> list[ActionEvent]:
        released_events: list[ActionEvent] = []
        for completion in pool.drain_completed(now_ms):
            ignored = coordinator.add_completion(completion)
            for ignored_completion in ignored:
                self._record_completion_event(
                    completion=ignored_completion,
                    action_events=action_events,
                    latency_values_ms=latency_values_ms,
                    dropped=True,
                    discard_reason="ignored_late_result",
                )

        released, discarded = coordinator.release_ready()
        for completion in discarded:
            self._record_completion_event(
                completion=completion,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
                dropped=True,
                discard_reason="latest_ready_discarded",
            )
        for completion in released:
            event = self._record_completion_event(
                completion=completion,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
                dropped=False,
                discard_reason=None,
            )
            self.action_scheduler.enqueue(event)
            released_events.append(event)
        return released_events

    def _record_completion_event(
        self,
        *,
        completion: RealtimePolicyCompletion,
        action_events: list[ActionEvent],
        latency_values_ms: list[float],
        dropped: bool,
        discard_reason: str | None,
    ) -> ActionEvent:
        latency_record = completion.latency_record
        latency_record.device = completion.device if latency_record.device is None else latency_record.device
        latency_record.metadata = {
            **latency_record.metadata,
            "worker_id": completion.worker_id,
        }
        apply_latency_record_raw_fields(
            record=latency_record,
            latency_type="measured",
            profile_metadata=profile_metadata_from_source(self.latency_source),
        )
        event_metadata = dict(completion.policy_output.metadata)
        event_metadata["frame_ms"] = self.clock.frame_ms
        if discard_reason is not None:
            event_metadata["discard_reason"] = discard_reason
        event = ActionEvent(
            action_id=self._next_action_id,
            action=completion.policy_output.action,
            obs_env_step=completion.task.observation.env_step,
            ready_time_ms=completion.completion_time_ms,
            ready_step_float=self.clock.time_ms_to_step_float(completion.completion_time_ms),
            latency=latency_record,
            metadata=event_metadata,
        )
        apply_latency_raw_fields(
            event=event,
            latency_type="measured",
            frame_ms=self.clock.frame_ms,
            profile_metadata=profile_metadata_from_source(self.latency_source),
        )
        event.dropped = bool(dropped)
        self._next_action_id += 1
        action_events.append(event)
        latency_values_ms.append(float(latency_record.latency_ms))
        if self.logger is not None:
            self.logger.log_action_event(event)
            self.logger.log_latency(latency_record)
        return event

    def _make_observation_task(self, *, env_step: int, submit_time_ms: float) -> RealtimeObservationTask:
        observation = self.env.observe()
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        return RealtimeObservationTask(
            obs_id=obs_id,
            observation=observation,
            submit_env_step=env_step,
            submit_time_ms=submit_time_ms,
        )

    def _episode_metadata(
        self,
        *,
        seed: int | None,
        step_records: list[StepRecord],
        submitted_observation_frames: int,
        dropped_observation_count: int,
        env_deadline_miss_count: int,
        max_wall_clock_lag_ms: float,
        coordinator: _RealtimeResultCoordinator,
        pool: Any,
        pre_episode_drain: dict[str, float | int] | None = None,
    ) -> dict[str, Any]:
        elapsed_s = self.time_fn() - self._episode_start_s
        actual_env_fps = None if elapsed_s <= 0 else len(step_records) / elapsed_s
        return {
            "submitted_observation_frames": submitted_observation_frames,
            "dropped_observation_count": dropped_observation_count,
            "ignored_late_result_count": coordinator.ignored_late_result_count,
            "discarded_ready_result_count": coordinator.discarded_ready_result_count,
            "fifo_blocked_count": coordinator.fifo_blocked_count,
            "target_env_fps": self.clock.env_fps,
            "actual_env_fps": actual_env_fps,
            "env_deadline_miss_count": env_deadline_miss_count,
            "max_wall_clock_lag_ms": max_wall_clock_lag_ms,
            "in_flight_count": pool.in_flight_count,
            "idle_worker_count": pool.idle_worker_count,
            **(pre_episode_drain or {}),
            **episode_raw_fact_metadata(
                mode="realtime",
                episode_seed=seed,
                env_fps=self.clock.env_fps,
                obs_fps=self.clock.obs_fps,
                frame_ms=self.clock.frame_ms,
                latency_type="measured",
            ),
        }

    def _wall_elapsed_ms(self) -> float:
        return (self.time_fn() - self._episode_start_s) * 1000.0


def _worker_runtime_metadata(worker_id: int, start_method: str) -> dict[str, Any]:
    import torch

    return {
        "worker_id": int(worker_id),
        "worker_pid": os.getpid(),
        "worker_start_method": start_method,
        "worker_python_executable": sys.executable,
        "worker_torch_version": torch.__version__,
        "worker_torch_cuda_version": torch.version.cuda,
        "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_cuda_device_count": torch.cuda.device_count(),
    }


def _action_event_dispatch_payload(event: ActionEvent) -> dict[str, Any]:
    return {
        "action_id": event.action_id,
        "action": event.action,
        "obs_env_step": event.obs_env_step,
        "ready_time_ms": event.ready_time_ms,
        "ready_step_float": event.ready_step_float,
        "latency_raw_frames": event.latency_raw_frames,
        "ready_raw_frame": event.ready_raw_frame,
    }


def _action_event_from_dispatch_payload(payload: dict[str, Any]) -> RealtimeActionDispatch:
    return RealtimeActionDispatch(
        action_id=payload["action_id"],
        action=payload["action"],
        obs_env_step=payload["obs_env_step"],
        ready_time_ms=payload["ready_time_ms"],
        ready_step_float=payload["ready_step_float"],
        latency_raw_frames=payload["latency_raw_frames"],
        ready_raw_frame=payload["ready_raw_frame"],
    )


def _action_event_ack_payload(event: RealtimeActionDispatch) -> dict[str, Any]:
    return {
        "action_id": event.action_id,
        "applied": event.applied,
        "apply_raw_frame": event.apply_raw_frame,
    }


def _inference_worker(connection: Any, config: dict[str, Any], worker_id: int, device: str, start_method: str) -> None:
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.policy.factory import build_policy

        runtime_metadata = _worker_runtime_metadata(worker_id, start_method)
        worker_config = deepcopy(config)
        worker_config.setdefault("policy", {})["device"] = device
        worker_config.setdefault("latency", {})["device"] = device
        action_resolver = ActionResolver(worker_config["env"])
        policy = build_policy(worker_config, action_resolver)
        latency_cfg = worker_config.get("latency", {})
        measurer = RuntimeLatencyMeasurer(
            device=device,
            sync_cuda=bool(latency_cfg.get("sync_cuda", True)),
        )
        connection.send(("ready", {"worker_id": int(worker_id), "device": device}))

        while True:
            command, payload = connection.recv()
            payload = payload or {}
            if command == "close":
                return
            if command == "predict":
                task = payload["task"]
                policy_output, latency_record = measurer.run_and_measure(
                    policy.predict,
                    task.observation,
                    obs_id=task.obs_id,
                )
                latency_record.metadata = {
                    **latency_record.metadata,
                    **runtime_metadata,
                }
                result = RealtimePolicyResult(
                    policy_output=policy_output,
                    completion_time_ms=task.submit_time_ms + latency_record.latency_ms,
                    latency_record=latency_record,
                )
                try:
                    connection.send(("result", {"result": result}))
                except (BrokenPipeError, EOFError, OSError):
                    return
    except BaseException:
        try:
            connection.send(("error", {"traceback": traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError):
            return


def _env_loop_worker(
    connection: Any,
    config: dict[str, Any] | None,
    episode_id: int,
    seed: int | None,
    eval_max_steps: int,
    video_output_dir: str | None,
    video_fps: float | None,
) -> None:
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.envs.factory import build_env_adapter
        from latency_bench.logging.video import EpisodeVideoRecorder
        from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler

        runtime_config = config or {}
        action_resolver = ActionResolver(runtime_config["env"])
        env = build_env_adapter(runtime_config, action_resolver.noop_action)
        clock = EnvClock(
            env_fps=float(runtime_config["env"]["env_fps"]),
            obs_fps=float(runtime_config["env"]["obs_fps"]),
        )
        action_scheduler = ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=_build_realtime_persistence_policy(runtime_config.get("scheduler", {}), clock),
            ordering_policy=OrderingPolicy(str(runtime_config.get("scheduler", {}).get("ordering_policy", "issue_order_fifo"))),
        )
        decision_scheduler = ObservationCadenceDecisionScheduler(clock)
        env.reset(seed=seed)
        video_recorder = None
        if video_output_dir is not None:
            video_recorder = EpisodeVideoRecorder(video_output_dir, fps=video_fps)
            video_recorder.start_episode(episode_id=episode_id, metadata={})
            video_recorder.record_frame(env.render_game_frame())
        episode_start_s = time.perf_counter()
        next_obs_id = 0
        applied_event_count = 0

        for env_step in range(int(eval_max_steps)):
            scheduled_time_ms = clock.step_to_time_ms(env_step)
            target_wall_s = episode_start_s + scheduled_time_ms / 1000.0
            now_s = time.perf_counter()
            if now_s < target_wall_s:
                time.sleep(target_wall_s - now_s)
            wall_elapsed_ms = (time.perf_counter() - episode_start_s) * 1000.0
            wall_clock_lag_ms = max(0.0, wall_elapsed_ms - scheduled_time_ms)

            while connection.poll():
                command, payload = connection.recv()
                payload = payload or {}
                if command == "close":
                    env.close()
                    return
                if command == "action_dispatch":
                    action_scheduler.enqueue(_action_event_from_dispatch_payload(payload))

            if decision_scheduler.should_observe(env_step, scheduled_time_ms):
                observation = env.observe()
                task = RealtimeObservationTask(
                    obs_id=next_obs_id,
                    observation=observation,
                    submit_env_step=env_step,
                    submit_time_ms=wall_elapsed_ms,
                )
                next_obs_id += 1
                connection.send(("observation", {"task": task}))

            active_action = action_scheduler.update(env_step, scheduled_time_ms)
            applied_events = action_scheduler.get_applied_events()
            for event in applied_events[applied_event_count:]:
                connection.send(("action_ack", _action_event_ack_payload(event)))
            applied_event_count = len(applied_events)

            result = env.step(active_action)
            episode_done = bool(result.done or result.truncated)
            if video_recorder is not None and not episode_done:
                video_recorder.record_frame(env.render_game_frame())
            step_info = step_record_info(result.info)
            step_info.update(
                {
                    "scheduled_time_ms": scheduled_time_ms,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    "frame_time_ms": clock.frame_ms,
                    "wall_clock_lag_ms": wall_clock_lag_ms,
                    "observation_submitted": False,
                    "observation_dropped": False,
                    "in_flight_count": None,
                    "idle_worker_count": None,
                }
            )
            step_info.update(
                active_action_step_info(
                    applied_events=applied_events,
                    env_step=env_step,
                    scheduled_time_ms=scheduled_time_ms,
                    frame_ms=clock.frame_ms,
                    latency_type="measured",
                    profile_metadata={},
                )
            )
            connection.send((
                "step",
                {
                    "record": StepRecord(
                        episode_id=episode_id,
                        env_step=env_step,
                        sim_time_ms=scheduled_time_ms,
                        active_action=active_action,
                        reward=float(result.reward),
                        done=episode_done,
                        info=step_info,
                    )
                },
            ))
            if episode_done:
                break

        if video_recorder is not None:
            video_recorder.close_episode({})
        env.close()
        connection.send(("episode_done", {}))
    except BaseException:
        connection.send(("error", {"traceback": traceback.format_exc()}))


def _build_realtime_persistence_policy(scheduler_cfg: dict[str, Any], clock: EnvClock):
    mode = ActionHoldPolicy(str(scheduler_cfg.get("hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value)))
    if mode == ActionHoldPolicy.HOLD:
        return HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames)
    return OneFrameThenNoopPolicy()
