from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import (
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


class ProcessInferencePool:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        inference_devices: list[str],
        start_method: str | None = None,
    ):
        _set_thread_env_defaults()
        self.config = config
        self.inference_devices = list(inference_devices)
        self.start_method = _resolve_start_method(start_method)
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
        self._drain_all_ready()

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
        terminate = False
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", {}))
                except (BrokenPipeError, EOFError, OSError):
                    terminate = True
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self._processes:
            process.join(timeout=0.2)
        if terminate:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            for process in self._processes:
                process.join(timeout=1.0)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    def _start_workers(self) -> None:
        try:
            for worker_id, device in enumerate(self.inference_devices):
                parent_conn, child_conn = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=_inference_worker,
                    args=(child_conn, self.config, worker_id, device),
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
                    self._busy_worker_tasks.pop(worker_id, None)
                    completed.append(payload["completion"])
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
        self.action_scheduler.reset()
        self.env.reset(seed=seed)
        if self.video_recorder is not None:
            self.video_recorder.start_episode(episode_id=episode_id, metadata={})
            self.video_recorder.record_frame(self.env.render_game_frame())
        coordinator = _RealtimeResultCoordinator(ordering_policy=self.action_scheduler.ordering_policy)
        pool.reset_episode()
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
        pool.reset_episode()
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
            while not done:
                now_ms = self._wall_elapsed_ms()
                released = self._drain_and_release_results(
                    pool=pool,
                    coordinator=coordinator,
                    now_ms=now_ms,
                    action_events=action_events,
                    latency_values_ms=latency_values_ms,
                )
                for event in released:
                    parent_conn.send(("action_event", {"event": event}))
                    action_event_index[event.action_id] = action_events.index(event)

                if not parent_conn.poll(0.001):
                    if not process.is_alive():
                        done = True
                    continue

                tag, payload = parent_conn.recv()
                if tag == "observation":
                    task = payload["task"]
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
                elif tag == "applied_action_event":
                    event = payload["event"]
                    index = action_event_index.get(event.action_id)
                    if index is not None:
                        target = action_events[index]
                        target.applied = event.applied
                        target.apply_raw_frame = event.apply_raw_frame
                        target.ready_raw_frame = event.ready_raw_frame
                        target.latency_raw_frames = event.latency_raw_frames
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


def _inference_worker(connection: Any, config: dict[str, Any], worker_id: int, device: str) -> None:
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.policy.factory import build_policy

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
                    "worker_id": int(worker_id),
                }
                completion = RealtimePolicyCompletion(
                    task=task,
                    policy_output=policy_output,
                    completion_time_ms=task.submit_time_ms + latency_record.latency_ms,
                    latency_record=latency_record,
                    worker_id=int(worker_id),
                    device=device,
                )
                connection.send(("result", {"completion": completion}))
    except BaseException:
        connection.send(("error", {"traceback": traceback.format_exc()}))


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
                if command == "action_event":
                    action_scheduler.enqueue(payload["event"])

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
                connection.send(("applied_action_event", {"event": event}))
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
