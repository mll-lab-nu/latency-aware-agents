"""The realtime executor: drives episodes against a wall clock while a process
pool runs inference, coordinating in-flight results back into the action schedule.

Both episode paths (env-in-process and env-in-subprocess) accumulate into a shared
:class:`_EpisodeAccumulators` and converge on :meth:`_finalize_episode`, so the
per-path code is just the two genuinely different control loops.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import (
    ActionEvent,
    EpisodeMetrics,
    StepRecord,
)
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors.base import EpisodeExecutor
from latency_bench.executors.env_step_backend import _resolve_start_method
from latency_bench.executors import realtime_warmup
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
from latency_bench.logging.realtime_pipeline import PipelineProfile, REALTIME_PROFILE_METADATA_KEY
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler
from latency_bench.executors.realtime.pool import ProcessInferencePool, _RealtimeResultCoordinator
from latency_bench.executors.realtime.workers import (
    RealtimeObservationTask,
    RealtimePolicyCompletion,
    _action_event_dispatch_payload,
    _env_loop_worker,
    _profile_now_ms,
)


@dataclass
class _EpisodeAccumulators:
    """Per-episode tallies shared by both run-episode paths."""
    step_records: list[StepRecord] = field(default_factory=list)
    action_events: list[ActionEvent] = field(default_factory=list)
    latency_values_ms: list[float] = field(default_factory=list)
    submitted_observation_frames: int = 0
    dropped_observation_count: int = 0
    env_deadline_miss_count: int = 0
    max_wall_clock_lag_ms: float = 0.0


class RealtimeLatencyExecutor(EpisodeExecutor):
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
        self._profile_pipeline = bool((config or {}).get("logging", {}).get("realtime_pipeline_profile", False))
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

    def _finalize_episode(
        self,
        *,
        episode_id: int,
        seed: int | None,
        acc: _EpisodeAccumulators,
        coordinator: _RealtimeResultCoordinator,
        pool: Any,
        pre_episode_drain: dict[str, float | int] | None,
        enqueue_released_actions: bool,
        release_observation: Callable[[int | None], None] | None,
    ) -> EpisodeMetrics:
        self._drain_and_release_results(
            pool=pool,
            coordinator=coordinator,
            now_ms=self._wall_elapsed_ms(),
            action_events=acc.action_events,
            latency_values_ms=acc.latency_values_ms,
            enqueue_released_actions=enqueue_released_actions,
            release_observation=release_observation,
        )
        elapsed_s = self.time_fn() - self._episode_start_s
        actual_env_fps = None if elapsed_s <= 0 else len(acc.step_records) / elapsed_s
        metadata = {
            "submitted_observation_frames": acc.submitted_observation_frames,
            "dropped_observation_count": acc.dropped_observation_count,
            "ignored_late_result_count": coordinator.ignored_late_result_count,
            "discarded_ready_result_count": coordinator.discarded_ready_result_count,
            "fifo_blocked_count": coordinator.fifo_blocked_count,
            "target_env_fps": self.clock.env_fps,
            "actual_env_fps": actual_env_fps,
            "env_deadline_miss_count": acc.env_deadline_miss_count,
            "max_wall_clock_lag_ms": acc.max_wall_clock_lag_ms,
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
        metrics = compute_episode_metrics(
            episode_id=episode_id,
            step_records=acc.step_records,
            action_events=acc.action_events,
            latency_values_ms=acc.latency_values_ms,
            metadata=metadata,
            frame_ms=self.clock.frame_ms,
        )
        if self.logger is not None:
            self.logger.log_episode_metrics(metrics)
        return metrics

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

        acc = _EpisodeAccumulators()
        applied_event_count = 0

        for env_step in range(int(eval_max_steps)):
            scheduled_time_ms = self.clock.step_to_time_ms(env_step)
            target_wall_s = self._episode_start_s + scheduled_time_ms / 1000.0
            now_s = self.time_fn()
            if now_s < target_wall_s:
                self.sleep_fn(target_wall_s - now_s)
            current_wall_elapsed_ms = self._wall_elapsed_ms()
            wall_clock_lag_ms = max(0.0, current_wall_elapsed_ms - scheduled_time_ms)
            if wall_clock_lag_ms > self.clock.frame_ms:
                acc.env_deadline_miss_count += 1
            acc.max_wall_clock_lag_ms = max(acc.max_wall_clock_lag_ms, wall_clock_lag_ms)

            self._drain_and_release_results(
                pool=pool,
                coordinator=coordinator,
                now_ms=current_wall_elapsed_ms,
                action_events=acc.action_events,
                latency_values_ms=acc.latency_values_ms,
                enqueue_released_actions=True,
                release_observation=None,
            )

            observation_submitted = False
            observation_dropped = False
            if self.decision_scheduler.should_observe(env_step, scheduled_time_ms):
                task = self._make_observation_task(
                    env_step=env_step,
                    submit_time_ms=current_wall_elapsed_ms,
                    wall_clock_lag_ms=wall_clock_lag_ms,
                )
                if pool.submit(task):
                    if task.profile is not None:
                        task.profile.parent_observation_handle_ms = (
                            _profile_now_ms() - task.profile.parent_observation_handle_start_ms
                        )
                    coordinator.mark_submitted(task)
                    acc.submitted_observation_frames += 1
                    observation_submitted = True
                else:
                    acc.dropped_observation_count += 1
                    observation_dropped = True

            scheduler_update_start_ms = _profile_now_ms()
            active_action = self.action_scheduler.update(env_step, scheduled_time_ms)
            scheduler_update_ms = _profile_now_ms() - scheduler_update_start_ms
            new_applied_events = self.action_scheduler.get_applied_events_since(applied_event_count)
            for event in new_applied_events:
                profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                if profile is not None:
                    profile.env_scheduler_update_ms = scheduler_update_ms

            env_step_start_ms = _profile_now_ms()
            result = self.env.step(active_action)
            env_step_ms = _profile_now_ms() - env_step_start_ms
            for event in new_applied_events:
                profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                if profile is not None:
                    profile.env_step_ms = env_step_ms
            applied_event_count = self.action_scheduler.applied_event_count
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
                    active_event=self.action_scheduler.latest_applied_event,
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
            acc.step_records.append(record)
            if self.logger is not None:
                self.logger.log_step(record)
            if episode_done:
                break

        metrics = self._finalize_episode(
            episode_id=episode_id,
            seed=seed,
            acc=acc,
            coordinator=coordinator,
            pool=pool,
            pre_episode_drain=pre_episode_drain,
            enqueue_released_actions=True,
            release_observation=None,
        )
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
        acc = _EpisodeAccumulators()
        action_event_index: dict[int, int] = {}
        pending_step_info: dict[int, dict[str, Any]] = {}
        done = False

        try:
            def release_observation(obs_id: int | None) -> None:
                nonlocal done
                try:
                    parent_conn.send(("release_observation", {"obs_id": obs_id}))
                except BrokenPipeError:
                    done = True

            def release_ready_actions(now_ms: float) -> None:
                nonlocal done
                released = self._drain_and_release_results(
                    pool=pool,
                    coordinator=coordinator,
                    now_ms=now_ms,
                    action_events=acc.action_events,
                    latency_values_ms=acc.latency_values_ms,
                    enqueue_released_actions=False,
                    release_observation=release_observation,
                )
                first_released_index = len(acc.action_events) - len(released)
                for offset, event in enumerate(released):
                    profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                    if profile is not None:
                        profile.parent_to_env_send_start_ms = _profile_now_ms()
                    try:
                        parent_conn.send(("action_dispatch", _action_event_dispatch_payload(event)))
                    except BrokenPipeError:
                        done = True
                        break
                    if profile is not None:
                        profile.parent_to_env_send_call_ms = (
                            _profile_now_ms() - profile.parent_to_env_send_start_ms
                        )
                    action_event_index[event.action_id] = first_released_index + offset

            def handle_env_message(tag: str, payload: dict[str, Any]) -> None:
                nonlocal done
                if tag == "observation":
                    parent_observation_recv_ms = _profile_now_ms()
                    release_ready_actions(self._wall_elapsed_ms())
                    if done:
                        return
                    incoming_task = payload["task"]
                    profile = incoming_task.profile
                    if profile is not None:
                        profile.parent_observation_recv_ms = parent_observation_recv_ms
                        profile.env_to_parent_ipc_ms = (
                            parent_observation_recv_ms - profile.env_observation_send_start_ms
                        )
                        profile.parent_observation_handle_start_ms = _profile_now_ms()
                    task = RealtimeObservationTask(
                        obs_id=self._next_obs_id,
                        observation=incoming_task.observation,
                        submit_env_step=incoming_task.submit_env_step,
                        submit_time_ms=incoming_task.submit_time_ms,
                        profile=profile,
                        shared_obs_id=incoming_task.shared_obs_id,
                    )
                    self._next_obs_id += 1
                    if pool.submit(task):
                        if task.profile is not None:
                            task.profile.parent_observation_handle_ms = (
                                _profile_now_ms() - task.profile.parent_observation_handle_start_ms
                            )
                        coordinator.mark_submitted(task)
                        acc.submitted_observation_frames += 1
                        pending_step_info[int(task.submit_env_step)] = {
                            "observation_submitted": True,
                            "observation_dropped": False,
                            "in_flight_count": pool.in_flight_count,
                            "idle_worker_count": pool.idle_worker_count,
                        }
                    else:
                        acc.dropped_observation_count += 1
                        release_observation(incoming_task.shared_obs_id)
                        pending_step_info[int(task.submit_env_step)] = {
                            "observation_submitted": False,
                            "observation_dropped": True,
                            "in_flight_count": pool.in_flight_count,
                            "idle_worker_count": pool.idle_worker_count,
                        }
                elif tag == "action_ack":
                    index = action_event_index.get(payload["action_id"])
                    if index is not None:
                        target = acc.action_events[index]
                        target.applied = payload["applied"]
                        target.apply_raw_frame = payload["apply_raw_frame"]
                        if "profile" in payload:
                            target.metadata[REALTIME_PROFILE_METADATA_KEY].merge_dict(payload["profile"])
                elif tag == "step":
                    record = payload["record"]
                    annotation = pending_step_info.pop(int(record.env_step), None)
                    if annotation is not None:
                        record.info.update(annotation)
                    acc.max_wall_clock_lag_ms = max(acc.max_wall_clock_lag_ms, float(record.info.get("wall_clock_lag_ms", 0.0)))
                    if float(record.info.get("wall_clock_lag_ms", 0.0)) > self.clock.frame_ms:
                        acc.env_deadline_miss_count += 1
                    acc.step_records.append(record)
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

            metrics = self._finalize_episode(
                episode_id=episode_id,
                seed=seed,
                acc=acc,
                coordinator=coordinator,
                pool=pool,
                pre_episode_drain=pre_episode_drain,
                enqueue_released_actions=False,
                release_observation=release_observation,
            )
            self._release_in_flight_observations(
                pool=pool,
                release_observation=release_observation,
            )
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

    def _release_in_flight_observations(
        self,
        *,
        pool: Any,
        release_observation: Callable[[int | None], None],
    ) -> None:
        while pool.in_flight_count:
            for completion in pool.drain_completed(self._wall_elapsed_ms()):
                release_observation(completion.task.shared_obs_id)
            if pool.in_flight_count:
                self.sleep_fn(0.001)

    def _drain_and_release_results(
        self,
        *,
        pool: Any,
        coordinator: _RealtimeResultCoordinator,
        now_ms: float,
        action_events: list[ActionEvent],
        latency_values_ms: list[float],
        enqueue_released_actions: bool,
        release_observation: Callable[[int | None], None] | None,
    ) -> list[ActionEvent]:
        released_events: list[ActionEvent] = []
        for completion in pool.drain_completed(now_ms):
            if release_observation is not None:
                release_observation(completion.task.shared_obs_id)
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
            if enqueue_released_actions:
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
        release_start_ms = _profile_now_ms()
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
        if completion.profile is not None:
            event_metadata[REALTIME_PROFILE_METADATA_KEY] = completion.profile
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
        profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
        if profile is not None:
            profile.parent_release_ms = _profile_now_ms() - release_start_ms
        self._next_action_id += 1
        action_events.append(event)
        latency_values_ms.append(float(latency_record.latency_ms))
        if self.logger is not None:
            self.logger.log_action_event(event)
            self.logger.log_latency(latency_record)
        return event

    def _make_observation_task(
        self,
        *,
        env_step: int,
        submit_time_ms: float,
        wall_clock_lag_ms: float = 0.0,
    ) -> RealtimeObservationTask:
        observe_start_ms = _profile_now_ms()
        observation = self.env.observe()
        observe_end_ms = _profile_now_ms()
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        profile = None
        if self._profile_pipeline:
            # Local-env path: the env shares this process, so the env<->parent
            # IPC stages are genuinely zero (a real measurement, not a fallback).
            profile = PipelineProfile(
                env_observe_ms=observe_end_ms - observe_start_ms,
                env_wall_lag_ms=wall_clock_lag_ms,
                env_to_parent_ipc_ms=0.0,
                parent_observation_handle_start_ms=_profile_now_ms(),
                parent_to_env_ipc_ms=0.0,
                env_action_recv_to_enqueue_ms=0.0,
            )
        return RealtimeObservationTask(
            obs_id=obs_id,
            observation=observation,
            submit_env_step=env_step,
            submit_time_ms=submit_time_ms,
            profile=profile,
        )

    def _wall_elapsed_ms(self) -> float:
        return (self.time_fn() - self._episode_start_s) * 1000.0
