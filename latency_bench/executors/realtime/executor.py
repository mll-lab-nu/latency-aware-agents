"""The realtime executor: drives episodes against a wall clock while a process
pool runs inference, coordinating in-flight results back into the action schedule.

Production runtimes own a persistent environment subprocess. Injected local
runtimes support deterministic clock tests; both paths share episode accounting.
"""
from __future__ import annotations

import gc
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.core.types import (
    ActionEvent,
    EpisodeMetrics,
    Observation,
    StepRecord,
)
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors import realtime_warmup
from latency_bench.executors.base import EpisodeExecutor
from latency_bench.executors.env_step_backend import env_action_space
from latency_bench.executors.realtime.env_process import RealtimeEnvProcess
from latency_bench.executors.realtime.messages import (
    RealtimeObservationTask,
    RealtimePolicyCompletion,
    _action_event_dispatch_payload,
)
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.executors.realtime.shared_observation import ObservationSlotPool
from latency_bench.executors.realtime.timing import monotonic_ms
from latency_bench.logging.metrics import (
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
    PROFILE_METADATA_KEYS,
)
from latency_bench.logging.records import build_step_record
from latency_bench.logging.realtime_pipeline import PipelineProfile, REALTIME_PROFILE_METADATA_KEY
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.action_prefix import with_action_prefix
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler
from latency_bench.scheduler.result_coordinator import ResultCoordinator


@contextmanager
def _gc_paused_until_episode_end():
    """Keep automatic GC out of the paced loop.

    Gen-2 collections pause the whole process for a time proportional to the
    live-object count, which is long enough to delay action delivery by whole
    frames. Collection instead happens here, at the episode boundary, where
    there is no wall-clock constraint; freezing the survivors keeps them out
    of future scans.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        gc.freeze()
        if was_enabled:
            gc.enable()


@dataclass
class _EpisodeAccumulators:
    """Per-episode tallies shared by both run-episode paths."""
    step_records: list[StepRecord] = field(default_factory=list)
    action_events: list[ActionEvent] = field(default_factory=list)
    latency_values_ms: list[float] = field(default_factory=list)
    observation_attempts: dict[int, dict[str, Any]] = field(default_factory=dict)
    submitted_observation_frames: int = 0
    dropped_observation_count: int = 0
    env_deadline_miss_count: int = 0
    max_wall_clock_lag_ms: float = 0.0


@dataclass(frozen=True)
class RealtimeRuntime:
    """Environment and scheduling dependencies for wall-clock execution."""
    env: EnvAdapter | RealtimeEnvProcess
    decision_scheduler: DecisionScheduler
    action_scheduler: ActionScheduler
    clock: EnvClock


@dataclass
class _DecisionWindow:
    """One submitted or dropped decision awaiting its output and final applied action."""
    admitted: bool
    issued: Any = None
    applied: Any = None


class _RealtimeDecisionHistory:
    """Join single-worker results to their original completed decision windows."""

    def __init__(self, env, clock, decisions):
        self.history = DecisionActionHistory(env_action_space(env), num_envs=1, decisions=decisions)
        self.stride = clock.obs_stride_raw_frames
        self.noop = env.noop_action.value
        self.windows: dict[int, _DecisionWindow] = {}

    def reset(self):
        self.history.reset()
        self.windows.clear()

    def finish_frame(self, frame, action):
        if (frame + 1) % self.stride == 0:
            self.windows[frame + 1 - self.stride].applied = action

    def observation(self):
        # A busy worker can span several windows. Its command belongs to the
        # source window; flush that row and the intervening drops in order.
        while self.windows:
            frame = next(iter(self.windows))
            window = self.windows[frame]
            if window.applied is None or (window.admitted and window.issued is None):
                break
            self.history.append([0], [window.admitted],
                                [window.issued if window.admitted else self.noop], [window.applied])
            del self.windows[frame]
        return self.history.observation()[0]


class RealtimeLatencyExecutor(EpisodeExecutor):
    """Coordinate environment stepping with asynchronous worker inference."""
    def __init__(
        self,
        runtime: RealtimeRuntime,
        logger: TrajectoryLogger | None = None,
        time_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
        inference_pool: Any | None = None,
        config: dict[str, Any] | None = None,
        inference_devices: list[str] | None = None,
        video_recorder: Any | None = None,
        action_prefix=None,
    ):
        self._install_runtime(runtime)
        self._profile_metadata = {key: None for key in PROFILE_METADATA_KEYS}
        self.logger = logger
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.inference_pool = inference_pool
        self.config = config
        self.action_prefix = action_prefix
        self._profile_pipeline = bool((config or {}).get("logging", {}).get("realtime_pipeline_profile", False))
        self.inference_devices = list(inference_devices or [])
        self.use_env_process = isinstance(runtime.env, RealtimeEnvProcess)
        self.video_recorder = video_recorder
        self._owns_inference_pool = inference_pool is None
        self._episode_start_s = 0.0
        self._next_obs_id = 0
        self._next_action_id = 0
        self._decision_history = None
        if config is not None and "env" in config and "action_history_decisions" in config["env"]:
            decisions = config["env"]["action_history_decisions"]
            if decisions is not None:
                self._decision_history = _RealtimeDecisionHistory(self.env, self.clock, decisions)

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        pool = self._ensure_inference_pool()
        if self._decision_history is not None:
            self._decision_history.reset()
        with _gc_paused_until_episode_end():
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

    def run_warmup(self, *, artifact_dir: str | Path | None = None) -> dict[str, Any]:
        with _gc_paused_until_episode_end():
            return self._run_warmup_impl(artifact_dir)

    def _run_warmup_impl(self, artifact_dir: str | Path | None) -> dict[str, Any]:
        pool = self._ensure_inference_pool()
        pool.drain_until_idle()
        pool.reset_state(0)
        if self._decision_history is not None:
            self._decision_history.reset()
        self.action_scheduler.reset()
        self.env.reset(seed=None)
        latency_records, profiles = [], []
        submitted_count = 0
        sample_target = realtime_warmup.MAX_WARMUP_LATENCY_SAMPLES
        observation_pool = None
        if not self.use_env_process:
            template = self._with_episode_metadata(
                with_action_prefix(self.env.observe(), self.action_scheduler, self.action_prefix),
                episode_id=-1, episode_seed=0,
            )
            observation_pool = ObservationSlotPool.from_observation(template, capacity=pool.worker_count)
        try:
            while len(latency_records) < sample_target:
                while pool.idle_worker_count and submitted_count < sample_target:
                    if self.use_env_process:
                        incoming = self.env.warmup_observation()
                        received = monotonic_ms()
                        incoming.profile.env_to_parent_ipc_ms = received - incoming.profile.env_observation_send_start_ms
                        task = RealtimeObservationTask(
                            obs_id=self._next_obs_id,
                            observation=self._with_episode_metadata(incoming.observation, episode_id=-1, episode_seed=0),
                            submit_env_step=incoming.submit_env_step, submit_time_ms=0.0,
                            observation_start_time_ms=incoming.observation_start_time_ms,
                            profile=incoming.profile, shared_obs_id=incoming.shared_obs_id,
                        )
                    else:
                        observe_start = monotonic_ms()
                        observation = self._with_episode_metadata(
                            with_action_prefix(self.env.observe(), self.action_scheduler, self.action_prefix),
                            episode_id=-1, episode_seed=0,
                        )
                        shared = observation_pool.write(obs_id=self._next_obs_id, observation=observation)
                        task = RealtimeObservationTask(
                            obs_id=self._next_obs_id, observation=shared,
                            submit_env_step=observation.env_step, submit_time_ms=0.0,
                            observation_start_time_ms=observe_start,
                            shared_obs_id=self._next_obs_id,
                            profile=PipelineProfile(
                                env_observe_ms=monotonic_ms() - observe_start,
                                env_wall_lag_ms=0.0, env_to_parent_ipc_ms=0.0,
                                parent_to_env_ipc_ms=0.0, env_action_recv_to_enqueue_ms=0.0,
                            ),
                        )
                    self._next_obs_id += 1
                    task.profile.parent_observation_handle_start_ms = monotonic_ms()
                    pool.submit(task)
                    task.profile.parent_observation_handle_ms = monotonic_ms() - task.profile.parent_observation_handle_start_ms
                    submitted_count += 1
                completions = pool.drain_completed()
                for completion in completions:
                    if self.use_env_process:
                        self.env.release_observation(completion.task.shared_obs_id)
                    else:
                        observation_pool.release(completion.task.shared_obs_id)
                    action = completion.policy_output.action
                    start = monotonic_ms()
                    result = self.env.step(action)
                    step_ms = result.info["env_step_ms"] if self.use_env_process else monotonic_ms() - start
                    if self._decision_history is not None:
                        self._decision_history.history.append([0], [True], [action.value], [action.value])
                    latency_records.append(completion.latency_record)
                    if completion.profile is not None:
                        completion.profile.env_step_ms = step_ms
                        profiles.append(completion.profile)
                    if result.done or result.truncated:
                        self.env.reset(seed=None)
                        if self._decision_history is not None:
                            self._decision_history.reset()
                if not completions:
                    self.sleep_fn(0.001)
        finally:
            if observation_pool is not None:
                observation_pool.close()
        output_dir = self.config["logging"]["output_dir"] if artifact_dir is None else artifact_dir
        summary = realtime_warmup.write_realtime_warmup_artifacts(output_dir, latency_records, profiles)
        if summary["warmup_status"] != "stable":
            print(f"WARNING: realtime pipeline warmup did not stabilize; inspect {output_dir}.",
                  file=sys.stderr, flush=True)
        return summary

    def close(self) -> None:
        self._close_runtime(close_inference_pool=self._owns_inference_pool)

    def _close_runtime(self, *, close_inference_pool: bool) -> None:
        if close_inference_pool and self.inference_pool is not None:
            self.inference_pool.close()
        self.inference_pool = None
        self.env.close()

    def _install_runtime(self, runtime: RealtimeRuntime) -> None:
        self.env = runtime.env
        self.decision_scheduler = runtime.decision_scheduler
        self.action_scheduler = runtime.action_scheduler
        self.clock = runtime.clock
        self.inference_pool = None

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
        coordinator: ResultCoordinator[RealtimePolicyCompletion],
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
            observation_attempts=acc.observation_attempts,
            enqueue_released_actions=enqueue_released_actions,
            release_observation=release_observation,
        )
        elapsed_s = self.time_fn() - self._episode_start_s
        actual_env_fps = None if elapsed_s <= 0 else len(acc.step_records) / elapsed_s
        in_flight_count = pool.in_flight_count
        idle_worker_count = pool.idle_worker_count
        if self.logger is not None:
            self._drain_in_flight_observation_attempts(
                pool=pool,
                observation_attempts=acc.observation_attempts,
                release_observation=release_observation,
            )
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
            "in_flight_count": in_flight_count,
            "idle_worker_count": idle_worker_count,
            "inference_worker_count": pool.worker_count,
            **self.action_scheduler.chunk_metrics(),
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
            self.logger.log_observation_attempts(acc.observation_attempts.values())
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
        pool.reset_state(0)
        self.action_scheduler.reset()
        self.env.reset(seed=seed)
        if self.video_recorder is not None:
            self.video_recorder.start_episode(episode_id=episode_id, metadata={})
            self.video_recorder.record_frame(self.env.render_game_frame())
        coordinator = ResultCoordinator(ordering_policy=self.action_scheduler.ordering_policy)
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
                observation_attempts=acc.observation_attempts,
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
                    episode_id=episode_id,
                    episode_seed=seed,
                )
                submitted = pool.submit(task)
                self._record_observation_attempt(
                    acc=acc,
                    episode_id=episode_id,
                    task=task,
                    submitted=submitted,
                )
                if submitted:
                    if task.profile is not None:
                        task.profile.parent_observation_handle_ms = (
                            monotonic_ms() - task.profile.parent_observation_handle_start_ms
                        )
                    coordinator.mark_submitted(int(task.obs_id))
                    acc.submitted_observation_frames += 1
                    observation_submitted = True
                else:
                    acc.dropped_observation_count += 1
                    observation_dropped = True

            scheduler_update_start_ms = monotonic_ms()
            active_action = self.action_scheduler.update(env_step, scheduled_time_ms)
            scheduler_update_ms = monotonic_ms() - scheduler_update_start_ms
            new_applied_events = self.action_scheduler.get_applied_events_since(applied_event_count)
            for event in new_applied_events:
                profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                if profile is not None:
                    profile.env_scheduler_update_ms = scheduler_update_ms

            env_step_start_ms = monotonic_ms()
            result = self.env.step(active_action)
            if self._decision_history is not None:
                self._decision_history.finish_frame(env_step, active_action.value)
            env_step_ms = monotonic_ms() - env_step_start_ms
            result.info["env_step_ms"] = env_step_ms
            for event in new_applied_events:
                profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                if profile is not None:
                    profile.env_step_ms = env_step_ms
            applied_event_count = self.action_scheduler.applied_event_count
            episode_done = result.done or result.truncated
            if self.video_recorder is not None and not episode_done:
                self.video_recorder.record_frame(self.env.render_game_frame())
            record = build_step_record(
                episode_id=episode_id,
                env_step=env_step,
                scheduled_time_ms=scheduled_time_ms,
                active_action=active_action,
                reward=result.reward,
                done=episode_done,
                info=result.info,
                active_event=self.action_scheduler.latest_applied_event,
                frame_ms=self.clock.frame_ms,
                latency_type="measured",
                pre_active_info={
                    "scheduled_time_ms": scheduled_time_ms,
                    "wall_elapsed_ms": current_wall_elapsed_ms,
                    "frame_time_ms": self.clock.frame_ms,
                    "wall_clock_lag_ms": wall_clock_lag_ms,
                    "observation_submitted": observation_submitted,
                    "observation_dropped": observation_dropped,
                    "in_flight_count": pool.in_flight_count,
                    "idle_worker_count": pool.idle_worker_count,
                },
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
        pool.reset_state(0)
        parent_conn = self.env.connection
        process = self.env.process
        self.action_scheduler.reset()
        self._episode_start_s = self.env.start_episode(
            episode_id=episode_id, seed=seed, eval_max_steps=eval_max_steps,
        )
        coordinator = ResultCoordinator(ordering_policy=self.action_scheduler.ordering_policy)
        acc = _EpisodeAccumulators()
        action_event_index: dict[int, int] = {}
        pending_step_info: dict[int, dict[str, Any]] = {}
        done = False

        try:
            def release_observation(obs_id: int | None) -> None:
                parent_conn.send(("release_observation", {"obs_id": obs_id}))

            def release_ready_actions(now_ms: float) -> None:
                nonlocal done
                released = self._drain_and_release_results(
                    pool=pool,
                    coordinator=coordinator,
                    now_ms=now_ms,
                    action_events=acc.action_events,
                    latency_values_ms=acc.latency_values_ms,
                    observation_attempts=acc.observation_attempts,
                    enqueue_released_actions=False,
                    release_observation=release_observation,
                )
                first_released_index = len(acc.action_events) - len(released)
                for offset, event in enumerate(released):
                    profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
                    if profile is not None:
                        profile.parent_to_env_send_start_ms = monotonic_ms()
                    parent_conn.send(("action_dispatch", _action_event_dispatch_payload(event)))
                    if profile is not None:
                        profile.parent_to_env_send_call_ms = (
                            monotonic_ms() - profile.parent_to_env_send_start_ms
                        )
                    action_event_index[event.action_id] = first_released_index + offset

            def handle_env_message(tag: str, payload: dict[str, Any]) -> None:
                nonlocal done
                if tag == "observation":
                    parent_observation_recv_ms = monotonic_ms()
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
                        profile.parent_observation_handle_start_ms = monotonic_ms()
                    task = RealtimeObservationTask(
                        obs_id=self._next_obs_id,
                        observation=self._with_episode_metadata(
                            incoming_task.observation,
                            episode_id=episode_id,
                            episode_seed=seed,
                        ),
                        submit_env_step=incoming_task.submit_env_step,
                        submit_time_ms=incoming_task.submit_time_ms,
                        observation_start_time_ms=(
                            incoming_task.observation_start_time_ms
                        ),
                        profile=profile,
                        shared_obs_id=incoming_task.shared_obs_id,
                    )
                    self._next_obs_id += 1
                    submitted = pool.submit(task)
                    self._record_observation_attempt(
                        acc=acc,
                        episode_id=episode_id,
                        task=task,
                        submitted=submitted,
                    )
                    if submitted:
                        if task.profile is not None:
                            task.profile.parent_observation_handle_ms = (
                                monotonic_ms() - task.profile.parent_observation_handle_start_ms
                            )
                        coordinator.mark_submitted(int(task.obs_id))
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
                        for key in (
                            "hold_last_chunk_action",
                            "stale_prefix_frames",
                            "chunk_age_raw_frames",
                            "chunk_age_ms",
                            "underflow_count",
                            "underflow_frames",
                            "underflow_duration_ms",
                        ):
                            if key in payload:
                                target.metadata[key] = payload[key]
                        if "profile" in payload:
                            target.metadata[REALTIME_PROFILE_METADATA_KEY].merge_dict(payload["profile"])
                elif tag == "step":
                    record = payload["record"]
                    if self._decision_history is not None:
                        self._decision_history.finish_frame(record.env_step, record.active_action.value)
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
                        raise ChildProcessError(f"Environment worker exited: {process.exitcode}")
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
            self.env.finish_episode()
            return metrics
        except BaseException:
            self.close()
            raise

    def _release_in_flight_observations(
        self,
        *,
        pool: Any,
        release_observation: Callable[[int | None], None],
    ) -> None:
        while pool.in_flight_count:
            for completion in pool.drain_completed():
                release_observation(completion.task.shared_obs_id)
            if pool.in_flight_count:
                self.sleep_fn(0.001)

    def _drain_in_flight_observation_attempts(
        self,
        *,
        pool: Any,
        observation_attempts: dict[int, dict[str, Any]],
        release_observation: Callable[[int | None], None] | None,
    ) -> None:
        while pool.in_flight_count:
            for completion in pool.drain_completed():
                self._record_observation_completion(
                    completion=completion,
                    observation_attempts=observation_attempts,
                )
                if release_observation is not None:
                    release_observation(completion.task.shared_obs_id)
            if pool.in_flight_count:
                self.sleep_fn(0.001)

    def _drain_and_release_results(
        self,
        *,
        pool: Any,
        coordinator: ResultCoordinator[RealtimePolicyCompletion],
        now_ms: float,
        action_events: list[ActionEvent],
        latency_values_ms: list[float],
        observation_attempts: dict[int, dict[str, Any]],
        enqueue_released_actions: bool,
        release_observation: Callable[[int | None], None] | None,
    ) -> list[ActionEvent]:
        released_events: list[ActionEvent] = []
        for completion in pool.drain_completed():
            self._record_observation_completion(
                completion=completion,
                observation_attempts=observation_attempts,
            )
            if release_observation is not None:
                release_observation(completion.task.shared_obs_id)
            ignored = coordinator.add_completion(int(completion.task.obs_id), completion)
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

    def _record_observation_attempt(
        self,
        *,
        acc: _EpisodeAccumulators,
        episode_id: int,
        task: RealtimeObservationTask,
        submitted: bool,
    ) -> None:
        if self._decision_history is not None:
            self._decision_history.windows[task.submit_env_step] = _DecisionWindow(submitted)
        acc.observation_attempts[task.obs_id] = {
            "episode_id": episode_id,
            "obs_id": task.obs_id,
            "obs_env_step": task.submit_env_step,
            "submitted": submitted,
        }

    def _record_observation_completion(
        self,
        *,
        completion: RealtimePolicyCompletion,
        observation_attempts: dict[int, dict[str, Any]],
    ) -> None:
        if self._decision_history is not None:
            self._decision_history.windows[completion.task.submit_env_step].issued = completion.policy_output.action.value
        observation_attempts[completion.task.obs_id].update(
            {
                "worker_id": completion.worker_id,
                "observation_to_action_latency_ms": (
                    completion.observation_to_action_latency_ms
                ),
                "worker_service_time_ms": completion.worker_service_time_ms,
            }
        )

    def _record_completion_event(
        self,
        *,
        completion: RealtimePolicyCompletion,
        action_events: list[ActionEvent],
        latency_values_ms: list[float],
        dropped: bool,
        discard_reason: str | None,
    ) -> ActionEvent:
        release_start_ms = monotonic_ms()
        latency_record = completion.latency_record
        latency_record.device = completion.device if latency_record.device is None else latency_record.device
        latency_record.metadata = {
            **latency_record.metadata,
            "worker_id": completion.worker_id,
        }
        apply_latency_record_raw_fields(
            record=latency_record,
            latency_type="measured",
            profile_metadata=self._profile_metadata,
        )
        event_metadata = dict(completion.policy_output.metadata)
        event_metadata["frame_ms"] = self.clock.frame_ms
        event_metadata["worker_service_raw_frames"] = self.clock.latency_ms_to_raw_frames(completion.worker_service_time_ms)
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
            action_chunk=completion.policy_output.action_chunk,
        )
        event.dropped = bool(dropped)
        profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
        if profile is not None:
            profile.parent_release_ms = monotonic_ms() - release_start_ms
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
        episode_id: int,
        episode_seed: int | None,
    ) -> RealtimeObservationTask:
        observe_start_ms = monotonic_ms()
        observation = self._with_episode_metadata(
            with_action_prefix(self.env.observe(), self.action_scheduler, self.action_prefix),
            episode_id=episode_id,
            episode_seed=episode_seed,
        )
        observe_end_ms = monotonic_ms()
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
                parent_observation_handle_start_ms=monotonic_ms(),
                parent_to_env_ipc_ms=0.0,
                env_action_recv_to_enqueue_ms=0.0,
            )
        return RealtimeObservationTask(
            obs_id=obs_id,
            observation=observation,
            submit_env_step=env_step,
            submit_time_ms=submit_time_ms,
            observation_start_time_ms=observe_start_ms,
            profile=profile,
        )

    def _with_episode_metadata(
        self,
        observation: Observation,
        *,
        episode_id: int,
        episode_seed: int | None,
    ) -> Observation:
        data = observation.data
        if self._decision_history is not None:
            transport = self._decision_history.observation()
            data = {**data, "transport": transport} if isinstance(data, Mapping) else {"obs": data, "transport": transport}
        return Observation(
            data=data,
            env_step=observation.env_step,
            sim_time_ms=observation.sim_time_ms,
            metadata=observation.metadata
            | {
                "episode_id": episode_id,
                "action_noise_seed": episode_seed,
                "slot_id": 0,
            },
        )

    def _wall_elapsed_ms(self) -> float:
        return (self.time_fn() - self._episode_start_s) * 1000.0
