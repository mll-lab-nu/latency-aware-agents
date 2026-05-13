from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, Observation, PolicyOutput, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


@dataclass
class RealtimeObservationTask:
    obs_id: int
    observation: Observation
    submit_env_step: int
    submit_time_ms: float


@dataclass
class RealtimePolicyCompletion:
    task: RealtimeObservationTask
    policy_output: PolicyOutput
    completion_time_ms: float
    error: BaseException | None = None
    latency_record: LatencyRecord | None = None
    worker_start_time_ms: float | None = None


class BoundedObservationQueue:
    def __init__(self, max_len: int):
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.max_len = int(max_len)
        self._items: list[RealtimeObservationTask] = []

    def push_drop_oldest(self, task: RealtimeObservationTask) -> RealtimeObservationTask | None:
        dropped = None
        if len(self._items) >= self.max_len:
            dropped = self._items.pop(0)
        self._items.append(task)
        return dropped

    def pop(self) -> RealtimeObservationTask | None:
        if not self._items:
            return None
        return self._items.pop(0)


class SerialPolicyWorker:
    def __init__(
        self,
        policy: PolicyRunner,
        max_pending_observations: int,
        time_fn: Callable[[], float],
        latency_profiler: RuntimeLatencyMeasurer | None = None,
        episode_start_s_getter: Callable[[], float] | None = None,
    ):
        self.policy = policy
        self.time_fn = time_fn
        self.latency_profiler = latency_profiler
        self.episode_start_s_getter = episode_start_s_getter
        self.queue = BoundedObservationQueue(max_pending_observations)
        self._condition = threading.Condition()
        self._completed: list[RealtimePolicyCompletion] = []
        self._failure: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="latency-bench-policy-worker", daemon=True)
        self._thread.start()

    def submit(self, task: RealtimeObservationTask) -> RealtimeObservationTask | None:
        with self._condition:
            if self._closed:
                raise RuntimeError("policy worker is closed")
            dropped = self.queue.push_drop_oldest(task)
            self._condition.notify()
            return dropped

    def drain_completed(self) -> list[RealtimePolicyCompletion]:
        with self._condition:
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed:
                    task = self.queue.pop()
                    if task is not None:
                        break
                    self._condition.wait()
                else:
                    return

            try:
                worker_start_time_ms = self._time_ms()
                if self.latency_profiler is None:
                    start_wall_time = self.time_fn()
                    policy_output = self.policy.predict(task.observation)
                    end_wall_time = self.time_fn()
                    elapsed_ms = (end_wall_time - start_wall_time) * 1000.0
                    latency_record = LatencyRecord(
                        obs_id=task.obs_id,
                        latency_ms=elapsed_ms,
                        start_wall_time=start_wall_time,
                        end_wall_time=end_wall_time,
                        inference_ms=elapsed_ms,
                    )
                else:
                    policy_output, latency_record = self.latency_profiler.run_and_measure(
                        self.policy.predict,
                        task.observation,
                        obs_id=task.obs_id,
                    )
            except BaseException as exc:
                with self._condition:
                    self._failure = exc
                    self._closed = True
                    self._condition.notify_all()
                return

            completion = RealtimePolicyCompletion(
                task=task,
                policy_output=policy_output,
                completion_time_ms=self._time_ms(),
                latency_record=latency_record,
                worker_start_time_ms=worker_start_time_ms,
            )
            with self._condition:
                self._completed.append(completion)
                self._condition.notify_all()

    def _time_ms(self) -> float:
        episode_start_s = self.episode_start_s_getter() if self.episode_start_s_getter is not None else 0.0
        return (self.time_fn() - episode_start_s) * 1000.0


class RealtimeLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        max_pending_observations: int = 16,
        time_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
        policy_worker: SerialPolicyWorker | None = None,
        latency_profiler: RuntimeLatencyMeasurer | None = None,
        max_catchup_steps_per_loop: int | None = None,
    ):
        if max_pending_observations <= 0:
            raise ValueError("max_pending_observations must be positive")
        if max_catchup_steps_per_loop is not None and int(max_catchup_steps_per_loop) <= 0:
            raise ValueError("max_catchup_steps_per_loop must be positive or None")
        self.env = env
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.latency_source = latency_source
        self.action_scheduler = action_scheduler
        self.clock = clock
        self.logger = logger
        self.max_pending_observations = int(max_pending_observations)
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.policy_worker = policy_worker
        self.latency_profiler = latency_profiler
        self.max_catchup_steps_per_loop = (
            None if max_catchup_steps_per_loop is None else int(max_catchup_steps_per_loop)
        )
        self._episode_start_s = 0.0
        self._next_obs_id = 0
        self._next_action_id = 0

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        if eval_max_steps < 0:
            raise ValueError("eval_max_steps must be non-negative")

        self.action_scheduler.reset()
        self.env.reset(seed=seed)

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        dropped_observation_count = 0
        skipped_observation_frames = 0
        submitted_observation_frames = 0
        total_catchup_steps = 0
        total_missed_ticks = 0
        max_wall_clock_lag_ms = 0.0
        catchup_capped = False
        dropped_catchup_frames = 0
        realtime_catchup_truncated = False
        last_submitted_observation_env_step: int | None = None
        episode_latency_type = "measured"
        episode_metadata = episode_raw_fact_metadata(
            mode="realtime",
            episode_seed=seed,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            frame_ms=self.clock.frame_ms,
            latency_type=episode_latency_type,
        )

        self._episode_start_s = self.time_fn()
        policy_worker = self.policy_worker
        if policy_worker is None:
            policy_worker = SerialPolicyWorker(
                policy=self.policy,
                max_pending_observations=self.max_pending_observations,
                time_fn=self.time_fn,
                latency_profiler=self.latency_profiler,
                episode_start_s_getter=lambda: self._episode_start_s,
            )

        def wall_elapsed_ms() -> float:
            return (self.time_fn() - self._episode_start_s) * 1000.0

        def due_step_for_wall_time(elapsed_ms: float) -> int:
            epsilon_ms = max(1e-9, self.clock.frame_ms * 1e-9)
            return int(math.floor((float(elapsed_ms) + epsilon_ms) / self.clock.frame_ms))

        def drain_policy_completions(env_step: int, frame_time_ms: float) -> None:
            completions = policy_worker.drain_completed()
            for completion in completions:
                if completion.error is not None:
                    raise completion.error
                if completion.latency_record is None:
                    latency_ms = (
                        0.0
                        if completion.worker_start_time_ms is None
                        else completion.completion_time_ms - completion.worker_start_time_ms
                    )
                    latency_record = LatencyRecord(
                        obs_id=completion.task.obs_id,
                        latency_ms=latency_ms,
                        inference_ms=latency_ms,
                    )
                else:
                    latency_record = completion.latency_record
                ready_time_ms = completion.completion_time_ms
                latency_type = "measured"
                profile_metadata: dict[str, Any] = {}

                effective_latency_ms = ready_time_ms - completion.task.submit_time_ms
                queue_wait_ms = (
                    None
                    if completion.worker_start_time_ms is None
                    else completion.worker_start_time_ms - completion.task.submit_time_ms
                )
                latency_record.metadata = {
                    **latency_record.metadata,
                    "queue_wait_ms": queue_wait_ms,
                    "effective_latency_ms": effective_latency_ms,
                    "submit_time_ms": completion.task.submit_time_ms,
                    "ready_time_ms": ready_time_ms,
                }
                apply_latency_record_raw_fields(
                    record=latency_record,
                    latency_type=latency_type,
                    profile_metadata=profile_metadata,
                )

                event = ActionEvent(
                    action_id=self._next_action_id,
                    action=completion.policy_output.action,
                    obs_env_step=completion.task.observation.env_step,
                    obs_time_ms=completion.task.observation.sim_time_ms,
                    ready_time_ms=ready_time_ms,
                    ready_step_float=self.clock.time_ms_to_step_float(ready_time_ms),
                    latency=latency_record,
                    metadata=completion.policy_output.metadata,
                )
                apply_latency_raw_fields(
                    event=event,
                    latency_type=latency_type,
                    frame_ms=self.clock.frame_ms,
                    profile_metadata=profile_metadata,
                )
                self._next_action_id += 1
                self.action_scheduler.enqueue(event)
                action_events.append(event)
                latency_values_ms.append(latency_record.latency_ms)
                if self.logger is not None:
                    self.logger.log_action_event(event)
                    self.logger.log_latency(latency_record)

        def submit_observation(env_step: int, submit_time_ms: float) -> None:
            nonlocal dropped_observation_count
            nonlocal submitted_observation_frames
            nonlocal last_submitted_observation_env_step
            observation = self.env.observe()
            obs_id = self._next_obs_id
            self._next_obs_id += 1
            task = RealtimeObservationTask(
                obs_id=obs_id,
                observation=observation,
                submit_env_step=env_step,
                submit_time_ms=submit_time_ms,
            )
            dropped_task = policy_worker.submit(task)
            if dropped_task is not None:
                dropped_observation_count += 1
            submitted_observation_frames += 1
            last_submitted_observation_env_step = int(env_step)

        try:
            env_step = 0
            episode_done = False
            while env_step < eval_max_steps:
                scheduled_time_ms = self.clock.step_to_time_ms(env_step)
                target_wall_s = self._episode_start_s + scheduled_time_ms / 1000.0
                now_s = self.time_fn()
                if now_s < target_wall_s:
                    self.sleep_fn(target_wall_s - now_s)

                current_wall_elapsed_ms = max(wall_elapsed_ms(), scheduled_time_ms)
                target_due_step = max(env_step, due_step_for_wall_time(current_wall_elapsed_ms))
                due_steps = min(eval_max_steps - env_step, target_due_step - env_step + 1)
                uncapped_due_steps = due_steps
                loop_catchup_capped = False
                loop_dropped_catchup_frames = 0
                if (
                    self.max_catchup_steps_per_loop is not None
                    and due_steps > self.max_catchup_steps_per_loop
                ):
                    loop_catchup_capped = True
                    loop_dropped_catchup_frames = due_steps - self.max_catchup_steps_per_loop
                    due_steps = self.max_catchup_steps_per_loop
                    catchup_capped = True
                    dropped_catchup_frames += loop_dropped_catchup_frames
                    realtime_catchup_truncated = True

                is_catchup_batch = uncapped_due_steps > 1

                skipped_this_batch = 0
                executed_this_batch = 0
                for _batch_index in range(due_steps):
                    frame_time_ms = self.clock.step_to_time_ms(env_step)
                    wall_clock_lag_ms = max(0.0, current_wall_elapsed_ms - frame_time_ms)
                    max_wall_clock_lag_ms = max(max_wall_clock_lag_ms, wall_clock_lag_ms)

                    drain_policy_completions(env_step, frame_time_ms)
                    policy_worker.raise_if_failed()

                    observation_due = self.decision_scheduler.should_observe(env_step, frame_time_ms)
                    observation_submitted = False
                    observation_skipped = False
                    if observation_due:
                        if is_catchup_batch:
                            skipped_observation_frames += 1
                            skipped_this_batch += 1
                            observation_skipped = True
                        else:
                            if last_submitted_observation_env_step != env_step:
                                submit_observation(env_step, current_wall_elapsed_ms)
                                observation_submitted = True

                    active_action = self.action_scheduler.update(env_step, frame_time_ms)
                    result = self.env.step(active_action)
                    executed_this_batch += 1
                    step_info = dict(result.info)
                    step_info.update(
                        {
                            "scheduled_time_ms": frame_time_ms,
                            "wall_elapsed_ms_at_loop_start": current_wall_elapsed_ms,
                            "frame_time_ms": self.clock.frame_ms,
                            "wall_clock_lag_ms": wall_clock_lag_ms,
                            "catchup_frame": is_catchup_batch,
                            "catchup_batch_size": due_steps,
                            "missed_ticks_before_frame": max(0, target_due_step - env_step),
                            "observation_submitted": observation_submitted,
                            "observation_skipped_due_to_catchup": observation_skipped,
                            "observation_submission_mode": "latest_only",
                            "catchup_capped": loop_catchup_capped,
                            "dropped_catchup_frames": loop_dropped_catchup_frames,
                            "uncapped_catchup_batch_size": uncapped_due_steps,
                        }
                    )
                    step_info.update(
                        active_action_step_info(
                            applied_events=self.action_scheduler.get_applied_events(),
                            env_step=env_step,
                            scheduled_time_ms=frame_time_ms,
                            frame_ms=self.clock.frame_ms,
                            latency_type=episode_latency_type,
                            profile_metadata=profile_metadata_from_source(self.latency_source),
                        )
                    )
                    record = StepRecord(
                        episode_id=episode_id,
                        env_step=env_step,
                        sim_time_ms=frame_time_ms,
                        active_action=active_action,
                        reward=float(result.reward),
                        done=bool(result.done or result.truncated),
                        info=step_info,
                    )
                    step_records.append(record)
                    if self.logger is not None:
                        self.logger.log_step(record)
                    env_step += 1
                    if result.done or result.truncated:
                        episode_done = True
                        break

                if is_catchup_batch:
                    total_catchup_steps += executed_this_batch
                    total_missed_ticks += max(0, executed_this_batch - 1)

                if episode_done:
                    break
                if realtime_catchup_truncated:
                    break

                if is_catchup_batch and skipped_this_batch > 0 and env_step < eval_max_steps:
                    submit_observation(env_step, current_wall_elapsed_ms)

            metrics = compute_episode_metrics(
                episode_id=episode_id,
                step_records=step_records,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
                metadata={
                    "dropped_observation_count": dropped_observation_count,
                    "skipped_observation_frames": skipped_observation_frames,
                    "submitted_observation_frames": submitted_observation_frames,
                    "observation_submission_mode": "latest_only",
                    "total_missed_ticks": total_missed_ticks,
                    "max_wall_clock_lag_ms": max_wall_clock_lag_ms,
                    "total_catchup_steps": total_catchup_steps,
                    "catchup_capped": catchup_capped,
                    "dropped_catchup_frames": dropped_catchup_frames,
                    "realtime_catchup_truncated": realtime_catchup_truncated,
                    **episode_metadata,
                },
                frame_ms=self.clock.frame_ms,
            )
            if self.logger is not None:
                self.logger.log_episode_metrics(metrics)
            return metrics
        finally:
            policy_worker.close()
