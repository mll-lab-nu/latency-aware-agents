from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, Observation, PolicyOutput, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.latency.profiler import LatencyProfiler
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import compute_episode_metrics
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
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
        latency_profiler: LatencyProfiler | None = None,
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
                    policy_output = self.policy.predict(task.observation)
                    latency_record = None
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
        observation_builder: ObservationBuilder,
        history: HistoryBuffer,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        max_pending_observations: int = 16,
        time_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
        policy_worker: SerialPolicyWorker | None = None,
        latency_profiler: LatencyProfiler | None = None,
    ):
        if max_pending_observations <= 0:
            raise ValueError("max_pending_observations must be positive")
        self.env = env
        self.policy = policy
        self.observation_builder = observation_builder
        self.history = history
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
        self._episode_start_s = 0.0
        self._next_obs_id = 0
        self._next_action_id = 0

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        max_env_steps: int = 10000,
    ) -> EpisodeMetrics:
        if max_env_steps < 0:
            raise ValueError("max_env_steps must be non-negative")

        self.history.clear()
        self.action_scheduler.reset()
        initial_observation = self.env.reset(seed=seed)
        self.history.append(initial_observation)

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        dropped_observation_count = 0

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

        try:
            env_step = 0
            while env_step < max_env_steps:
                scheduled_time_ms = self.clock.step_to_time_ms(env_step)
                target_wall_s = self._episode_start_s + scheduled_time_ms / 1000.0
                now_s = self.time_fn()
                if now_s < target_wall_s:
                    self.sleep_fn(target_wall_s - now_s)

                tick_time_ms = max(
                    scheduled_time_ms,
                    (self.time_fn() - self._episode_start_s) * 1000.0,
                )

                completions = policy_worker.drain_completed()
                for completion in completions:
                    if completion.error is not None:
                        raise completion.error
                    if completion.latency_record is None:
                        source_latency_ms = float(self.latency_source.next_latency_ms({
                            "completion": completion,
                            "observation": completion.task.observation,
                            "obs_id": completion.task.obs_id,
                            "env_step": env_step,
                            "sim_time_ms": tick_time_ms,
                        }))
                        ready_time_ms = completion.completion_time_ms + source_latency_ms
                        latency_record = LatencyRecord(
                            obs_id=completion.task.obs_id,
                            latency_ms=source_latency_ms,
                            source=self.latency_source.__class__.__name__,
                        )
                    else:
                        ready_time_ms = completion.completion_time_ms
                        latency_record = completion.latency_record

                    effective_latency_ms = ready_time_ms - completion.task.submit_time_ms
                    if completion.latency_record is not None:
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

                    event = ActionEvent(
                        action_id=self._next_action_id,
                        obs_id=completion.task.obs_id,
                        action=completion.policy_output.action,
                        obs_env_step=completion.task.observation.env_step,
                        obs_time_ms=completion.task.observation.sim_time_ms,
                        latency_ms=effective_latency_ms,
                        ready_time_ms=ready_time_ms,
                        ready_step_float=self.clock.time_ms_to_step_float(ready_time_ms),
                        metadata=completion.policy_output.metadata,
                    )
                    self._next_action_id += 1
                    self.action_scheduler.enqueue(event)
                    action_events.append(event)
                    latency_values_ms.append(effective_latency_ms)
                    if self.logger is not None:
                        self.logger.log_action_event(event)
                        self.logger.log_latency(latency_record)

                policy_worker.raise_if_failed()

                if self.decision_scheduler.should_observe(env_step, scheduled_time_ms):
                    observation = self.observation_builder.build(self.env, self.history)
                    obs_id = self._next_obs_id
                    self._next_obs_id += 1
                    task = RealtimeObservationTask(
                        obs_id=obs_id,
                        observation=observation,
                        submit_env_step=env_step,
                        submit_time_ms=tick_time_ms,
                    )
                    dropped_task = policy_worker.submit(task)
                    if dropped_task is not None:
                        dropped_observation_count += 1

                active_action = self.action_scheduler.update(env_step, tick_time_ms)
                result = self.env.step(active_action)
                record = StepRecord(
                    episode_id=episode_id,
                    env_step=env_step,
                    sim_time_ms=scheduled_time_ms,
                    active_action=active_action,
                    reward=float(result.reward),
                    done=bool(result.done or result.truncated),
                    info=result.info,
                )
                step_records.append(record)
                if self.logger is not None:
                    self.logger.log_step(record)
                if result.observation is not None:
                    self.history.append(result.observation)

                env_step += 1
                if result.done or result.truncated:
                    break

            metrics = compute_episode_metrics(
                episode_id=episode_id,
                step_records=step_records,
                action_events=action_events,
                latency_values_ms=latency_values_ms,
                metadata={"dropped_observation_count": dropped_observation_count},
            )
            if self.logger is not None:
                self.logger.log_episode_metrics(metrics)
            return metrics
        finally:
            policy_worker.close()
