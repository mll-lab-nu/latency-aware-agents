from __future__ import annotations

import pytest

from latency_bench.core.types import Action, LatencyRecord, PolicyOutput
from latency_bench.executors.realtime import (
    BoundedObservationQueue,
    RealtimeLatencyExecutor,
    RealtimeObservationTask,
    RealtimePolicyCompletion,
)
from latency_bench.latency.sources import ZeroLatencySource
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.scheduler.decision import EveryFrameDecisionScheduler
from executor_mode_helpers import SequencePolicyRunner, make_dummy_parts


class ManualPolicyWorker:
    def __init__(self):
        self.submitted: list[RealtimeObservationTask] = []
        self.drain_count = 0
        self.closed = False

    def submit(self, task: RealtimeObservationTask):
        self.submitted.append(task)
        return None

    def drain_completed(self):
        self.drain_count += 1
        return []

    def raise_if_failed(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class CompleteFirstTaskOnSecondDrainWorker(ManualPolicyWorker):
    def __init__(self, action: Action, completion_time_ms: float):
        super().__init__()
        self.action = action
        self.completion_time_ms = completion_time_ms
        self.completed = False

    def drain_completed(self):
        self.drain_count += 1
        if self.completed or self.drain_count < 2 or not self.submitted:
            return []
        self.completed = True
        task = self.submitted[0]
        return [
            RealtimePolicyCompletion(
                task=task,
                policy_output=SequencePolicyRunner([self.action]).predict(task.observation),
                completion_time_ms=self.completion_time_ms,
            )
        ]


class CompleteMeasuredTaskOnSecondDrainWorker(ManualPolicyWorker):
    def __init__(
        self,
        action: Action,
        completion_time_ms: float,
        inference_latency_ms: float,
        worker_start_time_ms: float,
    ):
        super().__init__()
        self.action = action
        self.completion_time_ms = completion_time_ms
        self.inference_latency_ms = inference_latency_ms
        self.worker_start_time_ms = worker_start_time_ms
        self.completed = False

    def drain_completed(self):
        self.drain_count += 1
        if self.completed or self.drain_count < 2 or not self.submitted:
            return []
        self.completed = True
        task = self.submitted[0]
        return [
            RealtimePolicyCompletion(
                task=task,
                policy_output=PolicyOutput(action=self.action),
                completion_time_ms=self.completion_time_ms,
                latency_record=LatencyRecord(
                    obs_id=task.obs_id,
                    latency_ms=self.inference_latency_ms,
                    source="measured",
                ),
                worker_start_time_ms=self.worker_start_time_ms,
            )
        ]


def _executor(
    *,
    worker,
    actions: list[Action],
    time_fn,
    sleep_fn,
    max_steps: int = 5,
) -> RealtimeLatencyExecutor:
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, fps_env=30.0)
    return RealtimeLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_source=ZeroLatencySource(),
        action_scheduler=action_scheduler,
        clock=clock,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        policy_worker=worker,
    )


def _measured_executor(
    *,
    worker,
    actions: list[Action],
    time_fn,
    sleep_fn,
    max_steps: int = 5,
    logger=None,
) -> RealtimeLatencyExecutor:
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, fps_env=30.0)
    return RealtimeLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_source=ZeroLatencySource(),
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        policy_worker=worker,
    )


def test_realtime_sleeps_to_each_frame_target():
    now = 10.0
    sleeps: list[float] = []

    def time_fn() -> float:
        return now

    def sleep_fn(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    worker = ManualPolicyWorker()
    executor = _executor(
        worker=worker,
        actions=[Action("jump")],
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        max_steps=3,
    )

    executor.run_episode(episode_id=0, max_env_steps=3)

    assert sleeps == pytest.approx([executor.clock.frame_ms / 1000.0, executor.clock.frame_ms / 1000.0])
    assert worker.closed


def test_realtime_completion_applies_on_next_frame_tick():
    now = 0.0

    def time_fn() -> float:
        return now

    def sleep_fn(seconds: float) -> None:
        nonlocal now
        now += seconds

    jump = Action("jump")
    worker = CompleteFirstTaskOnSecondDrainWorker(action=jump, completion_time_ms=1.0)
    executor = _executor(worker=worker, actions=[jump], time_fn=time_fn, sleep_fn=sleep_fn, max_steps=2)

    executor.run_episode(episode_id=0, max_env_steps=2)

    assert [action.value for action in executor.env.applied_actions] == ["noop", "jump"]


def test_bounded_observation_queue_drops_oldest_pending_task():
    queue = BoundedObservationQueue(max_len=2)
    tasks = [
        RealtimeObservationTask(obs_id=i, observation=None, submit_env_step=i, submit_time_ms=float(i))
        for i in range(3)
    ]

    assert queue.push_drop_oldest(tasks[0]) is None
    assert queue.push_drop_oldest(tasks[1]) is None
    assert queue.push_drop_oldest(tasks[2]) is tasks[0]
    assert queue.pop() is tasks[1]
    assert queue.pop() is tasks[2]
    assert queue.pop() is None


def test_realtime_measured_latency_records_policy_duration(tmp_path):
    now = 0.0
    jump = Action("jump")
    worker = CompleteMeasuredTaskOnSecondDrainWorker(
        action=jump,
        completion_time_ms=50.0,
        inference_latency_ms=40.0,
        worker_start_time_ms=10.0,
    )
    logger = TrajectoryLogger(tmp_path)
    executor = _measured_executor(
        worker=worker,
        actions=[jump],
        time_fn=lambda: now,
        sleep_fn=lambda seconds: None,
        max_steps=2,
        logger=logger,
    )

    executor.run_episode(episode_id=0, max_env_steps=2)

    assert logger.latency_records[0].source == "measured"
    assert logger.latency_records[0].latency_ms > 0


def test_realtime_measured_latency_sets_effective_action_delay():
    now = 0.0

    def time_fn() -> float:
        return now

    def sleep_fn(seconds: float) -> None:
        nonlocal now
        now += seconds

    jump = Action("jump")
    worker = CompleteMeasuredTaskOnSecondDrainWorker(
        action=jump,
        completion_time_ms=50.0,
        inference_latency_ms=50.0,
        worker_start_time_ms=0.0,
    )
    executor = _measured_executor(worker=worker, actions=[jump], time_fn=time_fn, sleep_fn=sleep_fn, max_steps=3)

    executor.run_episode(episode_id=0, max_env_steps=3)

    assert [action.value for action in executor.env.applied_actions] == ["noop", "noop", "jump"]
    assert executor.action_scheduler.get_applied_events()[0].latency_ms == 50.0


def test_realtime_measured_metadata_includes_queue_wait_and_effective_latency(tmp_path):
    now = 0.0
    jump = Action("jump")
    worker = CompleteMeasuredTaskOnSecondDrainWorker(
        action=jump,
        completion_time_ms=50.0,
        inference_latency_ms=40.0,
        worker_start_time_ms=10.0,
    )
    logger = TrajectoryLogger(tmp_path)
    executor = _measured_executor(
        worker=worker,
        actions=[jump],
        time_fn=lambda: now,
        sleep_fn=lambda seconds: None,
        max_steps=2,
        logger=logger,
    )

    executor.run_episode(episode_id=0, max_env_steps=2)

    metadata = logger.latency_records[0].metadata
    assert metadata["queue_wait_ms"] == 10.0
    assert metadata["effective_latency_ms"] == 50.0
    assert metadata["submit_time_ms"] == 0.0
    assert metadata["ready_time_ms"] == 50.0
