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
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from executor_mode_helpers import SequencePolicyRunner, make_dummy_parts


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = float(now)
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


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
    max_catchup_steps_per_loop: int | None = None,
) -> RealtimeLatencyExecutor:
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, env_fps=30.0)
    return RealtimeLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=ZeroLatencySource(),
        action_scheduler=action_scheduler,
        clock=clock,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        policy_worker=worker,
        max_catchup_steps_per_loop=max_catchup_steps_per_loop,
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
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, env_fps=30.0)
    return RealtimeLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
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

    executor.run_episode(episode_id=0, eval_max_steps=3)

    assert sleeps == pytest.approx([executor.clock.frame_ms / 1000.0, executor.clock.frame_ms / 1000.0])
    assert worker.closed


def test_realtime_on_time_ticking_steps_once_per_frame():
    fake_clock = FakeClock()
    worker = ManualPolicyWorker()
    executor = _executor(
        worker=worker,
        actions=[Action("jump")],
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        max_steps=4,
    )

    metrics = executor.run_episode(episode_id=0, eval_max_steps=4)

    assert len(executor.env.applied_actions) == 4
    assert metrics.metadata["total_catchup_steps"] == 0
    assert metrics.metadata["total_missed_ticks"] == 0


def test_realtime_late_by_three_frames_catches_up_in_one_loop():
    fake_clock = FakeClock()
    worker = ManualPolicyWorker()
    executor = _executor(
        worker=worker,
        actions=[Action("jump")],
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        max_steps=10,
    )
    original_step = executor.env.step

    def step_and_jump_wall_clock(action):
        result = original_step(action)
        if len(executor.env.applied_actions) == 1:
            fake_clock.now = executor.clock.step_to_time_ms(3) / 1000.0
        return result

    executor.env.step = step_and_jump_wall_clock

    metrics = executor.run_episode(episode_id=0, eval_max_steps=5)

    assert len(executor.env.applied_actions) == 5
    assert metrics.survival_steps == 5
    assert metrics.metadata["total_catchup_steps"] == 3
    assert metrics.metadata["total_missed_ticks"] == 2
    assert metrics.metadata["skipped_observation_frames"] == 3
    assert metrics.metadata["submitted_observation_frames"] == 2


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

    executor.run_episode(episode_id=0, eval_max_steps=2)

    assert [action.value for action in executor.env.applied_actions] == ["noop", "jump"]


def test_realtime_action_ready_midway_through_catchup_applies_only_after_ready_frame():
    fake_clock = FakeClock()
    jump = Action("jump")
    worker = CompleteMeasuredTaskOnSecondDrainWorker(
        action=jump,
        completion_time_ms=80.0,
        inference_latency_ms=80.0,
        worker_start_time_ms=0.0,
    )
    executor = _measured_executor(
        worker=worker,
        actions=[jump],
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        max_steps=10,
    )
    original_step = executor.env.step

    def step_and_jump_wall_clock(action):
        result = original_step(action)
        if len(executor.env.applied_actions) == 1:
            fake_clock.now = executor.clock.step_to_time_ms(3) / 1000.0
        return result

    executor.env.step = step_and_jump_wall_clock

    executor.run_episode(episode_id=0, eval_max_steps=4)

    assert [action.value for action in executor.env.applied_actions] == ["noop", "noop", "noop", "jump"]
    assert executor.action_scheduler.get_applied_events()[0].applied_env_step == 3


def test_realtime_episode_done_stops_during_catchup():
    fake_clock = FakeClock()
    worker = ManualPolicyWorker()
    executor = _executor(
        worker=worker,
        actions=[Action("jump")],
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        max_steps=2,
    )
    original_step = executor.env.step

    def step_and_jump_wall_clock(action):
        result = original_step(action)
        if len(executor.env.applied_actions) == 1:
            fake_clock.now = executor.clock.step_to_time_ms(3) / 1000.0
        return result

    executor.env.step = step_and_jump_wall_clock

    metrics = executor.run_episode(episode_id=0, eval_max_steps=10)

    assert len(executor.env.applied_actions) == 2
    assert metrics.survival_steps == 2
    assert metrics.metadata["total_catchup_steps"] == 1


def test_realtime_catchup_cap_logs_dropped_frames_and_truncates_episode():
    fake_clock = FakeClock()
    worker = ManualPolicyWorker()
    executor = _executor(
        worker=worker,
        actions=[Action("jump")],
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        max_steps=10,
        max_catchup_steps_per_loop=2,
    )
    original_step = executor.env.step

    def step_and_jump_wall_clock(action):
        result = original_step(action)
        if len(executor.env.applied_actions) == 1:
            fake_clock.now = executor.clock.step_to_time_ms(5) / 1000.0
        return result

    executor.env.step = step_and_jump_wall_clock

    metrics = executor.run_episode(episode_id=0, eval_max_steps=10)

    assert len(executor.env.applied_actions) == 3
    assert metrics.metadata["catchup_capped"] is True
    assert metrics.metadata["realtime_catchup_truncated"] is True
    assert metrics.metadata["dropped_catchup_frames"] == 3


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

    executor.run_episode(episode_id=0, eval_max_steps=2)

    assert logger.latency_records[0].source == "measured"
    assert logger.latency_records[0].latency_ms > 0
    assert logger.latency_records[0].latency_ms_measured == 40.0
    assert logger.latency_records[0].latency_ms_sampled is None
    assert logger.action_records[0].latency_ms_measured == 50.0
    assert logger.action_records[0].latency_ms_sampled is None


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

    executor.run_episode(episode_id=0, eval_max_steps=3)

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

    executor.run_episode(episode_id=0, eval_max_steps=2)

    metadata = logger.latency_records[0].metadata
    assert metadata["queue_wait_ms"] == 10.0
    assert metadata["effective_latency_ms"] == 50.0
    assert metadata["submit_time_ms"] == 0.0
    assert metadata["ready_time_ms"] == 50.0
