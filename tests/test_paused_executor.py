from __future__ import annotations

from pathlib import Path

from latency_bench.core.types import Action
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.latency.sources import FixedLatencySource
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.scheduler.decision import EveryFrameDecisionScheduler
from executor_mode_helpers import SequencePolicyRunner, make_dummy_parts


def _executor(
    latency_ms: float,
    actions: list[Action],
    max_steps: int = 5,
    logger: TrajectoryLogger | None = None,
    sleep_s: float = 0.0,
) -> PausedLatencyExecutor:
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, fps_env=30.0)
    return PausedLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions, sleep_s=sleep_s),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_source=FixedLatencySource(latency_ms),
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


def test_paused_fixed_latency_action_applies_same_frame():
    _, clock, noop_action, jump_action, _ = make_dummy_parts()
    executor = _executor(clock.frame_ms * 2, [jump_action], max_steps=3)

    executor.run_episode(episode_id=0, eval_max_steps=3)

    assert executor.env.applied_actions[0] is jump_action
    assert executor.env.applied_actions[0] is not noop_action


def test_paused_slow_policy_does_not_change_trajectory():
    actions = [Action("first"), Action("second"), Action("third")]
    _, clock, _, _, _ = make_dummy_parts()
    fast_executor = _executor(clock.frame_ms * 2, actions, max_steps=4, sleep_s=0.0)
    slow_executor = _executor(clock.frame_ms * 2, actions, max_steps=4, sleep_s=0.01)

    fast_executor.run_episode(episode_id=0, eval_max_steps=4)
    slow_executor.run_episode(episode_id=1, eval_max_steps=4)

    assert [action.value for action in fast_executor.env.applied_actions] == [
        action.value for action in slow_executor.env.applied_actions
    ]


def test_paused_logs_configured_latency_without_delaying_action(tmp_path: Path):
    _, clock, _, jump_action, _ = make_dummy_parts()
    fixed_latency_ms = clock.frame_ms * 2
    logger = TrajectoryLogger(tmp_path)
    executor = _executor(fixed_latency_ms, [jump_action], max_steps=2, logger=logger)

    metrics = executor.run_episode(episode_id=0, eval_max_steps=2)

    assert logger.action_records[0].latency_ms == 0.0
    assert logger.latency_records[0].latency_ms == fixed_latency_ms
    assert logger.action_records[0].metadata["paused_configured_latency_ms"] == fixed_latency_ms
    assert metrics.mean_latency_ms == fixed_latency_ms
