from __future__ import annotations

from pathlib import Path

from latency_bench.core.types import Action
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.utils.io import load_jsonl_records
from executor_mode_helpers import SequencePolicyRunner, make_dummy_parts


def _executor(
    actions: list[Action],
    max_steps: int = 5,
    logger: TrajectoryLogger | None = None,
    sleep_s: float = 0.0,
) -> PausedLatencyExecutor:
    env, clock, _, _, action_scheduler = make_dummy_parts(max_steps=max_steps, env_fps=30.0)
    return PausedLatencyExecutor(
        env=env,
        policy=SequencePolicyRunner(actions, sleep_s=sleep_s),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


def test_paused_action_applies_same_frame():
    _, _, noop_action, jump_action, _ = make_dummy_parts()
    executor = _executor([jump_action], max_steps=3)

    executor.run_episode(episode_id=0, eval_max_steps=3)

    assert executor.env.applied_actions[0] is jump_action
    assert executor.env.applied_actions[0] is not noop_action


def test_paused_slow_policy_does_not_change_trajectory():
    actions = [Action("first"), Action("second"), Action("third")]
    fast_executor = _executor(actions, max_steps=4, sleep_s=0.0)
    slow_executor = _executor(actions, max_steps=4, sleep_s=0.01)

    fast_executor.run_episode(episode_id=0, eval_max_steps=4)
    slow_executor.run_episode(episode_id=1, eval_max_steps=4)

    assert [action.value for action in fast_executor.env.applied_actions] == [
        action.value for action in slow_executor.env.applied_actions
    ]


def test_paused_resets_policy_state_between_episodes():
    """Regression: a new episode must not inherit the previous policy state."""
    class EpisodePolicy(SequencePolicyRunner):
        def reset_state(self, slot_id=None):
            self.index = 0

    actions = [Action("first"), Action("second")]
    executor = _executor(actions, max_steps=2)
    executor.policy = EpisodePolicy(actions)
    try:
        for episode_id in range(2):
            executor.run_episode(episode_id=episode_id, eval_max_steps=2)
            assert [action.value for action in executor.env.applied_actions] == ["first", "second"]
    finally:
        executor.close()


def test_paused_logs_zero_latency(tmp_path: Path):
    _, _, _, jump_action, _ = make_dummy_parts()
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor([jump_action], max_steps=2, logger=logger)

    metrics = executor.run_episode(episode_id=0, eval_max_steps=2)

    # Records are flushed to disk (and cleared from memory) at episode end.
    assert load_jsonl_records(tmp_path / "actions.jsonl")[0]["latency"]["latency_ms"] == 0.0
    assert load_jsonl_records(tmp_path / "latencies.jsonl")[0]["latency_ms"] == 0.0
    assert metrics.mean_latency_ms == 0.0
