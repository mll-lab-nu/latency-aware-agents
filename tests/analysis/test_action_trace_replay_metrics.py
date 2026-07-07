from __future__ import annotations

from latency_bench.analysis.action_trace_replay_metrics import replay_step_records
from latency_bench.core.types import Action, StepRecord, StepResult


class ReplayMetricEnv:
    def __init__(self):
        self.actions: list[str] = []

    def step(self, action: Action) -> StepResult:
        self.actions.append(str(action.name))
        done = len(self.actions) == 2
        reward = -1.0 if done else 0.1
        return StepResult(observation=None, reward=reward, done=done)


def test_replay_step_records_reports_reward_and_done_mismatch_counts():
    env = ReplayMetricEnv()
    records = [
        StepRecord(
            episode_id=0,
            env_step=0,
            sim_time_ms=0.0,
            active_action=Action(0, name="noop"),
            reward=0.1,
            done=False,
        ),
        StepRecord(
            episode_id=0,
            env_step=1,
            sim_time_ms=1.0,
            active_action=Action(1, name="flap"),
            reward=0.1,
            done=False,
        ),
    ]

    summary = replay_step_records(env, records)

    assert summary == {
        "steps_replayed": 2,
        "episode_return": -0.9,
        "done": True,
        "reward_mismatch_count": 1,
        "done_mismatch_count": 1,
    }
    assert env.actions == ["noop", "flap"]
