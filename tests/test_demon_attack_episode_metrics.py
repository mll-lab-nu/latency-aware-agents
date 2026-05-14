from __future__ import annotations

import json

import gymnasium as gym
import numpy as np


class _ScriptedAtariInfoEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0

    def reset(self, **kwargs):
        return np.zeros((2, 2), dtype=np.uint8), {}

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        return np.full((2, 2), self.index, dtype=np.uint8), float(reward), terminated, truncated, dict(info)


def test_learner_reward_stays_clipped_while_episode_return_is_raw(tmp_path):
    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})]),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(tmp_path / "episodes.jsonl"),
    )

    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)

    assert reward == 1.0
    assert info["raw_reward"] == 10.0
    assert info["episodic_return"] == 10.0
    assert info["episodic_raw_return"] == 10.0
    assert "clipped_reward" not in info
    assert "episodic_clipped_return" not in info
    assert "episode/clipped_return" not in info["episode_extra_stats"]


def test_true_game_over_emits_raw_full_game_return_after_life_loss(tmp_path):
    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv(
                [
                    (10.0, True, False, {"lives": 2}),
                    (20.0, True, False, {"lives": 0}),
                ]
            ),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, first_reward, _, _, first_info = wrapped.step(0)
    wrapped.reset()
    _, second_reward, _, _, second_info = wrapped.step(0)

    assert first_reward == 1.0
    assert first_info["is_true_episode_end"] is False
    assert "episodic_return" not in first_info
    assert second_reward == 1.0
    assert second_info["is_true_episode_end"] is True
    assert second_info["episodic_return"] == 30.0
    assert second_info["episodic_raw_return"] == 30.0
    assert "episodic_clipped_return" not in second_info

    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["return"] == 30.0
    assert events[0]["raw_return"] == 30.0
    assert "clipped_return" not in events[0]


def test_episode_metrics_records_only_raw_return(tmp_path):
    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})]),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)

    assert reward == 1.0
    assert info["episodic_return"] == 10.0
    assert info["episode_extra_stats"]["episode/return"] == 10.0
    assert info["episode_extra_stats"]["episode/raw_return"] == 10.0
    assert "episode/clipped_return" not in info["episode_extra_stats"]

    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["return"] == 10.0
    assert event["raw_return"] == 10.0
    assert "clipped_return" not in event
