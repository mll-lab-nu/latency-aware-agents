from __future__ import annotations

import numpy as np

from latency_bench.core.types import Action
from latency_bench.envs import demon_attack


class ScriptedDemonGymEnv:
    def __init__(self, steps):
        self.steps = list(steps)
        self.reset_seeds: list[int | None] = []
        self.reset_count = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.reset_count += 1
        frame = np.full((2, 2, 3), self.reset_count, dtype=np.uint8)
        return frame, {"reset_count": self.reset_count, "seed": seed}

    def step(self, action):
        step = self.steps.pop(0)
        if "was_real_done" in step:
            self.was_real_done = bool(step["was_real_done"])
        frame = np.full((2, 2, 3), int(step.get("frame_value", 9)), dtype=np.uint8)
        return (
            frame,
            float(step.get("reward", 1.0)),
            bool(step.get("terminated", False)),
            bool(step.get("truncated", False)),
            dict(step.get("info", {})),
        )

    def close(self):
        return None


def _adapter(monkeypatch, scripted_env: ScriptedDemonGymEnv, *, continue_life_loss_terminal: bool):
    monkeypatch.setattr(
        demon_attack,
        "make_demon_attack_benchmark_env",
        lambda **_kwargs: scripted_env,
    )
    adapter = demon_attack.DemonAttackEnvAdapter(
        terminal_on_life_loss=True,
        clip_reward=False,
        continue_life_loss_terminal=continue_life_loss_terminal,
    )
    adapter.reset(seed=123)
    return adapter


def test_demon_attack_adapter_soft_resets_life_loss_terminal_without_ending_episode(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {
            "reward": 5.0,
            "terminated": True,
            "info": {"is_true_episode_end": False, "lives": 2},
            "frame_value": 7,
        }
    ])
    adapter = _adapter(monkeypatch, scripted_env, continue_life_loss_terminal=True)

    result = adapter.step(Action(value=1, name="fire"))

    assert result.reward == 5.0
    assert result.done is False
    assert result.truncated is False
    assert result.info["soft_reset"] is True
    assert result.info["ignored_life_loss_terminal"] is True
    assert result.info["original_terminated"] is True
    assert result.info["is_true_episode_end"] is False
    assert result.observation is not None
    assert result.observation.metadata["reset_count"] == 2
    assert scripted_env.reset_seeds == [123, None]


def test_demon_attack_adapter_uses_episodic_life_wrapper_state_when_info_is_sparse(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {
            "reward": 3.0,
            "terminated": True,
            "info": {},
            "was_real_done": False,
        }
    ])
    adapter = _adapter(monkeypatch, scripted_env, continue_life_loss_terminal=True)

    result = adapter.step(Action(value=1, name="fire"))

    assert result.done is False
    assert result.info["soft_reset"] is True
    assert scripted_env.reset_seeds == [123, None]


def test_demon_attack_adapter_keeps_true_game_over_terminal(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {
            "reward": 11.0,
            "terminated": True,
            "info": {"is_true_episode_end": True, "lives": 0},
        }
    ])
    adapter = _adapter(monkeypatch, scripted_env, continue_life_loss_terminal=True)

    result = adapter.step(Action(value=1, name="fire"))

    assert result.done is True
    assert result.info["is_true_episode_end"] is True
    assert "soft_reset" not in result.info
    assert scripted_env.reset_seeds == [123]


def test_demon_attack_adapter_can_count_life_loss_as_terminal_when_compat_flag_disabled(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {
            "reward": 5.0,
            "terminated": True,
            "info": {"is_true_episode_end": False, "lives": 2},
        }
    ])
    adapter = _adapter(monkeypatch, scripted_env, continue_life_loss_terminal=False)

    result = adapter.step(Action(value=1, name="fire"))

    assert result.done is True
    assert "soft_reset" not in result.info
    assert scripted_env.reset_seeds == [123]


def test_demon_attack_observe_stacks_only_observed_frames(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {"frame_value": 2},
        {"frame_value": 3},
    ])
    monkeypatch.setattr(
        demon_attack,
        "make_demon_attack_benchmark_env",
        lambda **_kwargs: scripted_env,
    )
    adapter = demon_attack.DemonAttackEnvAdapter(frame_stack=2)
    try:
        reset_observation = adapter.reset(seed=123)
        first_policy_observation = adapter.observe()
        adapter.step(Action(value=0, name="noop"))
        second_step = adapter.step(Action(value=0, name="noop"))
        second_policy_observation = adapter.observe()

        assert reset_observation.data.shape == (2, 2, 3)
        assert second_step.observation is not None
        assert second_step.observation.data.shape == (2, 2, 3)
        assert first_policy_observation.data.shape == (6, 2, 2)
        assert second_policy_observation.data.shape == (6, 2, 2)
        assert np.all(second_policy_observation.data[:3, ...] == 1)
        assert np.all(second_policy_observation.data[3:, ...] == 3)
    finally:
        adapter.close()
