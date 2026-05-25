from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from latency_bench.core.types import Action
from latency_bench.envs import demon_attack
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


class ScriptedDemonGymEnv:
    def __init__(self, steps):
        self.steps = list(steps)
        self.reset_seeds: list[int | None] = []
        self.reset_count = 0
        self.closed = False
        self.ale = SimpleDemonAle(self)
        self.current_rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.reset_count += 1
        frame = np.full((2, 2, 3), self.reset_count, dtype=np.uint8)
        self.current_rgb = frame.copy()
        return frame, {"reset_count": self.reset_count, "seed": seed}

    def step(self, action):
        step = self.steps.pop(0)
        if "was_real_done" in step:
            self.was_real_done = bool(step["was_real_done"])
        frame = np.full((2, 2, 3), int(step.get("frame_value", 9)), dtype=np.uint8)
        self.current_rgb = frame.copy()
        return (
            frame,
            float(step.get("reward", 1.0)),
            bool(step.get("terminated", False)),
            bool(step.get("truncated", False)),
            dict(step.get("info", {})),
        )

    def close(self):
        self.closed = True
        return None


class SimpleDemonAle:
    def __init__(self, env):
        self.env = env

    def getScreenRGB(self):
        return self.env.current_rgb.copy()


class ProcessedObservationRawRgbDemonGymEnv:
    def __init__(self):
        self.closed = False
        self.ale = SimpleDemonAle(self)
        self.current_rgb = np.zeros((210, 160, 3), dtype=np.uint8)
        self.next_value = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        del seed
        self.next_value = 1
        self.current_rgb = np.full((210, 160, 3), 11, dtype=np.uint8)
        return np.full((84, 84, 1), self.next_value, dtype=np.uint8), {}

    def step(self, action):
        del action
        self.next_value = 2
        self.current_rgb = np.full((210, 160, 3), 22, dtype=np.uint8)
        return np.full((84, 84, 1), self.next_value, dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        self.closed = True


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


def test_demon_attack_seeded_reset_rebuilds_gym_env_after_previous_episode(monkeypatch):
    scripted_envs = [ScriptedDemonGymEnv([]), ScriptedDemonGymEnv([])]
    created_envs: list[ScriptedDemonGymEnv] = []

    def make_env(**_kwargs):
        env = scripted_envs.pop(0)
        created_envs.append(env)
        return env

    monkeypatch.setattr(demon_attack, "make_demon_attack_benchmark_env", make_env)
    adapter = demon_attack.DemonAttackEnvAdapter()

    try:
        adapter.reset(seed=123)
        adapter.reset(seed=124)
    finally:
        adapter.close()

    assert created_envs[0].closed is True
    assert created_envs[0].reset_seeds == [123]
    assert created_envs[1].reset_seeds == [124]


def test_demon_attack_observe_stacks_raw_frames_between_policy_observations(monkeypatch):
    scripted_env = ScriptedDemonGymEnv([
        {"frame_value": 2},
        {"frame_value": 3},
    ])
    monkeypatch.setattr(
        demon_attack,
        "make_demon_attack_benchmark_env",
        lambda **_kwargs: scripted_env,
    )
    adapter = demon_attack.DemonAttackEnvAdapter(
        frame_stack=2,
        export_env_raw_rgb_frames=True,
    )
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
        assert np.all(second_policy_observation.data[:3, ...] == 2)
        assert np.all(second_policy_observation.data[3:, ...] == 3)
        raw_stack = second_policy_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        assert raw_stack.shape == (2, 2, 2, 3)
        assert np.all(raw_stack[0] == 2)
        assert np.all(raw_stack[1] == 3)
    finally:
        adapter.close()


def test_demon_attack_raw_rgb_export_preserves_unprocessed_atari_render(monkeypatch):
    scripted_env = ProcessedObservationRawRgbDemonGymEnv()
    monkeypatch.setattr(
        demon_attack,
        "make_demon_attack_benchmark_env",
        lambda **_kwargs: scripted_env,
    )
    adapter = demon_attack.DemonAttackEnvAdapter(
        frame_stack=2,
        export_env_raw_rgb_frames=True,
    )
    try:
        adapter.reset(seed=123)
        adapter.step(Action(value=0, name="noop"))
        policy_observation = adapter.observe()

        raw_stack = policy_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        assert policy_observation.data.shape == (2, 84, 84)
        assert raw_stack.shape == (2, 210, 160, 3)
        assert np.all(raw_stack[0] == 11)
        assert np.all(raw_stack[1] == 22)
    finally:
        adapter.close()


class ScriptedAle:
    def __init__(self, env):
        self.env = env

    def getScreenRGB(self):
        return self.env.current_rgb.copy()


class ScriptedAleRgbEnv(gym.Env):
    action_space = Discrete(2)
    observation_space = Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)

    def __init__(self):
        self.current_rgb = np.zeros((3, 2, 3), dtype=np.uint8)
        self.ale = ScriptedAle(self)
        self._next_value = 0

    def reset(self, seed=None, options=None):
        del seed, options
        self._next_value = 1
        self.current_rgb = np.full((3, 2, 3), self._next_value, dtype=np.uint8)
        return np.zeros((2, 2, 1), dtype=np.uint8), {}

    def step(self, action):
        del action
        self._next_value += 1
        self.current_rgb = np.full((3, 2, 3), self._next_value, dtype=np.uint8)
        return np.zeros((2, 2, 1), dtype=np.uint8), 0.0, False, False, {}


def test_ale_raw_rgb_frame_stack_info_wrapper_reports_oldest_to_newest_stack():
    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackInfoWrapper

    env = RawRgbFrameStackInfoWrapper(
        ScriptedAleRgbEnv(),
        num_frames=3,
        raw_frame_fn=lambda wrapped_env: wrapped_env.unwrapped.ale.getScreenRGB(),
    )

    _, reset_info = env.reset(seed=123)
    _, _, _, _, step_info_1 = env.step(0)
    _, _, _, _, step_info_2 = env.step(0)

    assert reset_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].shape == (3, 3, 2, 3)
    assert reset_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][0, 0, 0].tolist() == [1, 1, 1]
    assert reset_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1, 0, 0].tolist() == [1, 1, 1]
    assert step_info_1[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0].tolist() == [[1, 1, 1], [1, 1, 1], [2, 2, 2]]
    assert step_info_2[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0].tolist() == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
