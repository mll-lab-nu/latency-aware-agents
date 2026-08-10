from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.gymnasium_atari as gymnasium_atari
import scripts.scaffold_gymnasium_atari as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config


AIR_RAID_ACTIONS = [
    "noop",
    "fire",
    "right",
    "left",
    "rightfire",
    "leftfire",
]
AIR_RAID_BASE_PROMPT = (
    "Protect the two buildings by moving sideways and shooting the flying saucers. "
    "Choose exactly one action from: noop, fire, right, left, rightfire, leftfire."
)
AIR_RAID_MAKE_KWARGS = {
    "ale_env_id": "ALE/AirRaid-v5",
    "render_mode": "rgb_array",
    "screen_size": 84,
    "noop_max": 30,
    "frameskip": 1,
    "repeat_action_probability": 0.0,
    "full_action_space": False,
    "terminal_on_life_loss": False,
    "max_num_frames_per_episode": 108000,
    "grayscale_obs": True,
    "mode": 1,
    "difficulty": 0,
}


class _ImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(4, 5, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(6)

    def __init__(self) -> None:
        self.observation = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        self.raw_frame = np.arange(126, dtype=np.uint8).reshape(6, 7, 3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation, {}

    def step(self, action):
        return self.observation, float(action), False, False, {}

    def render(self):
        return self.raw_frame


def test_atari_observation_wrapper_exposes_chw_policy_obs_and_keeps_hwc_render():
    base_env = _ImageEnv()
    env = gymnasium_atari.AtariChannelFirstObservation(base_env)

    observation, _ = env.reset(seed=7)
    rendered = env.render()

    assert env.observation_space.shape == (3, 4, 5)
    assert observation.shape == (3, 4, 5)
    assert observation.dtype == np.uint8
    assert np.array_equal(observation, np.transpose(base_env.observation, (2, 0, 1)))
    assert rendered is base_env.raw_frame
    assert rendered.shape == (6, 7, 3)


def test_atari_observation_wrapper_adds_channel_to_grayscale_obs():
    base_env = _ImageEnv()
    base_env.observation_space = gym.spaces.Box(0, 255, shape=(4, 5), dtype=np.uint8)
    base_env.observation = np.arange(20, dtype=np.uint8).reshape(4, 5)
    env = gymnasium_atari.AtariChannelFirstObservation(base_env)

    observation, _ = env.reset(seed=3)

    assert env.observation_space.shape == (1, 4, 5)
    assert observation.shape == (1, 4, 5)
    assert np.array_equal(observation[0], base_env.observation)


def test_atari_factory_forwards_explicit_ale_and_preprocessing_contract(monkeypatch):
    base_env = _ImageEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["make_kwargs"] = kwargs
        return base_env

    class FakeAtariPreprocessing(gym.Wrapper):
        def __init__(self, env, **kwargs):
            captured["preprocessing_kwargs"] = kwargs
            super().__init__(env)

    monkeypatch.setattr(gymnasium_atari.gym, "make", fake_make)
    monkeypatch.setattr(
        gymnasium_atari.gym.wrappers,
        "AtariPreprocessing",
        FakeAtariPreprocessing,
    )

    env = gymnasium_atari.make_latency_sensitive_atari_env(
        ale_env_id="ALE/AirRaid-v5",
        render_mode="rgb_array",
        screen_size=84,
        noop_max=30,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        terminal_on_life_loss=False,
        max_num_frames_per_episode=108000,
        grayscale_obs=True,
        mode=1,
        difficulty=0,
    )

    assert captured["env_id"] == "ALE/AirRaid-v5"
    assert captured["make_kwargs"] == {
        "render_mode": "rgb_array",
        "obs_type": "rgb",
        "frameskip": 1,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "max_num_frames_per_episode": 108000,
        "mode": 1,
        "difficulty": 0,
    }
    assert captured["preprocessing_kwargs"] == {
        "noop_max": 30,
        "frame_skip": 1,
        "screen_size": 84,
        "terminal_on_life_loss": False,
        "grayscale_obs": True,
        "grayscale_newaxis": False,
        "scale_obs": False,
    }
    assert isinstance(env, gymnasium_atari.AtariChannelFirstObservation)


class _FakeAle:
    def getScreenRGB(self):
        return np.zeros((250, 160, 3), dtype=np.uint8)


class _ScaffoldEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.ale = _FakeAle()
        self.reset_seeds: list[int] = []
        self.closed = False

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros((1, 84, 84), dtype=np.uint8), {}

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    def close(self):
        self.closed = True


def test_air_raid_scaffold_outputs_pal_clock_actions_and_explicit_kwargs(monkeypatch, capsys):
    fake_env = _ScaffoldEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(scaffold.gym, "make", fake_make)

    scaffold.main(["--ale-env-id", "ALE/AirRaid-v5"])
    output = yaml.safe_load(capsys.readouterr().out)
    env_cfg = output["env"]

    assert captured["env_id"] == gymnasium_atari.GYMNASIUM_ATARI_ENV_ID
    assert captured["kwargs"] == env_cfg["make_kwargs"]
    assert fake_env.reset_seeds == [0]
    assert fake_env.closed is True
    assert env_cfg["task_name"] == "air_raid"
    assert env_cfg["env_fps"] == 50.0
    assert env_cfg["obs_fps"] == 12.5
    assert env_cfg["frame_stack"] == 1
    assert env_cfg["noop_action"] == "noop"
    assert env_cfg["action_order"] == AIR_RAID_ACTIONS
    assert env_cfg["action_map"] == dict(zip(AIR_RAID_ACTIONS, range(6)))
    assert env_cfg["make_kwargs"] == AIR_RAID_MAKE_KWARGS


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/air_raid/latency_bench_eval.yaml",
        "configs/examples/gymnasium/air_raid/small_model_train.yaml",
        "configs/examples/gymnasium/air_raid/small_model_eval.yaml",
        "configs/examples/gymnasium/air_raid/starvla_eval.yaml",
    ],
)
def test_air_raid_configs_share_scaffold_contract_and_l5_latency(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_atari.GYMNASIUM_ATARI_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.gymnasium_atari"]
    assert env_cfg["action_order"] == AIR_RAID_ACTIONS
    assert env_cfg["action_map"] == dict(zip(AIR_RAID_ACTIONS, range(6)))
    assert env_cfg["base_prompt"] == AIR_RAID_BASE_PROMPT
    assert env_cfg["make_kwargs"] == AIR_RAID_MAKE_KWARGS
    assert clock.obs_stride_raw_frames == 4
    assert clock.latency_ms_to_raw_frames(config["latency"]["fixed_latency_ms"]) == 5
