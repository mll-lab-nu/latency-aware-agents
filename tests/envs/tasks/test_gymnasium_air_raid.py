from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.tasks.air_raid as gymnasium_air_raid
import scripts.scaffold_gymnasium_air_raid as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper


ACTION_LABELS = ["noop", "fire", "right", "left", "rightfire", "leftfire"]


class _FakeAle:
    def __init__(self) -> None:
        self.episode_frame = 8

    def getEpisodeFrameNumber(self):
        return self.episode_frame

    def getAvailableModes(self):
        return [1, 2, 3, 4, 5, 6, 7, 8]

    def getAvailableDifficulties(self):
        return [0]


class _AirRaidProbeEnv(gym.Env):
    action_space = gym.spaces.Discrete(6)
    observation_space = gym.spaces.Box(0, 255, shape=(250, 160, 3), dtype=np.uint8)

    def __init__(self) -> None:
        self.ale = _FakeAle()
        self.metadata = {"render_modes": ["rgb_array"]}
        self.spec = SimpleNamespace(id="ALE/AirRaid-v5", entry_point="fake:AtariEnv", max_episode_steps=None, kwargs={"render_mode": "rgb_array"})
        self.actions: list[int] = []
        self.closed = False

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ale.episode_frame = 8
        return np.zeros((250, 160, 3), dtype=np.uint8), {"lives": 1, "episode_frame_number": 8}

    def step(self, action):
        self.actions.append(action)
        self.ale.episode_frame += 1
        obs = np.full((250, 160, 3), self.ale.episode_frame, dtype=np.uint8)
        return obs, 0.0, False, False, {"lives": 1, "episode_frame_number": self.ale.episode_frame}

    def render(self):
        return np.full((250, 160, 3), 7, dtype=np.uint8)

    def close(self):
        self.closed = True


def test_air_raid_wrapper_exposes_chw_policy_obs_and_native_rgb_render(monkeypatch):
    native_env = _AirRaidProbeEnv()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(gymnasium_air_raid, "_register_ale", lambda: None)

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(gymnasium_air_raid.gym, "make", fake_make)
    env = gymnasium_air_raid.make_latency_sensitive_air_raid_env(
        base_env_id="ALE/AirRaid-v5",
        render_mode="rgb_array",
        screen_size=84,
        noop_max=0,
        base_make_kwargs={"frameskip": 1},
    )
    observation, _ = env.reset(seed=4)
    next_observation, _, _, _, _ = env.step(5)

    assert captured == {"env_id": "ALE/AirRaid-v5", "kwargs": {"render_mode": "rgb_array", "frameskip": 1}}
    assert env.observation_space.shape == (1, 84, 84)
    assert observation.shape == (1, 84, 84)
    assert next_observation.shape == (1, 84, 84)
    assert observation.dtype == np.uint8
    assert native_env.actions == [5]
    assert env.render().shape == (250, 160, 3)


def test_l1_scheduler_advances_four_raw_frames_and_delays_first_action(monkeypatch):
    native_env = _AirRaidProbeEnv()
    monkeypatch.setattr(gymnasium_air_raid, "_register_ale", lambda: None)
    monkeypatch.setattr(gymnasium_air_raid.gym, "make", lambda *args, **kwargs: native_env)
    task_env = gymnasium_air_raid.make_latency_sensitive_air_raid_env(
        base_env_id="ALE/AirRaid-v5", render_mode="rgb_array", screen_size=84, noop_max=0, base_make_kwargs={}
    )
    env = GymnasiumActionCatalogWrapper(task_env, list(range(6)))
    env = ActionLatencyWrapper(env, fixed_latency_raw_frames=1, env_fps=50, obs_fps=12.5, hold_policy="hold", ordering_policy="issue_order_fifo", noop_action=0)
    env.reset(seed=0)
    env.step(4)
    env.step(5)

    assert native_env.actions == [0, 4, 4, 4, 4, 5, 5, 5]


def test_air_raid_fire_reset_starts_game_with_fire_actions(monkeypatch):
    native_env = _AirRaidProbeEnv()
    monkeypatch.setattr(gymnasium_air_raid, "_register_ale", lambda: None)
    monkeypatch.setattr(gymnasium_air_raid.gym, "make", lambda *args, **kwargs: native_env)
    env = gymnasium_air_raid.make_latency_sensitive_air_raid_env(
        base_env_id="ALE/AirRaid-v5",
        render_mode="rgb_array",
        screen_size=84,
        noop_max=0,
        base_make_kwargs={},
        fire_reset=True,
    )

    env.reset(seed=0)

    assert native_env.actions == [1, 2]


def test_scaffold_reports_pal_clock_and_native_action_meanings(monkeypatch, capsys):
    native_env = _AirRaidProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(scaffold.importlib.metadata, "version", lambda package: {"gymnasium": "0.29.1", "ale-py": "0.8.1"}[package])
    scaffold.main([])
    output = yaml.safe_load(capsys.readouterr().out)

    assert output["probe_report"]["native_env_fps"] == 50
    assert output["probe_report"]["video_standard"] == "PAL"
    assert output["probe_report"]["action_space"]["meanings"] == ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    assert output["probe_report"]["observation_space"]["shape"] == [250, 160, 3]
    assert output["probe_report"]["render"] == {"shape": [250, 160, 3], "dtype": "uint8"}
    assert output["probe_report"]["reset_episode_frame_number"] == 8
    assert output["probe_report"]["post_step_episode_frame_number"] == 9
    assert output["env_config"]["env"]["action_order"] == ACTION_LABELS
    assert output["env_config"]["env"]["noop_action"] == "noop"
    assert output["env_config"]["env"]["frame_stack"] == 4
    assert native_env.closed is True


@pytest.mark.parametrize("config_path", [
    "configs/examples/gymnasium/air_raid/latency_bench_eval.yaml",
    "configs/examples/gymnasium/air_raid/small_model_train.yaml",
    "configs/examples/gymnasium/air_raid/small_model_eval.yaml",
])
def test_air_raid_configs_share_zero_latency_four_frame_contract(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_air_raid.GYMNASIUM_AIR_RAID_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.tasks.air_raid"]
    assert env_cfg["make_kwargs"]["base_env_id"] == "ALE/AirRaid-v5"
    assert env_cfg["make_kwargs"]["screen_size"] == 84
    assert env_cfg["make_kwargs"]["noop_max"] == 0
    assert env_cfg["make_kwargs"]["base_make_kwargs"]["frameskip"] == 1
    assert env_cfg["make_kwargs"]["base_make_kwargs"]["repeat_action_probability"] == 0.0
    assert env_cfg["action_order"] == ACTION_LABELS
    assert env_cfg["action_map"] == dict(zip(ACTION_LABELS, range(6)))
    assert env_cfg["noop_action"] == "noop"
    assert env_cfg["frame_stack"] == 4
    assert clock.obs_stride_raw_frames == 4
    assert clock.env_fps == 50
    assert config["latency"] == {"method": "zero", "add_latency_info": False}


def test_air_raid_random_noop_eval_variant_is_explicit():
    config = load_config("configs/examples/gymnasium/air_raid/small_model_eval_random_noop.yaml")

    assert config["env"]["make_kwargs"]["noop_max"] == 30
    assert config["env"]["make_kwargs"]["fire_reset"] is True
    assert config["evaluation"]["eval_max_steps"] == 3600
    assert config["logging"]["output_dir"] == "outputs/examples/gymnasium_air_raid/eval_random_noop"


def test_air_raid_starvla_eval_uses_random_noop_fire_reset():
    config = load_config("configs/examples/gymnasium/air_raid/starvla_eval.yaml")

    assert config["env"]["make_kwargs"]["noop_max"] == 30
    assert config["env"]["make_kwargs"]["fire_reset"] is True
    assert config["evaluation"]["eval_max_steps"] == 3600


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/air_raid/latency_bench_eval.yaml",
        "configs/examples/gymnasium/air_raid/small_model_eval.yaml",
        "configs/examples/gymnasium/air_raid/starvla_eval.yaml",
    ],
)
def test_air_raid_eval_budget_is_3600_raw_frames(config_path):
    config = load_config(config_path)

    assert config["evaluation"]["eval_max_steps"] == 3600
