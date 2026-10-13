from __future__ import annotations

from latency_bench.envs.factory import resolve_env_config
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import scripts.scaffold_gymnasium_lunar_lander as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper


ACTION_LABELS = [
    "do_nothing",
    "fire_left_orientation_engine",
    "fire_main_engine",
    "fire_right_orientation_engine",
]


class _LunarLanderProbeEnv(gym.Env):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

    def __init__(self) -> None:
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
        self.spec = SimpleNamespace(
            id="LunarLander-v2",
            entry_point="fake:LunarLander",
            max_episode_steps=1000,
            kwargs={
                "render_mode": "rgb_array",
                "continuous": False,
                "gravity": -10.0,
                "enable_wind": False,
                "wind_power": 15.0,
                "turbulence_power": 1.5,
            },
        )
        self.actions: list[int] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(8, dtype=np.float32), -1.25, False, False, {"source": "native"}

    def render(self):
        return np.ones((400, 600, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_scaffold_reports_native_discrete_api_and_frozen_contract(monkeypatch, capsys):
    native_env = _LunarLanderProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(scaffold.gym, "make", fake_make)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "box2d-py": "2.3.5"}[package],
    )
    scaffold.main([])
    output = yaml.safe_load(capsys.readouterr().out)

    assert captured == {
        "env_id": "LunarLander-v2",
        "kwargs": {
            "render_mode": "rgb_array",
            "continuous": False,
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        },
    }
    assert output["probe_report"]["action_space"] == {
        "type": "Discrete",
        "n": 4,
        "dtype": "int64",
    }
    assert output["probe_report"]["native_env_fps"] == 50
    assert output["probe_report"]["render"] == {"shape": [400, 600, 3], "dtype": "uint8"}
    assert output["probe_report"]["noop_step"] == {
        "action": 0,
        "reward": -1.25,
        "terminated": False,
        "truncated": False,
        "info_keys": ["source"],
    }
    assert output["probe_report"]["action_catalog"]["source"] == "gymnasium_api"
    assert output["env_config"]["env"]["action_order"] == ACTION_LABELS
    assert output["env_config"]["env"]["action_map"] == dict(zip(ACTION_LABELS, range(4)))
    assert output["env_config"]["env"]["noop_action"] == "do_nothing"
    assert native_env.closed is True


def test_l1_scheduler_applies_do_nothing_before_issued_engine_action():
    native_env = _LunarLanderProbeEnv()
    env = GymnasiumActionCatalogWrapper(native_env, [0, 1, 2, 3])
    env = ActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=1,
        env_fps=50,
        obs_fps=50,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=0,
    )
    env.reset(seed=0)

    env.step(2)
    env.step(3)

    assert native_env.actions == [0, 2]


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/lunar_lander/latency_bench_eval.yaml",
        "configs/examples/gymnasium/lunar_lander/small_model_train.yaml",
        "configs/examples/gymnasium/lunar_lander/small_model_eval.yaml",
        "configs/examples/gymnasium/lunar_lander/starvla_eval.yaml",
    ],
)
def test_lunar_lander_configs_share_direct_task_contract_and_l1_latency(config_path):
    config = load_config(config_path)
    resolve_env_config(config)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == "LunarLander-v2"
    assert env_cfg["registration_imports"] == []
    assert env_cfg["make_kwargs"] == {
        "render_mode": "rgb_array",
        "continuous": False,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    }
    assert env_cfg["action_order"] == ACTION_LABELS
    assert env_cfg["action_map"] == dict(zip(ACTION_LABELS, range(4)))
    assert env_cfg["noop_action"] == "do_nothing"
    assert clock.obs_stride_raw_frames == 1
    assert clock.env_fps == 50
    assert clock.latency_ms_to_raw_frames(config["latency"]["fixed_latency_ms"]) == 1
