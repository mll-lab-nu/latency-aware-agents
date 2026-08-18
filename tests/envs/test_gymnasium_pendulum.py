from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.gymnasium_pendulum as gymnasium_pendulum
import scripts.scaffold_gymnasium_pendulum as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import (
    GymnasiumActionCatalogWrapper,
    GymnasiumActionLatencyWrapper,
)


TORQUE_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
ACTION_LABELS = [
    "torque_clockwise_hard",
    "torque_clockwise_soft",
    "zero_torque",
    "torque_counterclockwise_soft",
    "torque_counterclockwise_hard",
]


class _PendulumProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=np.asarray([-1.0, -1.0, -8.0], dtype=np.float32),
        high=np.asarray([1.0, 1.0, 8.0], dtype=np.float32),
        dtype=np.float32,
    )

    def __init__(self) -> None:
        self.dt = 0.05
        self.max_torque = 2.0
        self.max_speed = 8
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
        self.spec = SimpleNamespace(
            id="Pendulum-v1",
            entry_point="fake:PendulumEnv",
            max_episode_steps=200,
            kwargs={"render_mode": "rgb_array", "g": 10.0},
        )
        self.step_actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray([0.5, 0.5, -0.25], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self.step_actions.append(action)
        return (
            np.asarray([0.6, 0.4, 0.1], dtype=np.float32),
            -0.75,
            False,
            False,
            {"source": "native"},
        )

    def render(self):
        return np.ones((12, 13, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_pendulum_wrapper_maps_catalog_ids_to_native_torque_values(monkeypatch):
    native_env = _PendulumProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(gymnasium_pendulum.gym, "make", fake_make)
    env = gymnasium_pendulum.make_latency_sensitive_pendulum_env(
        base_env_id="Pendulum-v1",
        render_mode="rgb_array",
        torque_values=TORQUE_VALUES,
        base_make_kwargs={"g": 10.0},
    )

    observation, info = env.reset(seed=4)
    stepped = env.step(3)

    assert captured == {
        "env_id": "Pendulum-v1",
        "kwargs": {"render_mode": "rgb_array", "g": 10.0},
    }
    assert env.action_space == gym.spaces.Discrete(5)
    assert np.array_equal(observation, np.asarray([0.5, 0.5, -0.25], dtype=np.float32))
    assert info == {"seed": 4}
    assert native_env.step_actions[0].dtype == np.float32
    assert np.array_equal(native_env.step_actions[0], np.asarray([1.0], dtype=np.float32))
    assert stepped[1:] == (-0.75, False, False, {"source": "native"})
    assert env.render().shape == (12, 13, 3)


def test_l1_scheduler_applies_zero_torque_before_the_issued_torque(monkeypatch):
    native_env = _PendulumProbeEnv()
    monkeypatch.setattr(gymnasium_pendulum.gym, "make", lambda *args, **kwargs: native_env)
    task_env = gymnasium_pendulum.make_latency_sensitive_pendulum_env(
        base_env_id="Pendulum-v1",
        render_mode="rgb_array",
        torque_values=TORQUE_VALUES,
        base_make_kwargs={"g": 10.0},
    )
    env = GymnasiumActionCatalogWrapper(task_env, [0, 1, 2, 3, 4])
    env = GymnasiumActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=1,
        env_fps=20,
        obs_fps=20,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=2,
    )
    env.reset(seed=0)

    env.step(4)
    env.step(4)

    assert len(native_env.step_actions) == 2
    assert np.array_equal(native_env.step_actions[0], np.asarray([0.0], dtype=np.float32))
    assert np.array_equal(native_env.step_actions[1], np.asarray([2.0], dtype=np.float32))


def test_scaffold_reports_native_api_and_marks_torque_catalog_user_approved(monkeypatch, capsys):
    native_env = _PendulumProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(scaffold.gym, "make", fake_make)
    monkeypatch.setattr(scaffold.importlib.metadata, "version", lambda package: "0.29.1")
    scaffold.main([])
    output = yaml.safe_load(capsys.readouterr().out)

    assert captured == {
        "env_id": "Pendulum-v1",
        "kwargs": {"render_mode": "rgb_array", "g": 10.0},
    }
    assert output["probe_report"]["action_space"] == {
        "type": "Box",
        "shape": [1],
        "low": [-2.0],
        "high": [2.0],
        "dtype": "float32",
    }
    assert output["probe_report"]["native_env_fps"] == 20.0
    assert output["probe_report"]["metadata"]["render_fps"] == 30
    assert output["probe_report"]["resolved_spec"]["max_episode_steps"] == 200
    assert output["probe_report"]["zero_action_step"] == {
        "action": [0.0],
        "reward": -0.75,
        "terminated": False,
        "truncated": False,
        "info_keys": ["source"],
    }
    assert output["probe_report"]["render"] == {
        "shape": [12, 13, 3],
        "dtype": "uint8",
    }
    assert output["probe_report"]["action_catalog"]["source"] == "user_approved"
    assert output["env_config"]["env"]["action_order"] == ACTION_LABELS
    assert output["env_config"]["env"]["action_map"] == dict(zip(ACTION_LABELS, range(5)))
    assert output["env_config"]["env"]["noop_action"] == "zero_torque"
    assert native_env.closed is True


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/pendulum/latency_bench_eval.yaml",
        "configs/examples/gymnasium/pendulum/small_model_train.yaml",
        "configs/examples/gymnasium/pendulum/small_model_eval.yaml",
        "configs/examples/gymnasium/pendulum/starvla_eval.yaml",
    ],
)
def test_pendulum_configs_share_task_contract_and_l1_latency(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_pendulum.GYMNASIUM_PENDULUM_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.gymnasium_pendulum"]
    assert env_cfg["make_kwargs"] == {
        "base_env_id": "Pendulum-v1",
        "render_mode": "rgb_array",
        "torque_values": TORQUE_VALUES,
        "base_make_kwargs": {"g": 10.0},
    }
    assert env_cfg["action_order"] == ACTION_LABELS
    assert env_cfg["action_map"] == dict(zip(ACTION_LABELS, range(5)))
    assert env_cfg["noop_action"] == "zero_torque"
    assert clock.obs_stride_raw_frames == 1
    assert clock.env_fps == 20
    assert clock.latency_ms_to_raw_frames(config["latency"]["fixed_latency_ms"]) == 1
