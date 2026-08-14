from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.gymnasium_mujoco as gymnasium_mujoco
import scripts.scaffold_gymnasium_mujoco as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import (
    GymnasiumActionCatalogWrapper,
    GymnasiumActionLatencyWrapper,
)


FORCE_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]
ACTION_LABELS = [
    "push_left_hard",
    "push_left_soft",
    "zero_force",
    "push_right_soft",
    "push_right_hard",
]


class _InvertedPendulumProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-3.0, 3.0, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.02
        self.frame_skip = 2
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
        self.spec = SimpleNamespace(
            id="InvertedPendulum-v4",
            entry_point="fake:InvertedPendulumEnv",
            max_episode_steps=1000,
            kwargs={"render_mode": "rgb_array"},
        )
        self.terminated = True
        self.step_actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1.0, 2.0, 3.0, 4.0]), {"seed": seed}

    def step(self, action):
        self.step_actions.append(action)
        return (
            np.array([5.0, 6.0, 7.0, 8.0]),
            1.25,
            self.terminated,
            False,
            {"source": "native"},
        )

    def render(self):
        return np.ones((8, 9, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_inverted_pendulum_wrapper_maps_catalog_ids_to_native_force_values(monkeypatch):
    native_env = _InvertedPendulumProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(gymnasium_mujoco.gym, "make", fake_make)
    env = gymnasium_mujoco.make_latency_sensitive_inverted_pendulum_env(
        base_env_id="InvertedPendulum-v4",
        render_mode="rgb_array",
        force_values=FORCE_VALUES,
        base_make_kwargs={"healthy_reward": 1.0},
    )

    observation, info = env.reset(seed=4)
    stepped = env.step(3)

    assert captured == {
        "env_id": "InvertedPendulum-v4",
        "kwargs": {"render_mode": "rgb_array", "healthy_reward": 1.0},
    }
    assert env.action_space == gym.spaces.Discrete(5)
    assert np.array_equal(observation, np.array([1.0, 2.0, 3.0, 4.0]))
    assert info == {"seed": 4}
    assert native_env.step_actions[0].dtype == np.float32
    assert np.array_equal(native_env.step_actions[0], np.asarray([1.5], dtype=np.float32))
    assert np.array_equal(stepped[0], np.array([5.0, 6.0, 7.0, 8.0]))
    assert stepped[1:] == (1.25, True, False, {"source": "native"})
    assert env.render().shape == (8, 9, 3)


def test_l1_scheduler_applies_zero_force_before_the_issued_force(monkeypatch):
    native_env = _InvertedPendulumProbeEnv()
    native_env.terminated = False
    monkeypatch.setattr(gymnasium_mujoco.gym, "make", lambda *args, **kwargs: native_env)
    task_env = gymnasium_mujoco.make_latency_sensitive_inverted_pendulum_env(
        base_env_id="InvertedPendulum-v4",
        render_mode="rgb_array",
        force_values=FORCE_VALUES,
        base_make_kwargs={},
    )
    env = GymnasiumActionCatalogWrapper(task_env, [0, 1, 2, 3, 4])
    env = GymnasiumActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=1,
        env_fps=25,
        obs_fps=25,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=2,
    )
    env.reset(seed=0)

    env.step(4)
    env.step(4)

    assert len(native_env.step_actions) == 2
    assert np.array_equal(native_env.step_actions[0], np.asarray([0.0], dtype=np.float32))
    assert np.array_equal(native_env.step_actions[1], np.asarray([3.0], dtype=np.float32))


def test_scaffold_reports_native_api_and_marks_force_catalog_user_approved(monkeypatch, capsys):
    native_env = _InvertedPendulumProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(scaffold.gym, "make", fake_make)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package],
    )
    scaffold.main(["--base-make-kwargs-json", '{"healthy_reward": 1.0}'])
    output = yaml.safe_load(capsys.readouterr().out)

    assert captured == {
        "env_id": "InvertedPendulum-v4",
        "kwargs": {"render_mode": "rgb_array", "healthy_reward": 1.0},
    }
    assert output["probe_report"]["action_space"] == {
        "type": "Box",
        "shape": [1],
        "low": [-3.0],
        "high": [3.0],
        "dtype": "float32",
    }
    assert output["probe_report"]["native_env_fps"] == 50.0
    assert output["probe_report"]["unwrapped_frame_skip"] == 2
    assert output["probe_report"]["model_timestep_seconds"] == 0.01
    assert output["probe_report"]["resolved_spec"]["max_episode_steps"] == 1000
    assert output["probe_report"]["zero_action_step"] == {
        "action": [0.0],
        "reward": 1.25,
        "terminated": True,
        "truncated": False,
        "info_keys": ["source"],
    }
    assert output["probe_report"]["render"] == {
        "shape": [8, 9, 3],
        "dtype": "uint8",
    }
    assert output["probe_report"]["action_catalog"]["source"] == "user_approved"
    assert output["env_config"]["env"]["action_order"] == ACTION_LABELS
    assert output["env_config"]["env"]["action_map"] == dict(zip(ACTION_LABELS, range(5)))
    assert output["env_config"]["env"]["noop_action"] == "zero_force"
    assert native_env.closed is True


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/inverted_pendulum/latency_bench_eval.yaml",
        "configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml",
        "configs/examples/gymnasium/inverted_pendulum/small_model_eval.yaml",
        "configs/examples/gymnasium/inverted_pendulum/starvla_eval.yaml",
    ],
)
def test_inverted_pendulum_configs_share_task_contract_and_l1_latency(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_mujoco.GYMNASIUM_MUJOCO_INVERTED_PENDULUM_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.gymnasium_mujoco"]
    assert env_cfg["make_kwargs"] == {
        "base_env_id": "InvertedPendulum-v4",
        "render_mode": "rgb_array",
        "force_values": FORCE_VALUES,
        "base_make_kwargs": {},
    }
    assert env_cfg["action_order"] == ACTION_LABELS
    assert env_cfg["action_map"] == dict(zip(ACTION_LABELS, range(5)))
    assert env_cfg["noop_action"] == "zero_force"
    assert clock.obs_stride_raw_frames == 1
    assert clock.env_fps == 25
    assert clock.latency_ms_to_raw_frames(config["latency"]["fixed_latency_ms"]) == 1
