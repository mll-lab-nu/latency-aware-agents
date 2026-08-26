from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import yaml

import latency_bench.envs.gymnasium_hopper as gymnasium_hopper
import scripts.scaffold_gymnasium_hopper as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import GymnasiumActionCatalogWrapper, GymnasiumActionLatencyWrapper


class _HopperProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.008
        self.frame_skip = 4
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 125}
        self.spec = SimpleNamespace(id="Hopper-v4", entry_point="fake:HopperEnv", max_episode_steps=1000, kwargs={"render_mode": "rgb_array"})
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.arange(11, dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(11, dtype=np.float64), 1.25, False, False, {"x_position": 0.1}

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_hopper_wrapper_maps_catalog_id_to_native_torque_vector(monkeypatch):
    native_env = _HopperProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(gymnasium_hopper.gym, "make", fake_make)
    env = gymnasium_hopper.make_latency_sensitive_hopper_env(
        base_env_id="Hopper-v4",
        render_mode="rgb_array",
        action_values=scaffold.USER_APPROVED_ACTION_VALUES,
        base_make_kwargs={"ctrl_cost_weight": 0.001},
    )
    env.reset(seed=4)
    env.step(23)

    assert captured == {"env_id": "Hopper-v4", "kwargs": {"render_mode": "rgb_array", "ctrl_cost_weight": 0.001}}
    assert env.action_space == gym.spaces.Discrete(27)
    assert native_env.actions[0].dtype == np.float32
    assert np.array_equal(native_env.actions[0], np.asarray([1.0, 0.0, 1.0], dtype=np.float32))
    assert env.render().shape == (480, 480, 3)


def test_l1_scheduler_applies_zero_torque_before_issued_vector(monkeypatch):
    native_env = _HopperProbeEnv()
    monkeypatch.setattr(gymnasium_hopper.gym, "make", lambda *args, **kwargs: native_env)
    task_env = gymnasium_hopper.make_latency_sensitive_hopper_env(
        base_env_id="Hopper-v4",
        render_mode="rgb_array",
        action_values=scaffold.USER_APPROVED_ACTION_VALUES,
        base_make_kwargs={},
    )
    env = GymnasiumActionCatalogWrapper(task_env, list(range(27)))
    env = GymnasiumActionLatencyWrapper(env, fixed_latency_raw_frames=1, env_fps=125, obs_fps=125, hold_policy="hold", ordering_policy="issue_order_fifo", noop_action=13)
    env.reset(seed=0)
    env.step(26)
    env.step(0)

    assert np.array_equal(native_env.actions[0], np.zeros(3, dtype=np.float32))
    assert np.array_equal(native_env.actions[1], np.ones(3, dtype=np.float32))


def test_scaffold_reports_runtime_and_user_approved_catalog(monkeypatch, capsys):
    native_env = _HopperProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(scaffold.importlib.metadata, "version", lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package])
    scaffold.main([])
    output = yaml.safe_load(capsys.readouterr().out)

    assert output["probe_report"]["native_env_fps"] == 125.0
    assert output["probe_report"]["unwrapped_frame_skip"] == 4
    assert output["probe_report"]["model_timestep_seconds"] == 0.002
    assert output["probe_report"]["action_space"]["shape"] == [3]
    assert output["probe_report"]["render"] == {"shape": [480, 480, 3], "dtype": "uint8"}
    assert output["probe_report"]["action_catalog"]["source"] == "user_approved"
    assert len(output["env_config"]["env"]["action_order"]) == 27
    assert output["env_config"]["env"]["noop_action"] == "t_z_l_z_f_z"
    assert native_env.closed is True


@pytest.mark.parametrize("config_path", [
    "configs/examples/gymnasium/hopper/latency_bench_eval.yaml",
    "configs/examples/gymnasium/hopper/small_model_train.yaml",
    "configs/examples/gymnasium/hopper/small_model_eval.yaml",
    "configs/examples/gymnasium/hopper/starvla_eval.yaml",
])
def test_hopper_configs_share_contract_and_zero_latency(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_hopper.GYMNASIUM_HOPPER_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.gymnasium_hopper"]
    assert env_cfg["make_kwargs"]["action_values"] == scaffold.USER_APPROVED_ACTION_VALUES
    assert env_cfg["action_order"] == scaffold.USER_APPROVED_ACTION_LABELS
    assert env_cfg["action_map"] == dict(zip(scaffold.USER_APPROVED_ACTION_LABELS, range(27)))
    assert env_cfg["noop_action"] == "t_z_l_z_f_z"
    assert clock.obs_stride_raw_frames == 1
    assert clock.env_fps == 125
    assert config["latency"] == {"method": "zero", "add_latency_info": False}
