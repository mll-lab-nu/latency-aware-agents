from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.tasks.half_cheetah as gymnasium_half_cheetah
import scripts.scaffold_gymnasium_half_cheetah as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.tasks.contracts import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


class _HalfCheetahProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.05
        self.frame_skip = 5
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
        self.spec = SimpleNamespace(
            id="HalfCheetah-v4",
            entry_point="fake:HalfCheetahEnv",
            max_episode_steps=1000,
            kwargs={"render_mode": "rgb_array"},
        )
        self.closed = False
        self.actions = []

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.arange(17, dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(17, dtype=np.float64), -0.1, False, False, {"x_position": 0.0}

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_half_cheetah_wrapper_clips_and_records_native_torques(monkeypatch):
    native_env = _HalfCheetahProbeEnv()
    captured = {}

    def fake_make(env_id, **kwargs):
        captured.update({"env_id": env_id, "kwargs": kwargs})
        return native_env

    monkeypatch.setattr(gymnasium_half_cheetah.gym, "make", fake_make)
    env = gymnasium_half_cheetah.HalfCheetahContinuousActionEnv(
        base_env_id="HalfCheetah-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )
    _, _, _, _, info = env.step([-2.0, -0.5, 0.0, 0.25, 0.75, 2.0])

    expected = np.asarray([-1.0, -0.5, 0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    assert captured == {
        "env_id": "HalfCheetah-v4",
        "kwargs": {"render_mode": "rgb_array"},
    }
    assert env.action_space == native_env.action_space
    assert np.array_equal(native_env.actions[0], expected)
    assert info["applied_action"] == expected.tolist()


def test_half_cheetah_registered_entry_point_is_factory():
    assert (
        gym.spec(gymnasium_half_cheetah.GYMNASIUM_HALF_CHEETAH_ENV_ID).entry_point
        is gymnasium_half_cheetah.make_latency_sensitive_half_cheetah_env
    )


def test_half_cheetah_probe_reports_native_contract(monkeypatch):
    native_env = _HalfCheetahProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package],
    )

    report = scaffold.build_half_cheetah_probe_report(env_id="HalfCheetah-v4", render_mode="rgb_array")

    assert report["action_space"]["shape"] == [6]
    assert report["observation_space"]["shape"] == [17]
    assert report["native_env_fps"] == 20.0
    assert report["action_catalog"]["labels"] == scaffold.ACTION_LABELS
    assert report["action_catalog"]["noop_action"] == scaffold.NOOP_ACTION
    assert native_env.closed is True


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/half_cheetah/latency_bench_eval.yaml",
        "configs/examples/gymnasium/half_cheetah/small_model_train.yaml",
        "configs/examples/gymnasium/half_cheetah/small_model_eval.yaml",
        "configs/examples/gymnasium/half_cheetah/latency_bench_eval.yaml",
        "configs/examples/gymnasium/half_cheetah/small_model_train.yaml",
        "configs/examples/gymnasium/half_cheetah/small_model_eval.yaml",
        "configs/examples/gymnasium/half_cheetah/starvla_eval.yaml",
    ],
)
def test_half_cheetah_configs_share_native_zero_latency_contract(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_half_cheetah.GYMNASIUM_HALF_CHEETAH_ENV_ID
    assert env_cfg["registration_imports"] == ["latency_bench.envs.tasks.half_cheetah"]
    assert env_cfg["make_kwargs"] == {
        "base_env_id": "HalfCheetah-v4",
        "render_mode": "rgb_array",
        "base_make_kwargs": {},
    }
    assert env_cfg["action_space"] == {
        "type": "box",
        "labels": scaffold.ACTION_LABELS,
        "low": [-1.0] * 6,
        "high": [1.0] * 6,
        "dtype": "float32",
    }
    assert env_cfg["noop_action"] == scaffold.NOOP_ACTION
    assert clock.obs_stride_raw_frames == 1
    assert config["latency"]["method"] == "zero"


def test_half_cheetah_training_uses_sample_factory_mujoco_recipe():
    config = load_config("configs/examples/gymnasium/half_cheetah/small_model_train.yaml")
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["training"]["train_for_env_steps"] == 10_000_000
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.async_rl is False
    assert contract["action_space"]["labels"] == scaffold.ACTION_LABELS


def test_half_cheetah_eval_sweeps_five_latencies():
    config = load_config("configs/examples/gymnasium/half_cheetah/small_model_eval.yaml")

    assert config["evaluation"]["eval_latency_values"] == [0, 1, 2, 3, 4]
    assert config["evaluation"]["eval_episodes"] == 10


def test_half_cheetah_preserves_native_observation_order():
    config = load_config("configs/examples/gymnasium/half_cheetah/small_model_train.yaml")

    assert config["env"]["task_name"] == "half_cheetah"
    assert config["env"]["state_space"]["labels"] == [
        "front_tip_height",
        "front_tip_angle",
        "back_thigh_angle",
        "back_shin_angle",
        "back_foot_angle",
        "front_thigh_angle",
        "front_shin_angle",
        "front_foot_angle",
        "front_tip_x_velocity",
        "front_tip_z_velocity",
        "front_tip_angular_velocity",
        "back_thigh_angular_velocity",
        "back_shin_angular_velocity",
        "back_foot_angular_velocity",
        "front_thigh_angular_velocity",
        "front_shin_angular_velocity",
        "front_foot_angular_velocity",
    ]
