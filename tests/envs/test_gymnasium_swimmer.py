from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.gymnasium_swimmer as gymnasium_swimmer
import scripts.scaffold_gymnasium_swimmer as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


class _SwimmerProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.04
        self.frame_skip = 4
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 25}
        self.spec = SimpleNamespace(
            id="Swimmer-v4",
            entry_point="fake:SwimmerEnv",
            max_episode_steps=1000,
            kwargs={"render_mode": "rgb_array"},
        )
        self.actions = []
        self.closed = False

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.arange(8, dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(8, dtype=np.float64), 1.0, False, False, {"x_position": 0.0}

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_swimmer_wrapper_clips_and_records_native_torques(monkeypatch):
    native_env = _SwimmerProbeEnv()
    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return native_env

    monkeypatch.setattr(gymnasium_swimmer, "SwimmerV4Mujoco3Env", fake_env)
    env = gymnasium_swimmer.SwimmerContinuousActionEnv(
        render_mode="rgb_array",
        base_make_kwargs={"exclude_current_positions_from_observation": True},
    )
    _, _, _, _, info = env.step([-2.0, 2.0])

    expected = np.asarray([-1.0, 1.0], dtype=np.float32)
    assert captured == {
        "render_mode": "rgb_array",
        "exclude_current_positions_from_observation": True,
    }
    assert np.array_equal(native_env.actions[0], expected)
    assert info["applied_action"] == expected.tolist()


def test_swimmer_registered_entry_point_is_task_factory():
    assert (
        gym.spec(gymnasium_swimmer.GYMNASIUM_SWIMMER_ENV_ID).entry_point
        is gymnasium_swimmer.make_latency_sensitive_swimmer_env
    )


def test_swimmer_probe_reports_native_contract(monkeypatch):
    native_env = _SwimmerProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package],
    )

    report = scaffold.build_swimmer_probe_report(env_id="Swimmer-v4", render_mode="rgb_array")

    assert report["action_space"]["shape"] == [2]
    assert report["observation_space"]["shape"] == [8]
    assert report["native_env_fps"] == 25.0
    assert report["action_catalog"]["labels"] == scaffold.ACTION_LABELS
    assert report["state_labels"] == scaffold.STATE_LABELS
    assert native_env.closed is True


def test_swimmer_configs_preserve_native_rgb_state_contract():
    config_paths = [
        "configs/examples/gymnasium/swimmer/small_model_train.yaml",
        "configs/examples/gymnasium/swimmer/small_model_eval.yaml",
        "configs/examples/gymnasium/swimmer/latency_bench_eval.yaml",
        "configs/examples/gymnasium/swimmer/starvla_eval.yaml",
    ]
    configs = [load_config(path) for path in config_paths]
    envs = [config["env"] for config in configs]

    assert envs[1:] == envs[:-1]
    env = envs[0]
    assert env["env_id"] == gymnasium_swimmer.GYMNASIUM_SWIMMER_ENV_ID
    assert env["registration_imports"] == ["latency_bench.envs.gymnasium_swimmer"]
    assert env["make_kwargs"] == {
        "render_mode": "rgb_array",
        "base_make_kwargs": {"exclude_current_positions_from_observation": True},
    }
    assert EnvClock(env_fps=env["env_fps"], obs_fps=env["obs_fps"]).obs_stride_raw_frames == 1
    assert env["frame_stack"] == 4
    assert env["action_space"] == {
        "type": "box",
        "labels": scaffold.ACTION_LABELS,
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "dtype": "float32",
    }
    assert env["noop_action"] == scaffold.NOOP_ACTION
    assert env["state_space"]["labels"] == scaffold.STATE_LABELS
    assert configs[0]["latency"]["method"] == "zero"


def test_swimmer_training_uses_vendored_sample_factory_mujoco_recipe():
    config = load_config("configs/examples/gymnasium/swimmer/small_model_train.yaml")
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["training"]["train_for_env_steps"] == 10_000_000
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.async_rl is True
    assert contract["action_space"]["labels"] == scaffold.ACTION_LABELS
    assert contract["state_space"]["labels"] == scaffold.STATE_LABELS


def test_swimmer_eval_contracts_cover_expected_latencies():
    small_model = load_config("configs/examples/gymnasium/swimmer/small_model_eval.yaml")
    starvla = load_config("configs/examples/gymnasium/swimmer/starvla_eval.yaml")

    assert small_model["evaluation"]["eval_latency_values"] == [0, 1, 2, 3, 4]
    assert small_model["evaluation"]["eval_episodes"] == 10
    assert starvla["evaluation"]["eval_latency_values"] == [0, 2]
    assert starvla["evaluation"]["eval_episodes"] == 20
