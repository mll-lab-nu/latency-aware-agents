from __future__ import annotations

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.gymnasium_hopper_continuous as hopper_continuous
from latency_bench.core.config import load_config
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


class _HopperProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float64)

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.actions.append(np.asarray(action))
        return np.zeros(11), 1.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return np.zeros(11), {}


def test_hopper_continuous_clips_native_torques(monkeypatch):
    native = _HopperProbeEnv()
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: native)
    env = hopper_continuous.HopperContinuousActionEnv(
        base_env_id="Hopper-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )

    _, _, _, _, info = env.step([1.4, -1.2, 0.25])

    np.testing.assert_array_equal(native.actions[0], np.asarray([1.0, -1.0, 0.25], dtype=np.float32))
    assert info["applied_action"] == pytest.approx([1.0, -1.0, 0.25])


def test_hopper_continuous_config_matches_hf_mujoco_recipe():
    config = load_config("configs/examples/gymnasium/hopper_continuous/small_model_train.yaml")
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["experiment"]["seed"] == 3333
    assert config["latency"] == {"method": "zero", "add_latency_info": False}
    assert contract["action_space"]["type"] == "box"
    assert contract["action_space"]["labels"] == ["thigh_torque", "leg_torque", "foot_torque"]
    assert json.loads(cfg.gym_noop_action_json) == [0.0, 0.0, 0.0]
    assert cfg.train_for_env_steps == 10_000_000
    assert cfg.num_workers == 8
    assert cfg.num_envs_per_worker == 8
    assert cfg.worker_num_splits == 2
    assert cfg.batch_size == 1024
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.env_framestack == 4
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.lr_schedule == "linear_decay"
    assert cfg.ppo_clip_ratio == pytest.approx(0.2)
    assert cfg.ppo_clip_value == pytest.approx(1.0)
    assert cfg.kl_loss_coeff == pytest.approx(0.1)
    assert cfg.value_bootstrap is False
    assert cfg.async_rl is False
