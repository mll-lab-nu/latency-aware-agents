from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.gymnasium_hopper as hopper
import scripts.scaffold_gymnasium_hopper as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


class _HopperProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.008
        self.frame_skip = 4
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 125}
        self.spec = SimpleNamespace(
            id="Hopper-v4",
            entry_point="fake:HopperEnv",
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
        return np.arange(11, dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(11, dtype=np.float64), 1.0, False, False, {"x_position": 0.1}

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_hopper_wrapper_clips_and_records_native_torques(monkeypatch):
    native = _HopperProbeEnv()
    monkeypatch.setattr(hopper.gym, "make", lambda *args, **kwargs: native)
    env = hopper.HopperEnv(
        base_env_id="Hopper-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )

    _, _, _, _, info = env.step([1.4, -1.2, 0.25])

    expected = np.asarray([1.0, -1.0, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(native.actions[0], expected)
    assert info["applied_action"] == pytest.approx(expected.tolist())


def test_hopper_registered_entry_point_is_factory():
    assert (
        gym.spec(hopper.GYMNASIUM_HOPPER_ENV_ID).entry_point
        is hopper.make_latency_sensitive_hopper_env
    )


def test_hopper_probe_reports_native_contract(monkeypatch):
    native = _HopperProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package],
    )

    report = scaffold.build_hopper_probe_report(
        env_id="Hopper-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )

    assert report["action_space"]["shape"] == [3]
    assert report["observation_space"]["shape"] == [11]
    assert report["native_env_fps"] == 125.0
    assert report["state_labels"] == scaffold.STATE_LABELS
    assert native.closed is True


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/hopper/small_model_train.yaml",
        "configs/examples/gymnasium/hopper/small_model_eval.yaml",
        "configs/examples/gymnasium/hopper/latency_bench_eval.yaml",
        "configs/examples/gymnasium/hopper/starvla_eval.yaml",
    ],
)
def test_hopper_configs_share_contract(config_path):
    config = load_config(config_path)
    env = config["env"]
    clock = EnvClock(env_fps=env["env_fps"], obs_fps=env["obs_fps"])

    assert env["env_id"] == hopper.GYMNASIUM_HOPPER_ENV_ID
    assert env["task_name"] == "hopper"
    assert env["state_space"]["labels"] == scaffold.STATE_LABELS
    assert env["action_space"] == {
        "type": "box",
        "labels": scaffold.ACTION_LABELS,
        "low": [-1.0] * 3,
        "high": [1.0] * 3,
        "dtype": "float32",
    }
    assert env["noop_action"] == scaffold.NOOP_ACTION
    assert clock.obs_stride_raw_frames == 1
    assert config["latency"] == {"method": "zero", "add_latency_info": False}


def test_hopper_training_uses_sample_factory_mujoco_recipe():
    config = load_config("configs/examples/gymnasium/hopper/small_model_train.yaml")
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["experiment"]["seed"] == 3333
    assert config["training"]["train_for_env_steps"] == 10_000_000
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.async_rl is False
    assert contract["state_space"]["labels"] == scaffold.STATE_LABELS
    assert contract["action_space"]["labels"] == scaffold.ACTION_LABELS


def test_hopper_starvla_eval_uses_final_checkpoint():
    config = load_config("configs/examples/gymnasium/hopper/starvla_eval.yaml")

    assert config["policy"]["checkpoint_path"].endswith("steps_5000_pytorch_model.pt")
    assert "continuous_projector_sft_5k" in config["policy"]["checkpoint_path"]
    assert "continuous_projector_sft_5k" in config["policy"]["model_config_path"]
    assert config["logging"]["output_dir"].endswith("starvla_eval_continuous_projector")
    assert config["evaluation"]["eval_latency_values"] == [0, 2]
    assert config["evaluation"]["eval_episodes"] == 20
    assert config["evaluation"]["eval_max_steps"] == 1000
