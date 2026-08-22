from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.gymnasium_walker2d as gymnasium_walker2d
import scripts.scaffold_gymnasium_walker2d as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    encode_action_for_env,
    gymnasium_continuous_action_spec,
)
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from latency_bench.policy.starvla import action_from_starvla_payload
from training.gymnasium_sf.integration import (
    GymnasiumActionLatencyWrapper,
    build_cli_args_from_config,
    build_train_cfg_from_config,
)


class _Walker2dProbeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float64)

    def __init__(self) -> None:
        self.dt = 0.008
        self.frame_skip = 4
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 125}
        self.spec = SimpleNamespace(
            id="Walker2d-v4",
            entry_point="fake:Walker2dEnv",
            max_episode_steps=1000,
            kwargs={"render_mode": "rgb_array"},
        )
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.arange(17, dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return np.ones(17, dtype=np.float64), 1.0, False, False, {"x_position": 0.1}

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_walker2d_wrapper_passes_six_native_torques(monkeypatch):
    native_env = _Walker2dProbeEnv()
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return native_env

    monkeypatch.setattr(gymnasium_walker2d.gym, "make", fake_make)
    env = gymnasium_walker2d.make_latency_sensitive_walker2d_env(
        base_env_id="Walker2d-v4",
        render_mode="rgb_array",
        base_make_kwargs={"ctrl_cost_weight": 0.001},
    )
    env.reset(seed=4)
    action = [-2.0, -0.25, 0.0, 0.25, 0.5, 3.0]
    _, _, _, _, info = env.step(action)

    assert captured == {
        "env_id": "Walker2d-v4",
        "kwargs": {"render_mode": "rgb_array", "ctrl_cost_weight": 0.001},
    }
    assert env.action_space == native_env.action_space
    expected = np.asarray([-1.0, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    assert np.array_equal(native_env.actions[0], expected)
    assert info["applied_action"] == expected.tolist()
    assert env.render().shape == (480, 480, 3)


def test_zero_latency_applies_issued_walker2d_action_immediately(monkeypatch):
    native_env = _Walker2dProbeEnv()
    monkeypatch.setattr(gymnasium_walker2d.gym, "make", lambda *args, **kwargs: native_env)
    task_env = gymnasium_walker2d.make_latency_sensitive_walker2d_env(
        base_env_id="Walker2d-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )
    env = GymnasiumActionLatencyWrapper(
        task_env,
        fixed_latency_raw_frames=0,
        env_fps=125,
        obs_fps=125,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=scaffold.NOOP_ACTION,
    )
    env.reset(seed=0)
    env.step(np.ones(6, dtype=np.float32))

    assert np.array_equal(native_env.actions[0], np.ones(6, dtype=np.float32))


def test_scaffold_reports_native_continuous_action_contract(monkeypatch):
    native_env = _Walker2dProbeEnv()
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.7"}[package],
    )
    report = scaffold.build_walker2d_probe_report(
        base_env_id="Walker2d-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
    )
    env_config = scaffold.build_walker2d_env_config(
        base_env_id="Walker2d-v4",
        render_mode="rgb_array",
        base_make_kwargs={},
        env_fps=report["native_env_fps"],
    )

    assert report["action_space"]["shape"] == [6]
    assert report["observation_space"]["shape"] == [17]
    assert report["native_env_fps"] == 125.0
    assert env_config["action_space"] == {
        "type": "box",
        "labels": scaffold.ACTION_LABELS,
        "low": [-1.0] * 6,
        "high": [1.0] * 6,
        "dtype": "float32",
        "openvla_carrier_dim": 7,
        "openvla_padding": [0.0],
    }
    assert env_config["noop_action"] == scaffold.NOOP_ACTION
    assert report["action_catalog"]["openvla_carrier"][-1] == "OPENVLA_PAD"
    assert native_env.closed is True


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/gymnasium/walker2d/latency_bench_eval.yaml",
        "configs/examples/gymnasium/walker2d/small_model_train.yaml",
        "configs/examples/gymnasium/walker2d/small_model_train_ppo_tuned.yaml",
        "configs/examples/gymnasium/walker2d/small_model_train_sf_official_10m.yaml",
        "configs/examples/gymnasium/walker2d/small_model_eval_sf_official_10m.yaml",
        "configs/examples/gymnasium/walker2d/small_model_eval.yaml",
        "configs/examples/gymnasium/walker2d/starvla_eval.yaml",
    ],
)
def test_walker2d_configs_share_zero_latency_contract(config_path):
    config = load_config(config_path)
    env_cfg = config["env"]
    clock = EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"])

    assert env_cfg["env_id"] == gymnasium_walker2d.GYMNASIUM_WALKER2D_ENV_ID
    assert env_cfg["action_space"]["type"] == "box"
    assert env_cfg["action_space"]["labels"] == scaffold.ACTION_LABELS
    assert env_cfg["action_space"]["openvla_carrier_dim"] == 7
    assert env_cfg["noop_action"] == scaffold.NOOP_ACTION
    assert clock.obs_stride_raw_frames == 1
    assert config["latency"]["method"] == "zero"
    assert "fixed_latency_ms" not in config["latency"]


def test_walker2d_small_model_eval_sweeps_requested_latencies_and_episode_count():
    config = load_config("configs/examples/gymnasium/walker2d/small_model_eval.yaml")

    assert config["evaluation"]["eval_latency_values"] == [0, 1, 2, 3, 4]
    assert config["evaluation"]["eval_episodes"] == 10
    assert config["backend"]["train_dir"] == "/mnt/checkpoints/latency-sensitive-bench/walker2d"
    assert config["logging"]["output_dir"] == "/mnt/data/latency-sensitive-bench/walker2d/continuous_small_model_eval"


def test_walker2d_checkpoint_contract_preserves_box_and_openvla_carrier():
    config = load_config("configs/examples/gymnasium/walker2d/small_model_train_ppo_tuned.yaml")
    cfg = build_train_cfg_from_config(config)
    task_contract = gymnasium_task_contract(config["env"])

    assert cfg.gym_action_space_json == json.dumps(config["env"]["action_space"])
    assert json.loads(cfg.gym_noop_action_json) == [0.0] * 6
    assert task_contract["action_space"]["type"] == "box"
    assert task_contract["action_space"]["openvla_carrier_dim"] == 7
    assert task_contract["action_space"]["openvla_padding"] == [0.0]
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.kl_loss_coeff == pytest.approx(0.1)
    assert cfg.adaptive_stddev is False
    assert cfg.policy_initialization == "torch_default"
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.num_epochs == 2
    assert cfg.num_batches_per_epoch == 4
    assert cfg.ppo_clip_ratio == pytest.approx(0.2)
    assert cfg.value_loss_coeff == pytest.approx(1.3)
    assert cfg.max_grad_norm == pytest.approx(3.5)
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.lr_schedule == "linear_decay"
    assert cfg.async_rl is False
    assert cfg.value_bootstrap is True


def test_walker2d_formal_and_candidate_training_budgets_are_isolated():
    formal = load_config("configs/examples/gymnasium/walker2d/small_model_train.yaml")
    candidate = load_config("configs/examples/gymnasium/walker2d/small_model_train_ppo_tuned.yaml")
    official = load_config("configs/examples/gymnasium/walker2d/small_model_train_sf_official_10m.yaml")

    assert formal["training"]["train_for_env_steps"] == 10_000_000
    assert candidate["training"]["train_for_env_steps"] == 1_000_000
    assert candidate["experiment"]["name"].endswith("_candidate")
    assert official["training"]["train_for_env_steps"] == 10_000_000
    assert official["experiment"]["name"].endswith("_sf_official_10m")


def test_walker2d_rollout_and_starvla_keep_six_continuous_dimensions():
    action_spec = gymnasium_continuous_action_spec(scaffold.ACTION_LABELS)
    action = [-0.9, -0.5, -0.1, 0.2, 0.6, 0.95]

    encoded = encode_action_for_env(
        "walker2d_continuous_torque",
        action,
        action_spec=action_spec,
    )
    decoded, metadata = action_from_starvla_payload(
        payload=np.asarray(action, dtype=np.float32),
        env_name="walker2d_continuous_torque",
        action_by_raw_id={},
        action_output_type="rl_games_continuous",
    )

    assert action_spec.layout == ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1
    assert encoded["action"] == action
    assert decoded.value == pytest.approx(action)
    assert metadata["continuous_action"] == pytest.approx(action)
