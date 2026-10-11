from __future__ import annotations

from latency_bench.envs.factory import resolve_env_config
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.tasks.swimmer as gymnasium_swimmer
import scripts.scaffold_gymnasium_swimmer as scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.envs.tasks.contracts import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


@pytest.mark.parametrize("registration_module", [
    "latency_bench.envs.tasks.swimmer",
    "latency_bench.envs.gymnasium_swimmer",
])
def test_swimmer_native_reset_and_step_with_saved_registration_path(registration_module):
    env = GymnasiumEnvAdapter(
        env_id=gymnasium_swimmer.GYMNASIUM_SWIMMER_ENV_ID,
        registration_imports=[registration_module],
        make_kwargs={"render_mode": None, "base_make_kwargs": {}},
        env_fps=25.0,
        noop_action=Action(value=np.zeros(2)),
        frame_stack=1,
        export_env_raw_rgb_frames=False,
        include_state=True,
    )
    try:
        initial = env.reset(seed=0)
        result = env.step(Action(value=np.zeros(2)))
        assert initial.data.shape == (8,)
        assert result.observation.metadata["gymnasium_state"].shape == (8,)
        assert result.observation.env_step == 1
        assert result.observation.sim_time_ms == pytest.approx(40.0)
        assert np.isfinite(result.reward)
        assert env.gym_env.unwrapped.dt == pytest.approx(0.04)
    finally:
        env.close()


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


def test_swimmer_configs_preserve_native_state_contract():
    config_paths = [
        "configs/examples/gymnasium/swimmer/small_model_train.yaml",
        "configs/examples/gymnasium/swimmer/small_model_eval.yaml",
        "configs/examples/gymnasium/swimmer/latency_bench_eval.yaml",
        "configs/examples/gymnasium/swimmer/starvla_eval.yaml",
    ]
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        resolve_env_config(config)
    envs = [config["env"] for config in configs]

    assert envs[1:] == envs[:-1]
    env = envs[0]
    assert env["name"] == "swimmer"
    assert env["env_id"] == gymnasium_swimmer.GYMNASIUM_SWIMMER_ENV_ID
    assert env["registration_imports"] == ["latency_bench.envs.tasks.swimmer"]
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
    resolve_env_config(config)
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["training"]["train_for_env_steps"] == 10_000_000
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.async_rl is True
    assert cfg.value_bootstrap is False
    assert contract["action_space"]["labels"] == scaffold.ACTION_LABELS
    assert contract["state_space"]["labels"] == scaffold.STATE_LABELS


def test_swimmer_eval_contracts_cover_expected_latencies():
    small_model = load_config("configs/examples/gymnasium/swimmer/small_model_eval.yaml")
    resolve_env_config(small_model)
    starvla = load_config("configs/examples/gymnasium/swimmer/starvla_eval.yaml")
    resolve_env_config(starvla)

    assert [round(suite["latency"].get("fixed_latency_ms", 0) * small_model["env"]["env_fps"] / 1000) for suite in small_model["evaluation"]["suites"]] == [0, 1, 2, 3, 4]
    assert small_model["evaluation"]["episodes"] == 10
    assert [round(suite["latency"].get("fixed_latency_ms", 0) * starvla["env"]["env_fps"] / 1000) for suite in starvla["evaluation"]["suites"]] == [0, 2]
    assert starvla["evaluation"]["episodes"] == 20
