from __future__ import annotations

from latency_bench.envs.factory import resolve_env_config
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import latency_bench.envs.tasks.ant as gymnasium_ant
import latency_bench.envs.tasks.humanoid as gymnasium_humanoid
import scripts.scaffold_gymnasium_ant as ant_scaffold
import scripts.scaffold_gymnasium_humanoid as humanoid_scaffold
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.tasks.contracts import gymnasium_task_contract
from training.gymnasium_sf.integration import build_train_cfg_from_config


class _ProbeEnv(gym.Env):
    def __init__(
        self,
        *,
        env_id: str,
        action_low: float,
        action_high: float,
        action_dim: int,
        observation_dim: int,
        dt: float,
    ) -> None:
        self.action_space = gym.spaces.Box(
            action_low,
            action_high,
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(observation_dim,),
            dtype=np.float64,
        )
        self.dt = dt
        self.frame_skip = 5
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=dt / self.frame_skip))
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": round(1.0 / dt)}
        self.spec = SimpleNamespace(
            id=env_id,
            entry_point=f"fake:{env_id}",
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
        return np.arange(self.observation_space.shape[0], dtype=np.float64), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        return (
            np.ones(self.observation_space.shape[0], dtype=np.float64),
            1.0,
            False,
            False,
            {"x_position": 0.0},
        )

    def render(self):
        return np.ones((480, 480, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("module", "wrapper_cls", "env_id", "low", "high", "action_dim", "observation_dim"),
    [
        (gymnasium_ant, gymnasium_ant.AntContinuousActionEnv, "Ant-v4", -1.0, 1.0, 8, 27),
        (
            gymnasium_humanoid,
            gymnasium_humanoid.HumanoidContinuousActionEnv,
            "Humanoid-v4",
            -0.4,
            0.4,
            17,
            376,
        ),
    ],
)
def test_locomotion_wrappers_clip_and_record_native_torques(
    monkeypatch,
    module,
    wrapper_cls,
    env_id,
    low,
    high,
    action_dim,
    observation_dim,
):
    native_env = _ProbeEnv(
        env_id=env_id,
        action_low=low,
        action_high=high,
        action_dim=action_dim,
        observation_dim=observation_dim,
        dt=0.05 if env_id == "Ant-v4" else 0.015,
    )
    monkeypatch.setattr(module.gym, "make", lambda *args, **kwargs: native_env)
    env = wrapper_cls(base_env_id=env_id, render_mode="rgb_array", base_make_kwargs={})
    _, _, _, _, info = env.step(np.linspace(low * 2, high * 2, action_dim))

    expected = np.clip(
        np.linspace(low * 2, high * 2, action_dim).astype(np.float32),
        low,
        high,
    )
    assert np.array_equal(native_env.actions[0], expected)
    assert info["applied_action"] == expected.tolist()


def test_locomotion_registered_entry_points_are_task_factories():
    assert (
        gym.spec(gymnasium_ant.GYMNASIUM_ANT_ENV_ID).entry_point
        is gymnasium_ant.make_latency_sensitive_ant_env
    )
    assert (
        gym.spec(gymnasium_humanoid.GYMNASIUM_HUMANOID_ENV_ID).entry_point
        is gymnasium_humanoid.make_latency_sensitive_humanoid_env
    )


@pytest.mark.parametrize(
    ("scaffold", "builder", "env_id", "action_dim", "observation_dim", "dt"),
    [
        (ant_scaffold, ant_scaffold.build_ant_probe_report, "Ant-v4", 8, 27, 0.05),
        (
            humanoid_scaffold,
            humanoid_scaffold.build_humanoid_probe_report,
            "Humanoid-v4",
            17,
            376,
            0.015,
        ),
    ],
)
def test_locomotion_probes_report_native_contract(
    monkeypatch,
    scaffold,
    builder,
    env_id,
    action_dim,
    observation_dim,
    dt,
):
    native_env = _ProbeEnv(
        env_id=env_id,
        action_low=-1.0 if env_id == "Ant-v4" else -0.4,
        action_high=1.0 if env_id == "Ant-v4" else 0.4,
        action_dim=action_dim,
        observation_dim=observation_dim,
        dt=dt,
    )
    monkeypatch.setattr(scaffold.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setattr(
        scaffold.importlib.metadata,
        "version",
        lambda package: {"gymnasium": "0.29.1", "mujoco": "3.3.0"}[package],
    )

    report = builder(env_id=env_id, render_mode="rgb_array")

    assert report["action_space"]["shape"] == [action_dim]
    assert report["observation_space"]["shape"] == [observation_dim]
    assert report["native_env_fps"] == pytest.approx(1.0 / dt)
    assert report["action_catalog"]["labels"] == scaffold.ACTION_LABELS
    assert report["state_labels"] == scaffold.STATE_LABELS
    assert native_env.closed is True


@pytest.mark.parametrize(
    ("task", "scaffold", "env_id", "base_env_id", "fps", "action_dim", "state_dim"),
    [
        (
            "ant",
            ant_scaffold,
            gymnasium_ant.GYMNASIUM_ANT_ENV_ID,
            "Ant-v4",
            20.0,
            8,
            27,
        ),
        (
            "humanoid",
            humanoid_scaffold,
            gymnasium_humanoid.GYMNASIUM_HUMANOID_ENV_ID,
            "Humanoid-v4",
            200.0 / 3.0,
            17,
            376,
        ),
    ],
)
def test_locomotion_configs_preserve_native_state_contract(
    task,
    scaffold,
    env_id,
    base_env_id,
    fps,
    action_dim,
    state_dim,
):
    config_paths = [
        f"configs/examples/gymnasium/{task}/small_model_train.yaml",
        f"configs/examples/gymnasium/{task}/small_model_eval.yaml",
        f"configs/examples/gymnasium/{task}/latency_bench_eval.yaml",
        f"configs/examples/gymnasium/{task}/starvla_eval.yaml",
    ]
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        resolve_env_config(config)
    envs = [config["env"] for config in configs]

    assert envs[1:] == envs[:-1]
    env = envs[0]
    assert env["name"] == task
    assert env["env_id"] == env_id
    assert env["make_kwargs"]["base_env_id"] == base_env_id
    assert env["env_fps"] == pytest.approx(fps)
    assert EnvClock(env_fps=env["env_fps"], obs_fps=env["obs_fps"]).obs_stride_raw_frames == 1
    assert env["action_space"]["labels"] == scaffold.ACTION_LABELS
    assert len(env["action_space"]["labels"]) == action_dim
    assert env["noop_action"] == scaffold.NOOP_ACTION
    assert env["state_space"]["labels"] == scaffold.STATE_LABELS
    assert len(env["state_space"]["labels"]) == state_dim
    assert configs[0]["latency"]["method"] == "zero"


@pytest.mark.parametrize("task", ["ant", "humanoid"])
def test_locomotion_training_uses_vendored_sample_factory_mujoco_recipe(task):
    config = load_config(f"configs/examples/gymnasium/{task}/small_model_train.yaml")
    resolve_env_config(config)
    cfg = build_train_cfg_from_config(config)
    contract = gymnasium_task_contract(config["env"])

    assert config["training"]["train_for_env_steps"] == 10_000_000
    assert cfg.encoder_mlp_layers == [64, 64]
    assert cfg.nonlinearity == "tanh"
    assert cfg.learning_rate == pytest.approx(0.00295)
    assert cfg.rollout == 64
    assert cfg.recurrence == 1
    assert cfg.async_rl is False
    assert contract["action_space"]["labels"] == config["env"]["action_space"]["labels"]
    assert contract["state_space"]["labels"] == config["env"]["state_space"]["labels"]
