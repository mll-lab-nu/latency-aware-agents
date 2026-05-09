from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from filelock import FileLock
from gymnasium.spaces import Box, Discrete


def _guard_cfg(**overrides):
    cfg = {
        "vizdoom_init_max_parallel": 2,
        "vizdoom_init_lock_timeout_sec": 0.01,
        "vizdoom_init_lock_max_wait_sec": 0.2,
        "vizdoom_init_retry_jitter_sec": 0.0,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _env_config():
    return SimpleNamespace(worker_index=3, vector_index=1, env_id=13)


def _new_vizdoom_env(fake_game):
    from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

    env = VizdoomEnv.__new__(VizdoomEnv)
    env.game = fake_game
    return env


class _FakeGame:
    def __init__(self):
        self.init_calls = 0

    def init(self):
        self.init_calls += 1


def test_guard_retries_with_new_lock_after_first_lock_timeout(tmp_path, monkeypatch):
    from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed
    from sf_examples.vizdoom.doom import doom_gym

    locked_path = tmp_path / "doom_000.lockfile"
    free_path = tmp_path / "doom_001.lockfile"
    held_lock = FileLock(str(locked_path))
    held_lock.acquire(timeout=0)
    paths = iter([str(locked_path), str(free_path)])
    monkeypatch.setattr(doom_gym, "doom_lock_file", lambda max_parallel: next(paths))

    fake_game = _FakeGame()
    env = _new_vizdoom_env(fake_game)
    try:
        installed = ensure_vizdoom_init_guard_installed(
            env,
            env_name="doom_deadly_corridor",
            cfg=_guard_cfg(),
            env_config=_env_config(),
        )
        env._game_init()
    finally:
        held_lock.release()

    assert installed is True
    assert fake_game.init_calls == 1


def test_guard_fails_fast_when_all_locks_timeout(tmp_path, monkeypatch):
    from sample_factory.envs.env_utils import EnvCriticalError
    from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed
    from sf_examples.vizdoom.doom import doom_gym

    locked_path = tmp_path / "doom_000.lockfile"
    held_lock = FileLock(str(locked_path))
    held_lock.acquire(timeout=0)
    monkeypatch.setattr(doom_gym, "doom_lock_file", lambda max_parallel: str(locked_path))

    env = _new_vizdoom_env(_FakeGame())
    try:
        ensure_vizdoom_init_guard_installed(
            env,
            env_name="doom_deadly_corridor",
            cfg=_guard_cfg(vizdoom_init_lock_max_wait_sec=0.03),
            env_config=_env_config(),
        )
        with pytest.raises(EnvCriticalError) as exc_info:
            env._game_init()
    finally:
        held_lock.release()

    message = str(exc_info.value)
    assert "ViZDoom initialization lock wait exceeded" in message
    assert "doom_deadly_corridor" in message
    assert "worker_index=3" in message
    assert str(locked_path) in message


def test_guard_install_is_idempotent_and_updates_config():
    from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed

    env = _new_vizdoom_env(_FakeGame())

    assert ensure_vizdoom_init_guard_installed(
        env,
        env_name="doom_deadly_corridor",
        cfg=_guard_cfg(vizdoom_init_lock_timeout_sec=0.01),
        env_config=_env_config(),
    )
    first_method = env.__dict__["_game_init"]

    assert ensure_vizdoom_init_guard_installed(
        env,
        env_name="doom_deadly_corridor",
        cfg=_guard_cfg(vizdoom_init_lock_timeout_sec=0.02),
        env_config=_env_config(),
    )

    assert env.__dict__["_game_init"] is first_method
    assert env._latency_bench_vizdoom_init_guard_config.lock_timeout_sec == 0.02


def test_guard_fails_fast_when_upstream_game_init_signature_changes(monkeypatch):
    from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed
    from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

    def incompatible_game_init(self, timeout):
        return None

    monkeypatch.setattr(VizdoomEnv, "_game_init", incompatible_game_init)

    with pytest.raises(RuntimeError, match="Unsupported VizdoomEnv._game_init signature"):
        ensure_vizdoom_init_guard_installed(
            _new_vizdoom_env(_FakeGame()),
            env_name="doom_deadly_corridor",
            cfg=_guard_cfg(),
            env_config=_env_config(),
        )


class _DummyDoomGymEnv(gym.Env):
    observation_space = Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
    action_space = Discrete(2)

    def reset(self, **kwargs):
        return np.zeros((3, 2, 2), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((3, 2, 2), dtype=np.uint8), 0.0, False, False, {}


def test_deadly_corridor_env_factory_installs_guard(monkeypatch):
    from training.deadly_corridor_sf import integration

    captured = {}
    base_env = _DummyDoomGymEnv()

    def fake_make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
        return base_env

    def fake_install_guard(env, *, env_name, cfg, env_config):
        captured["env"] = env
        captured["env_name"] = env_name
        captured["cfg"] = cfg
        captured["env_config"] = env_config
        return True

    monkeypatch.setattr(
        "sf_examples.vizdoom.doom.doom_utils.make_doom_env_from_spec",
        fake_make_doom_env_from_spec,
    )
    monkeypatch.setattr(integration, "ensure_vizdoom_init_guard_installed", fake_install_guard)

    cfg = SimpleNamespace(
        disable_reward_scaling=False,
        env_fps=35,
        obs_fps=8.75,
        frame_stack=1,
        max_episode_steps=0,
        latency_raw_frame_values="0",
        latency_distribution_spec="",
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        episodes_per_stage=10,
        randomize_within_stage=False,
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        seed=0,
        add_latency_info=False,
        hold_policy="hold",
        ordering_policy="latest_ready",
        max_pending_actions=None,
    )
    env_config = _env_config()

    wrapped = integration.make_latency_doom_env_from_spec(
        SimpleNamespace(reward_scaling=1.0),
        "doom_deadly_corridor",
        cfg,
        env_config,
    )

    assert captured == {
        "env": base_env,
        "env_name": "doom_deadly_corridor",
        "cfg": cfg,
        "env_config": env_config,
    }
    wrapped.close()


def test_guard_does_not_affect_non_deadly_corridor_envs():
    from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed

    fake_env = SimpleNamespace()

    installed = ensure_vizdoom_init_guard_installed(
        fake_env,
        env_name="flappy_bird",
        cfg=_guard_cfg(),
        env_config=_env_config(),
    )

    assert installed is False
    assert "_game_init" not in fake_env.__dict__


def test_deadly_corridor_cli_args_emit_optional_vizdoom_guard_settings_only_when_configured():
    from training.deadly_corridor_sf import integration
    from training.sample_factory_common.config_adapter import load_training_config

    config = load_training_config("configs/train/deadly_corridor/latency_0/seed_0/train.yaml")

    default_args = integration.build_cli_args_from_config(config)

    assert "--vizdoom_init_max_parallel" not in default_args
    assert "--vizdoom_init_lock_timeout_sec" not in default_args
    assert "--vizdoom_init_lock_max_wait_sec" not in default_args
    assert "--vizdoom_init_retry_jitter_sec" not in default_args

    config["env"].update(
        {
            "vizdoom_init_max_parallel": 12,
            "vizdoom_init_lock_timeout_sec": 0.5,
            "vizdoom_init_lock_max_wait_sec": 30,
            "vizdoom_init_retry_jitter_sec": 0.1,
        }
    )

    args = integration.build_cli_args_from_config(config)

    assert args[args.index("--vizdoom_init_max_parallel") + 1] == "12"
    assert args[args.index("--vizdoom_init_lock_timeout_sec") + 1] == "0.5"
    assert args[args.index("--vizdoom_init_lock_max_wait_sec") + 1] == "30"
    assert args[args.index("--vizdoom_init_retry_jitter_sec") + 1] == "0.1"
