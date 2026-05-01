from __future__ import annotations

import pytest

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.envs.flappy import FlappyBirdEnvAdapter
from latency_bench.envs.factory import build_env_adapter, get_registered_env_names
from latency_bench.experiments.run_simulated_latency import build_executor


def _config(env_name: str = "dummy", output_dir: str = "outputs/factory_test") -> dict:
    return {
        "experiment": {
            "name": "factory_test",
            "seed": 0,
            "num_episodes": 1,
            "max_env_steps": 5,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": env_name,
            "fps_env": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "control": {
            "frequency_hz": 30,
            "observe_every_n_frames": 1,
        },
        "latency": {
            "type": "zero",
        },
        "scheduler": {
            "persistence_mode": "noop_after_oneshot",
            "stale_action_mode": "latest_ready",
        },
        "policy": {
            "type": "constant",
            "action": "noop",
        },
        "logging": {
            "output_dir": output_dir,
        },
    }


def _flappy_config(output_dir: str = "outputs/factory_test") -> dict:
    config = _config("flappy", output_dir=output_dir)
    config["env"].update(
        {
            "gym_id": "FlappyBird-v0",
            "render_mode": "rgb_array",
            "observation_mode": "image",
            "use_lidar": False,
            "normalize_obs": True,
            "audio_on": False,
            "action_map": {
                "noop": 0,
                "flap": 1,
            },
            "oneshot_actions": ["flap"],
        }
    )
    config["policy"] = {
        "type": "constant",
        "action": "flap",
        "actions": ["noop", "flap"],
    }
    return config


def test_dummy_env_adapter_is_registered():
    assert "dummy" in get_registered_env_names()
    assert "flappy" in get_registered_env_names()


def test_build_env_adapter_returns_dummy_adapter_for_dummy_config():
    noop = Action(value="noop", name="noop", is_noop=True)

    env = build_env_adapter(_config("dummy"), noop)

    assert isinstance(env, DummyEnvAdapter)
    assert env.fps_env == 30.0
    assert env.max_steps == 5
    assert env.noop_action is noop


def test_build_env_adapter_rejects_unsupported_env_name():
    noop = Action(value="noop", name="noop", is_noop=True)

    with pytest.raises(ValueError, match="Unsupported env.name: unknown"):
        build_env_adapter(_config("unknown"), noop)


def test_demon_attack_base_config_declares_action_repeat():
    config = load_config("configs/base/demon_attack.yaml")

    assert config["env"]["action_repeat"] == 1


def test_build_env_adapter_passes_demon_attack_action_repeat(monkeypatch):
    from latency_bench.envs import demon_attack

    captured = {}

    class FakeDemonAttackEnvAdapter:
        def __init__(self, *, action_repeat: int, **kwargs):
            captured["action_repeat"] = action_repeat
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        demon_attack,
        "DemonAttackEnvAdapter",
        FakeDemonAttackEnvAdapter,
    )
    config = _config("demon_attack")
    config["env"].update(
        {
            "env_id": "ALE/DemonAttack-v5",
            "action_repeat": 4,
            "screen_size": 84,
            "noop_max": 30,
            "terminal_on_life_loss": False,
            "clip_reward": True,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
        }
    )
    noop = Action(value=0, name="noop", is_noop=True)

    env = build_env_adapter(config, noop)

    assert isinstance(env, FakeDemonAttackEnvAdapter)
    assert captured["action_repeat"] == 4


def test_build_executor_uses_registered_env_factory(tmp_path):
    executor = build_executor(_config("dummy", output_dir=str(tmp_path)))

    assert isinstance(executor.env, DummyEnvAdapter)


def test_build_env_adapter_returns_flappy_adapter_for_flappy_config():
    noop = Action(value=0, name="noop", is_noop=True)
    env = build_env_adapter(_flappy_config(), noop)

    try:
        assert isinstance(env, FlappyBirdEnvAdapter)
        assert env.fps_env == 30.0
        assert env.noop_action is noop
    finally:
        env.close()


def test_build_executor_maps_flappy_policy_actions(tmp_path):
    executor = build_executor(_flappy_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor.env, FlappyBirdEnvAdapter)
        assert executor.policy.action.value == 1
        assert executor.policy.action.name == "flap"
        assert executor.policy.action.is_oneshot
        assert executor.action_scheduler.noop_action.value == 0
    finally:
        executor.env.close()
