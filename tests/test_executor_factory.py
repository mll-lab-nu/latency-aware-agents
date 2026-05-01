from __future__ import annotations

import pytest

from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.executors.factory import build_executor


@register_env_adapter("normalized_noop_test")
def _build_normalized_noop_env(config: dict, noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    experiment_cfg = config["experiment"]
    return DummyEnvAdapter(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(experiment_cfg.get("max_env_steps", 5)),
        noop_action=Action(value="env-noop", name="env-noop", is_noop=True),
    )


def _config(mode: str, output_dir: str) -> dict:
    return {
        "experiment": {
            "name": "executor_factory_test",
            "seed": 0,
            "num_episodes": 1,
            "max_env_steps": 5,
        },
        "executor": {
            "mode": mode,
        },
        "env": {
            "name": "normalized_noop_test",
            "fps_env": 30,
            "noop_action": "config-noop",
            "action_map": {
                "config-noop": "config-noop",
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
            "action": "jump",
        },
        "logging": {
            "output_dir": output_dir,
        },
    }


@pytest.mark.parametrize("mode", ["simulated", "paused", "realtime"])
def test_executor_scheduler_uses_env_normalized_noop(mode: str, tmp_path):
    executor = build_executor(_config(mode=mode, output_dir=str(tmp_path)))

    try:
        assert executor.env.noop_action.value == "env-noop"
        assert executor.action_scheduler.noop_action is executor.env.noop_action
    finally:
        executor.env.close()
