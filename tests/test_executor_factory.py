from __future__ import annotations

import pytest

from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.executors.factory import build_executor
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.experiments.run_simulated_latency import build_executor as build_legacy_executor


@register_env_adapter("normalized_noop_test")
def _build_normalized_noop_env(config: dict, noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    return DummyEnvAdapter(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=Action(value="env-noop", name="env-noop", is_noop=True),
    )


def _config(mode: str = "simulated", output_dir: str = "outputs/executor_factory_test") -> dict:
    return {
        "experiment": {
            "name": "executor_factory_test",
            "seed": 0,
            "num_episodes": 1,
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
        "evaluation": {
            "eval_max_steps": 5,
        },
        "logging": {
            "output_dir": output_dir,
        },
    }


def test_simulated_mode_builds_simulated_executor(tmp_path):
    executor = build_executor(_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


def test_paused_mode_builds_paused_executor(tmp_path):
    executor = build_executor(_config(mode="paused", output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, PausedLatencyExecutor)
    finally:
        executor.env.close()


def test_realtime_mode_builds_realtime_executor(tmp_path):
    executor = build_executor(_config(mode="realtime", output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, RealtimeLatencyExecutor)
    finally:
        executor.env.close()


def test_factory_accepts_measured_latency_for_realtime(tmp_path):
    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["latency"] = {"type": "measured"}

    executor = build_executor(config)

    try:
        assert isinstance(executor, RealtimeLatencyExecutor)
    finally:
        executor.env.close()


@pytest.mark.parametrize("mode", ["simulated", "paused"])
def test_factory_rejects_measured_latency_for_paused_and_simulated(tmp_path, mode):
    config = _config(mode=mode, output_dir=str(tmp_path))
    config["latency"] = {"type": "measured"}

    with pytest.raises(ValueError, match="measured"):
        build_executor(config)


def test_legacy_simulated_wrapper_uses_new_entrypoint(tmp_path):
    executor = build_legacy_executor(_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


@pytest.mark.parametrize("mode", ["simulated", "paused", "realtime"])
def test_executor_scheduler_uses_env_normalized_noop(mode: str, tmp_path):
    executor = build_executor(_config(mode=mode, output_dir=str(tmp_path)))

    try:
        assert executor.env.noop_action.value == "env-noop"
        assert executor.action_scheduler.noop_action is executor.env.noop_action
    finally:
        executor.env.close()
