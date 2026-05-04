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


@register_env_adapter("tuple_action_parallel_test")
def _build_tuple_action_parallel_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    import gymnasium as gym

    class TupleActionEnv(DummyEnvAdapter):
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2)))

    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    env = TupleActionEnv(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=noop_action,
    )
    env.gym_env = env
    return env


def test_simulated_mode_builds_simulated_executor(tmp_path):
    executor = build_executor(_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


def test_simulated_eval_parallel_envs_1_keeps_old_executor_path(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 1

    executor = build_executor(config)

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


def test_simulated_parallel_builds_batched_executor_for_supported_zero_latency(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    config = _config(output_dir=str(tmp_path))
    config["experiment"]["num_episodes"] = 3
    config["evaluation"]["eval_parallel_envs"] = 4

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert executor.num_slots == 3
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ["paused", "realtime"])
def test_non_simulated_modes_reject_parallel_eval(tmp_path, mode):
    config = _config(mode=mode, output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2

    with pytest.raises(ValueError, match="eval_parallel_envs.*simulated"):
        build_executor(config)


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


def test_factory_rejects_measured_latency_for_paused(tmp_path):
    mode = "paused"
    config = _config(mode=mode, output_dir=str(tmp_path))
    config["latency"] = {"type": "measured"}

    with pytest.raises(ValueError, match="measured"):
        build_executor(config)


def test_simulated_parallel_trace_replay_profile_is_not_rejected(tmp_path):
    trace_path = tmp_path / "profile_device_latency.jsonl"
    trace_path.write_text('{"latency_ms": 0.0}\n', encoding="utf-8")
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 4
    config["latency"] = {
        "type": "trace_replay",
        "trace_path": str(trace_path),
        "loop_trace": True,
    }

    executor = build_executor(config)

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


def test_parallel_simulated_tuple_action_space_fails_before_phase_3d(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["experiment"]["num_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["env"]["name"] = "tuple_action_parallel_test"
    config["env"]["noop_action"] = (0, 0)
    config["env"]["action_map"] = {"noop": (0, 0), "jump": (1, 0)}
    config["policy"]["action"] = "jump"

    with pytest.raises(ValueError, match="Tuple.*MultiDiscrete"):
        build_executor(config)


def test_parallel_simulated_stateful_policy_without_predict_batch_fails_fast(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["experiment"]["num_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["policy"] = {
        "type": "random",
        "actions": ["jump"],
    }

    with pytest.raises(ValueError, match="does not support safe batched simulated eval"):
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
