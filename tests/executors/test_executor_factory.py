from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import Action, OrderingPolicy
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.executors.factory import build_executor
import latency_bench.executors.factory as executor_factory
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime.executor import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.eval.driver import run_from_config
from latency_bench.latency.burst_profile import (
    build_hidden_regime_burst_model,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.latency.profile import save_latency_distribution, save_profile_manifest
from latency_bench.core.clock import EnvClock
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy


@register_env_adapter("normalized_noop_test")
def _build_normalized_noop_env(config: dict, noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    return DummyEnvAdapter(
        env_fps=float(env_cfg["env_fps"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=Action(value="env-noop", name="env-noop", is_noop=True),
    )


def _config(mode: str = "simulated", output_dir: str = "outputs/executor_factory_test") -> dict:
    return {
        "experiment": {
            "name": "executor_factory_test",
            "seed": 0,
        },
        "executor": {
            "mode": mode,
        },
        "env": {
            "name": "normalized_noop_test",
            "env_fps": 30,
            "obs_fps": 30,
            "noop_action": "config-noop",
            "action_map": {
                "config-noop": "config-noop",
                "jump": "jump",
            },
        },
        "latency": {
            "method": "measured" if mode == "realtime" else "zero",
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "policy": {
            "type": "constant",
            "action": "jump",
        },
        "evaluation": {
            "eval_episodes": 1,
            "eval_max_steps": 5,
        },
        "logging": {
            "output_dir": output_dir,
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _write_latency_profile(tmp_path) -> str:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    burst_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "worker_id": index % 2,
            "observation_to_action_latency_ms": float(frames) * 40.0,
            "worker_service_time_ms": float(frames) * 40.0,
            "worker_service_time_ms_bin": frames * 40,
        }
        for index, frames in enumerate([1, 1, 1, 1, 1, 1])
    ]
    burst_model = build_hidden_regime_burst_model(burst_samples)
    save_latency_distribution(
        build_worker_slot_distributions_from_sessions(
            [burst_samples],
            spike_threshold_ms_by_worker_slot=burst_model[
                "spike_threshold_ms_by_worker_slot"
            ],
        ),
        profile_dir / "latency_distribution.json",
    )
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(burst_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_profile_manifest(
        {
            "model_id": "openvla",
            "gpu_class": "2x-test-gpu",
            "workload_id": "flappy",
            "instance_id": "instance_test",
            "source_run_id": "run_test",
            "env_fps": 25.0,
            "frame_ms": 40.0,
            "n_samples": 6,
            "distribution_path": "latency_distribution.json",
            "latency_method": "temporal",
            "sample_model_type": "hidden_regime",
            "burst_model_path": "latency_burst_model.json",
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return str(profile_dir)


@register_env_adapter("tuple_action_parallel_test")
def _build_tuple_action_parallel_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    import gymnasium as gym

    class TupleActionEnv(DummyEnvAdapter):
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2)))

    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    env = TupleActionEnv(
        env_fps=float(env_cfg["env_fps"]),
        max_steps=int(evaluation_cfg["eval_max_steps"]),
        noop_action=noop_action,
    )
    env.gym_env = env
    return env


@register_env_adapter("nested_tuple_action_parallel_test")
def _build_nested_tuple_action_parallel_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    import gymnasium as gym

    class NestedTupleActionEnv(DummyEnvAdapter):
        action_space = gym.spaces.Tuple((
            gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2))),
            gym.spaces.Discrete(2),
        ))

    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    env = NestedTupleActionEnv(
        env_fps=float(env_cfg["env_fps"]),
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


def test_simulated_inference_pool_requires_cuda_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_factory, "visible_cuda_device_count", lambda: 0)
    config = _config(output_dir=str(tmp_path))
    config["executor"]["simulated_inference_pool"] = True

    with pytest.raises(RuntimeError, match="requires at least one visible CUDA device"):
        build_executor(config)


@pytest.mark.parametrize('configured_devices', [None, ['cuda:1']])
def test_simulated_inference_pool_builds_pool_and_no_parent_policy(tmp_path, monkeypatch, configured_devices):
    created = {}

    class FakePool:
        def __init__(self, *, config, inference_devices):
            created["devices"] = list(inference_devices)

        def close(self):
            created["closed"] = True

    def _fail_build_policy(*args, **kwargs):
        raise AssertionError("the parent process must not load the policy in pool mode")

    monkeypatch.setattr(executor_factory, "ProcessInferencePool", FakePool)
    monkeypatch.setattr(executor_factory, "visible_cuda_device_count", lambda: 2)
    monkeypatch.setattr(executor_factory, "build_policy", _fail_build_policy)
    config = _config(output_dir=str(tmp_path))
    config["executor"]["simulated_inference_pool"] = True
    config["policy"]["device"] = "cuda"
    if configured_devices is not None:
        config['executor']['inference_devices'] = configured_devices

    executor = build_executor(config)
    try:
        assert isinstance(executor.inference_pool, FakePool)
        assert created["devices"] == (configured_devices if configured_devices is not None else ["cuda:0", "cuda:1"])
        assert executor.policy is None
    finally:
        executor.close()
    assert created.get("closed") is True


def test_simulated_inference_pool_rejects_injected_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_factory, "visible_cuda_device_count", lambda: 2)
    config = _config(output_dir=str(tmp_path))
    config["executor"]["simulated_inference_pool"] = True
    config["policy"]["device"] = "cuda"

    with pytest.raises(ValueError, match="cannot cross the process boundary"):
        build_executor(config, policy=object())


def test_build_executor_uses_injected_policy_and_skips_build_policy(tmp_path, monkeypatch):
    # The in-process eval path injects a live policy; build_policy must not run.
    def _fail_build_policy(*args, **kwargs):
        raise AssertionError("build_policy should not be called when a policy is injected")

    monkeypatch.setattr(executor_factory, "build_policy", _fail_build_policy)
    sentinel = object()
    executor = build_executor(_config(mode="paused", output_dir=str(tmp_path)), policy=sentinel)

    try:
        assert isinstance(executor, PausedLatencyExecutor)
        assert executor.policy is sentinel
    finally:
        executor.env.close()


def test_build_executor_realtime_rejects_injected_policy(tmp_path):
    with pytest.raises(ValueError, match="cannot cross the process boundary"):
        build_executor(_config(mode="realtime", output_dir=str(tmp_path)), policy=object())


def test_realtime_uses_injected_task_backend_locally(tmp_path):
    import gymnasium as gym

    from latency_bench.executors.env_step_backend import (
        EnvStepBackendEnvAdapter,
        SerialEnvStepBackend,
    )

    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["executor"]["inference_devices"] = ["cpu"]
    env = _build_normalized_noop_env(config, Action(value="env-noop"))
    env.action_space = gym.spaces.Discrete(2)
    backend = SerialEnvStepBackend([env])

    executor = build_executor(config, env_backend=backend)

    try:
        assert isinstance(executor.env, EnvStepBackendEnvAdapter)
        assert executor.env.backend is backend
        assert executor.use_env_process is False
    finally:
        executor.close()


def test_factory_requires_ordering_policy(tmp_path):
    config = _config(output_dir=str(tmp_path))
    del config["scheduler"]["ordering_policy"]

    with pytest.raises(KeyError, match="ordering_policy"):
        build_executor(config)


def test_factory_preserves_explicit_latest_ready_ablation(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["scheduler"]["ordering_policy"] = "latest_ready"

    executor = build_executor(config)

    try:
        assert executor.action_scheduler.ordering_policy == OrderingPolicy.LATEST_READY
    finally:
        executor.env.close()


@pytest.mark.parametrize(
    ("hold_policy", "expected_type"),
    [
        ("one_frame_then_noop", OneFrameThenNoopPolicy),
        ("hold", HoldPolicy),
    ],
)
def test_factory_builds_explicit_hold_policies(tmp_path, hold_policy, expected_type):
    config = _config(output_dir=str(tmp_path))
    config["env"]["obs_fps"] = 10
    config["scheduler"]["hold_policy"] = hold_policy

    executor = build_executor(config)

    try:
        assert isinstance(executor.action_scheduler.persistence_policy, expected_type)
        if hold_policy == "hold":
            assert executor.action_scheduler.persistence_policy.duration_raw_frames == 3
    finally:
        executor.env.close()


def test_factory_requires_hold_policy(tmp_path):
    config = _config(output_dir=str(tmp_path))
    del config["scheduler"]["hold_policy"]

    with pytest.raises(KeyError, match="hold_policy"):
        build_executor(config)


def test_factory_rejects_unsupported_scheduler_fields(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["scheduler"]["unsupported_scheduler_field"] = "latest_ready"

    with pytest.raises(ValueError, match="unsupported fields.*hold_policy.*ordering_policy"):
        build_executor(config)


def test_factory_rejects_unsupported_ordering_policy_values(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["scheduler"]["ordering_policy"] = "unsupported_ordering"

    with pytest.raises(ValueError):
        build_executor(config)


def test_factory_rejects_unsupported_hold_policy_values(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["scheduler"]["hold_policy"] = "unsupported_hold"

    with pytest.raises(ValueError):
        build_executor(config)


def test_simulated_eval_parallel_envs_1_keeps_old_executor_path(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 1

    executor = build_executor(config)

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.env.close()


def test_injected_env_backend_uses_common_batched_executor_with_one_slot(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
    from latency_bench.executors.env_step_backend import SerialEnvStepBackend

    config = _config(output_dir=str(tmp_path))
    env = _build_normalized_noop_env(config, Action(value="env-noop"))
    backend = SerialEnvStepBackend([env])

    executor = build_executor(config, env_backend=backend)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert executor.env_backend is backend
    finally:
        executor.close()


def test_simulated_parallel_builds_batched_executor_for_supported_zero_latency(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
    from latency_bench.executors.env_step_backend import SerialEnvStepBackend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 3
    config["evaluation"]["eval_parallel_envs"] = 4

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert isinstance(executor.env_backend, SerialEnvStepBackend)
        assert executor.num_slots == 3
    finally:
        executor.close()


def test_simulated_parallel_builds_batched_executor_for_distribution_latency(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 3
    config["evaluation"]["eval_parallel_envs"] = 4
    config["latency"] = {
        "method": "normal",
        "mean_latency_ms": 10.0,
        "std_latency_ms": 1.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 20.0,
        "seed": 0,
    }

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert executor.num_slots == 3
    finally:
        close = getattr(executor, "close", None)
        if close is not None:
            close()
        else:
            executor.env.close()


def test_seeded_stochastic_latency_source_factory_derives_episode_streams():
    from latency_bench.latency.samplers import build_episode_sampler_factory

    factory = build_episode_sampler_factory(
        {
            "method": "normal",
            "mean_latency_ms": 10.0,
            "std_latency_ms": 1.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 20.0,
            "seed": 7,
        },
        EnvClock(env_fps=30.0, obs_fps=30.0),
    )

    episode_0_first_source = factory(0)
    episode_0_second_source = factory(0)
    episode_1_source = factory(1)
    episode_0_first = [episode_0_first_source.sample({"episode_id": 0}).latency_ms for _ in range(3)]
    episode_0_second = [episode_0_second_source.sample({"episode_id": 0}).latency_ms for _ in range(3)]
    episode_1 = [episode_1_source.sample({"episode_id": 1}).latency_ms for _ in range(3)]

    assert episode_0_first == episode_0_second
    assert episode_0_first != episode_1


def test_seeded_stochastic_latency_eval_is_parallel_invariant(tmp_path):
    serial_config = _config(output_dir=str(tmp_path / "serial"))
    serial_config["evaluation"]["eval_episodes"] = 3
    serial_config["evaluation"]["eval_parallel_envs"] = 1
    serial_config["evaluation"]["eval_max_steps"] = 4
    serial_config["latency"] = {
        "method": "normal",
        "mean_latency_ms": 10.0,
        "std_latency_ms": 1.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 20.0,
        "seed": 7,
    }

    parallel_config = _config(output_dir=str(tmp_path / "parallel"))
    parallel_config["evaluation"]["eval_episodes"] = 3
    parallel_config["evaluation"]["eval_parallel_envs"] = 3
    parallel_config["evaluation"]["eval_max_steps"] = 4
    parallel_config["latency"] = dict(serial_config["latency"])

    serial_metrics = run_from_config(serial_config)
    parallel_metrics = run_from_config(parallel_config)

    assert [(metric.episode_id, metric.mean_latency_ms) for metric in serial_metrics] == [
        (metric.episode_id, metric.mean_latency_ms) for metric in parallel_metrics
    ]


def test_iid_latency_eval_is_parallel_invariant_and_loads_distribution_once(tmp_path, monkeypatch):
    import latency_bench.latency.samplers as latency_samplers

    profile_dir = _write_latency_profile(tmp_path)
    read_paths = []
    read_json = latency_samplers.read_json

    def record_read(path):
        read_paths.append(Path(path))
        return read_json(path)

    monkeypatch.setattr(latency_samplers, "read_json", record_read)

    serial_config = _config(output_dir=str(tmp_path / "serial"))
    serial_config["evaluation"]["eval_episodes"] = 3
    serial_config["evaluation"]["eval_parallel_envs"] = 1
    serial_config["evaluation"]["eval_max_steps"] = 4
    serial_config["latency"] = {
        "method": "iid",
        "profile_path": profile_dir,
        "seed": 7,
    }

    parallel_config = _config(output_dir=str(tmp_path / "parallel"))
    parallel_config["evaluation"]["eval_episodes"] = 3
    parallel_config["evaluation"]["eval_parallel_envs"] = 3
    parallel_config["evaluation"]["eval_max_steps"] = 4
    parallel_config["latency"] = dict(serial_config["latency"])

    serial_metrics = run_from_config(serial_config)
    assert len(read_paths) == 1
    read_paths.clear()
    parallel_metrics = run_from_config(parallel_config)

    assert len(read_paths) == 1
    assert [(metric.episode_id, metric.mean_latency_ms) for metric in serial_metrics] == [
        (metric.episode_id, metric.mean_latency_ms) for metric in parallel_metrics
    ]


def test_simulated_parallel_explicit_subproc_builds_subproc_backend(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
    from latency_bench.executors.env_step_backend import SubprocEnvStepBackend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 3
    config["evaluation"]["eval_parallel_envs"] = 4
    config["evaluation"]["latency_bench_env_backend"] = "subproc"
    config["env"]["name"] = "dummy"
    config["env"]["noop_action"] = "noop"
    config["env"]["action_map"] = {"noop": "noop", "jump": "jump"}

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert isinstance(executor.env_backend, SubprocEnvStepBackend)
        assert executor.num_slots == 3
        metrics = executor.run_episodes(episode_ids=range(3), seeds=[1, 2, 3], eval_max_steps=1)
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1, 2]


def test_latency_bench_env_backend_auto_selects_serial_for_demon_attack_timing_eval(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"]["latency_bench_env_backend"] = "auto"
    config["env"]["name"] = "demon_attack"
    config["policy"]["type"] = "demon_attack_sf"
    config["logging"]["save_step_records"] = False
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "serial"


def test_latency_bench_env_backend_defaults_to_auto_for_parallel_cpu_bound_envs(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"].pop("latency_bench_env_backend", None)
    config["env"]["name"] = "deadly_corridor"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "subproc"


def test_latency_bench_env_backend_defaults_to_auto_for_parallel_defend_the_line(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"].pop("latency_bench_env_backend", None)
    config["env"]["name"] = "defend_the_line"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "subproc"


def test_latency_bench_env_backend_accepts_explicit_demon_attack_gpu_batched(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"]["latency_bench_env_backend"] = "demon_attack_gpu_batched"
    config["env"]["name"] = "demon_attack"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "demon_attack_gpu_batched"


def test_latency_bench_env_backend_auto_selects_flappy_gpu_batched_backend(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"]["latency_bench_env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = True
    config["policy"]["type"] = "flappy_sf"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "flappy_gpu_batched"


def test_latency_bench_env_backend_auto_selects_flappy_gpu_batched_for_starvla_without_gpu_render(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"]["latency_bench_env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = False
    config["policy"]["type"] = "starvla"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "flappy_gpu_batched"


def test_realtime_flappy_starvla_ignores_env_backend_selection_and_keeps_cpu_render(monkeypatch, tmp_path):
    """Realtime semantics: the env steps at its own fixed cadence in a dedicated
    process and must never share resources with policy inference. The batched
    GPU backend auto-selection is simulated-only; realtime must not consult it
    and must keep the CPU (pygame) render adapter from its config."""
    from latency_bench.envs.flappy import FlappyBirdEnvAdapter

    monkeypatch.setattr(executor_factory, "_resolve_realtime_inference_devices", lambda: ["cuda:0"])

    def _fail_if_consulted(*args, **kwargs):
        raise AssertionError("realtime build must never consult latency_bench_env_backend selection")

    monkeypatch.setattr(executor_factory, "resolve_latency_bench_env_backend", _fail_if_consulted)

    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["env"] = {
        "name": "flappy",
        "gym_id": "FlappyBird-v0",
        "env_fps": 30,
        "obs_fps": 30,
        "noop_action": "noop",
        "action_map": {"noop": 0, "flap": 1},
        "oneshot_actions": ["flap"],
        "obs_resize": [84, 84],
        "frame_stack": 1,
        "use_gpu_render": False,
    }
    config["policy"] = {"type": "starvla", "actions": ["noop", "flap"]}

    executor = build_executor(config)
    try:
        assert isinstance(executor, RealtimeLatencyExecutor)
        assert isinstance(executor.env, FlappyBirdEnvAdapter)
        assert executor.env.use_gpu_render is False
    finally:
        executor.env.close()


def test_latency_bench_env_backend_auto_keeps_serial_for_flappy_sf_without_gpu_render(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 2
    config["evaluation"]["latency_bench_env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = False
    config["policy"]["type"] = "flappy_sf"

    assert resolve_latency_bench_env_backend(config, eval_parallel_envs=2) == "serial"


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


def test_realtime_mode_builds_realtime_executor(monkeypatch, tmp_path):
    monkeypatch.setattr(executor_factory, "_resolve_realtime_inference_devices", lambda: ["cuda:0"])

    executor = build_executor(_config(mode="realtime", output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, RealtimeLatencyExecutor)
        assert executor.inference_devices == ["cuda:0"]
    finally:
        executor.env.close()


def test_realtime_mode_uses_injected_env_and_inference_devices(tmp_path):
    config = _config(mode="realtime", output_dir=str(tmp_path))
    env = DummyEnvAdapter(noop_action=Action(value="env-noop", is_noop=True))

    executor = build_executor(config, env=env, inference_devices=["cuda:1"])

    try:
        assert executor.env is env
        assert executor.inference_devices == ["cuda:1"]
        assert executor.use_env_process is False
    finally:
        executor.env.close()


def test_realtime_devices_follow_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert executor_factory._resolve_realtime_inference_devices() == ["cuda:0", "cuda:1"]


def test_realtime_devices_follow_torch_visible_count_when_env_is_unset(monkeypatch):
    import torch

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert executor_factory._resolve_realtime_inference_devices() == ["cuda:0", "cuda:1"]


def test_official_gr00t_uses_every_visible_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(
        executor_factory,
        "_resolve_realtime_inference_devices",
        lambda: ["cuda:0", "cuda:1"],
    )
    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["policy"] = {"type": "official_gr00t"}

    executor = build_executor(config)

    try:
        assert executor.inference_devices == ["cuda:0", "cuda:1"]
    finally:
        executor.env.close()


def test_factory_rejects_zero_method_for_realtime(monkeypatch, tmp_path):
    monkeypatch.setattr(executor_factory, "_resolve_realtime_inference_devices", lambda: ["cuda:0"])
    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["latency"] = {"method": "zero", "device": "cuda:0", "sync_cuda": True}

    with pytest.raises(ValueError, match="latency.method=measured"):
        build_executor(config)


def test_factory_rejects_non_zero_latency_for_paused(tmp_path):
    mode = "paused"
    config = _config(mode=mode, output_dir=str(tmp_path))
    config["latency"] = {"method": "fixed", "fixed_latency_ms": 10.0}

    with pytest.raises(ValueError, match="paused"):
        build_executor(config)


def test_factory_rejects_injected_latency_for_realtime(tmp_path):
    config = _config(mode="realtime", output_dir=str(tmp_path))
    config["latency"] = {"method": "fixed", "fixed_latency_ms": 10.0}

    with pytest.raises(ValueError, match="realtime"):
        build_executor(config)


def test_factory_rejects_removed_trace_replay_method(tmp_path):
    trace_path = tmp_path / "profile_device_latency.jsonl"
    trace_path.write_text('{"latency_ms": 0.0}\n', encoding="utf-8")
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_parallel_envs"] = 4
    config["latency"] = {
        "method": "trace_replay",
        "trace_path": str(trace_path),
        "loop_trace": True,
    }

    with pytest.raises(ValueError, match="Unsupported latency method"):
        build_executor(config)


def test_factory_builds_profile_sample_latency_source(tmp_path):
    profile_dir = _write_latency_profile(tmp_path)
    config = _config(output_dir=str(tmp_path))
    config["latency"] = {
        "method": "temporal",
        "profile_path": profile_dir,
        "seed": 0,
    }

    executor = build_executor(config)

    try:
        assert executor.latency_source.latency_type == "profile_sample"
        assert executor.latency_source.profile_metadata["model_id"] == "openvla"
    finally:
        executor.env.close()


@pytest.mark.parametrize("latency_method", ["empirical_sample", "empirical_trace_replay", "measured"])
def test_factory_rejects_removed_empirical_latency_methods(tmp_path, latency_method):
    profile_dir = _write_latency_profile(tmp_path)
    config = _config(output_dir=str(tmp_path))
    config["latency"] = {
        "method": latency_method,
        "profile_path": profile_dir,
        "seed": 0,
    }

    with pytest.raises(ValueError, match="Unsupported latency method"):
        build_executor(config)


def test_parallel_simulated_tuple_action_space_builds_batched_executor_after_phase_3d(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["env"]["name"] = "tuple_action_parallel_test"
    config["env"]["noop_action"] = (0, 0)
    config["env"]["action_map"] = {"noop": (0, 0), "jump": (1, 0)}
    config["policy"]["action"] = "jump"

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
    finally:
        executor.close()


def test_parallel_simulated_profile_sample_builds_batched_executor(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    profile_dir = _write_latency_profile(tmp_path)
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["env"]["name"] = "tuple_action_parallel_test"
    config["env"]["noop_action"] = (0, 0)
    config["env"]["action_map"] = {"noop": (0, 0), "jump": (1, 0)}
    config["policy"]["action"] = "jump"
    config["latency"] = {
        "method": "temporal",
        "profile_path": profile_dir,
        "seed": 0,
    }

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
    finally:
        executor.close()


def test_temporal_latency_source_gets_no_per_episode_streams(tmp_path):
    from latency_bench.latency.samplers import HiddenRegimeSampler, build_episode_sampler_factory

    profile_dir = _write_latency_profile(tmp_path)
    latency_cfg = {
        "method": "temporal",
        "profile_path": profile_dir,
        "seed": 0,
    }

    assert build_episode_sampler_factory(
        latency_cfg,
        EnvClock(env_fps=30.0, obs_fps=30.0),
    ) is None

    config = _config(output_dir=str(tmp_path))
    config["latency"] = dict(latency_cfg)
    executor = build_executor(config)
    try:
        assert executor.episode_latency_source_factory is None
        assert isinstance(executor.latency_source, HiddenRegimeSampler)
    finally:
        executor.env.close()


def test_parallel_simulated_temporal_keeps_one_timeline_per_replica(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
    from latency_bench.latency.samplers import HiddenRegimeSampler

    profile_dir = _write_latency_profile(tmp_path)
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 3
    config["evaluation"]["eval_parallel_envs"] = 3
    config["env"]["name"] = "tuple_action_parallel_test"
    config["env"]["noop_action"] = (0, 0)
    config["env"]["action_map"] = {"noop": (0, 0), "jump": (1, 0)}
    config["policy"]["action"] = "jump"
    config["latency"] = {
        "method": "temporal",
        "profile_path": profile_dir,
        "seed": 0,
    }

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert executor.episode_latency_source_factory is None
        sources = {id(slot.latency_source) for slot in executor.slots}
        assert len(sources) == 3
        assert isinstance(executor.slots[0].latency_source, HiddenRegimeSampler)
        # Identical seeds in distinct RNG objects would still couple replicas.
        streams = [tuple(slot.latency_source.rng.random() for _ in range(8)) for slot in executor.slots]
        assert len(set(streams)) == 3
    finally:
        executor.close()


def test_parallel_simulated_nested_tuple_action_space_still_fails_fast(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["env"]["name"] = "nested_tuple_action_parallel_test"
    config["env"]["noop_action"] = ((0, 0), 0)
    config["env"]["action_map"] = {"noop": ((0, 0), 0), "jump": ((1, 0), 1)}
    config["policy"]["action"] = "jump"

    with pytest.raises(ValueError, match="nested|unsupported|Tuple"):
        build_executor(config)


def test_parallel_simulated_stateful_policy_without_predict_batch_fails_fast(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["eval_episodes"] = 2
    config["evaluation"]["eval_parallel_envs"] = 2
    config["policy"] = {
        "type": "random",
        "actions": ["jump"],
    }

    with pytest.raises(ValueError, match="does not support safe batched simulated eval"):
        build_executor(config)


@pytest.mark.parametrize("mode", ["simulated", "paused", "realtime"])
def test_executor_scheduler_uses_env_normalized_noop(monkeypatch, mode: str, tmp_path):
    if mode == "realtime":
        monkeypatch.setattr(executor_factory, "_resolve_realtime_inference_devices", lambda: ["cuda:0"])
    executor = build_executor(_config(mode=mode, output_dir=str(tmp_path)))

    try:
        assert executor.env.noop_action.value == "env-noop"
        assert executor.action_scheduler.noop_action is executor.env.noop_action
    finally:
        executor.env.close()
