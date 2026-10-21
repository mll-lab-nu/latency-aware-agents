from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import Action, OrderingPolicy, PolicyOutput
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.adapters.dummy import DummyEnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.executors.factory import build_executor
import latency_bench.executors.factory as executor_factory
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime.executor import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.eval.driver import run_from_config
from latency_bench.latency.burst_profile import (
    build_hidden_regime_burst_model_from_sessions,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.latency.samplers import JointBurstSampler
from latency_bench.core.clock import EnvClock
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy
from latency_bench.policy.action_prefix import prepare_action_prefix
from latency_bench.policy.base import PolicyRunner
from latency_bench.utils.io import write_json


@pytest.mark.parametrize("mean_ms,fps,expected", [
    (37.880780, 50, 2), (99.176261, 50, 5), (60.013398, 50, 4),
    (60.013398, 20, 2), (0, 50, 0),
])
def test_task_prefix_configuration_freezes_profile_mean(tmp_path, mean_ms, fps, expected):
    """Long-lived factory contract: K comes from unrounded profile time and env FPS."""
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"mean_ms": mean_ms}}))
    contract = tmp_path / "task_contract.json"
    contract.write_text(json.dumps({"action_horizon": 8, "action_dim": 1}))
    config = {"env": {"env_fps": fps}, "policy": {
        "task_contract_path": str(contract),
        "action_prefix": {"mode": "profile_mean", "profile_path": str(profile)},
    }}
    resolved = prepare_action_prefix(config)
    assert resolved["frames"] == expected
    assert resolved["mean_ms"] == mean_ms
    profile.unlink()
    assert prepare_action_prefix(config) == resolved


def test_task_prefix_configuration_requires_a_suffix_and_explicit_mode(tmp_path):
    """Long-lived config boundary: no empty suffix or implicit baseline selection."""
    contract = tmp_path / "task_contract.json"
    contract.write_text(json.dumps({"action_horizon": 1, "action_dim": 1}))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"mean_ms": 20.0}}))
    config = {"env": {"env_fps": 50}, "policy": {
        "task_contract_path": str(contract),
        "action_prefix": {"mode": "profile_mean", "profile_path": str(profile)},
    }}
    with pytest.raises(ValueError, match="K=1, H=1"):
        prepare_action_prefix(config)
    config["policy"]["action_prefix"] = {"mode": "none"}
    assert prepare_action_prefix(config) == {"mode": "none"}
    del config["policy"]["action_prefix"]
    with pytest.raises(KeyError, match="action_prefix"):
        prepare_action_prefix(config)


@register_env_adapter("normalized_noop_test")
def _build_normalized_noop_env(config: dict, noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    evaluation_cfg = config["evaluation"]
    return DummyEnvAdapter(
        env_fps=float(env_cfg["env_fps"]),
        max_steps=int(evaluation_cfg["max_env_steps"]),
        noop_action=Action(value="env-noop", name="env-noop", is_noop=True),
    )


def _config(mode: str = "simulated", output_dir: str = "outputs/executor_factory_test") -> dict:
    return {
        "experiment": {
            "name": "executor_factory_test",
            "seed": 0,
        },
        "executor": {
            "inference_devices": ["cpu"],
            "mode": mode,
        },
        "env": {
            "name": "dummy" if mode == "realtime" else "normalized_noop_test",
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
            "episodes": 1,
            "max_env_steps": 5,
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
    burst_model = build_hidden_regime_burst_model_from_sessions([burst_samples])
    write_json(
        profile_dir / "latency_distribution.json",
        build_worker_slot_distributions_from_sessions(
            [burst_samples],
            spike_threshold_ms_by_worker_slot=burst_model[
                "spike_threshold_ms_by_worker_slot"
            ],
        ),
    )
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(burst_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        profile_dir / "profile.json",
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
        max_steps=int(evaluation_cfg["max_env_steps"]),
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
        max_steps=int(evaluation_cfg["max_env_steps"]),
        noop_action=noop_action,
    )
    env.gym_env = env
    return env


def test_simulated_mode_builds_simulated_executor(tmp_path):
    executor = build_executor(_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.close()


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("configured_devices", [None, ["cuda:1"]])
def test_simulated_inference_pool_builds_pool_and_no_parent_policy(
    tmp_path, monkeypatch, parallel, configured_devices
):
    created = {}

    class FakePool:
        def __init__(self, *, config, inference_devices):
            created["devices"] = list(inference_devices)

        def close(self):
            created["closed"] = True

    def _fail_build_policy(*args, **kwargs):
        raise AssertionError("the parent process must not load the policy in pool mode")

    monkeypatch.setattr(executor_factory, "ProcessInferencePool", FakePool)
    monkeypatch.setattr(executor_factory, "build_policy", _fail_build_policy)
    config = _config(output_dir=str(tmp_path))
    config["executor"]["inference_pool"] = True
    config["executor"]["inference_devices"] = ['cuda']
    if parallel:
        config["evaluation"]["episodes"] = 2
        config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2 if parallel else 1
    if configured_devices is not None:
        config['executor']['inference_devices'] = configured_devices

    executor = build_executor(config)
    try:
        if parallel:
            assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        else:
            assert isinstance(executor, SimulatedLatencyExecutor)
        assert isinstance(executor.inference_pool, FakePool)
        assert created["devices"] == (configured_devices if configured_devices is not None else ["cuda"])
        assert executor.policy is None
    finally:
        executor.close()
    assert created.get("closed") is True


def test_simulated_inference_pool_rejects_injected_policy(tmp_path, monkeypatch):
    config = _config(output_dir=str(tmp_path))
    config["executor"]["inference_pool"] = True
    config["executor"]["inference_devices"] = ['cuda']

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
        executor.close()


def test_paused_executor_uses_injected_environment(tmp_path):
    env = DummyEnvAdapter(env_fps=30, max_steps=2)
    executor = build_executor(_config(mode="paused", output_dir=str(tmp_path)), env=env)
    try:
        metrics = executor.run_episode(episode_id=0, seed=7, max_env_steps=5)
        assert metrics.survival_steps == 2
    finally:
        executor.close()
    assert env.closed


@pytest.mark.parametrize("mode", ["paused", "simulated"])
def test_policy_construction_failure_closes_environment(tmp_path, monkeypatch, mode):
    """Regression: failed executor construction must release its environment."""
    def fail_policy(config, action_resolver):
        raise RuntimeError("user policy construction failed")

    monkeypatch.setattr(executor_factory, "build_policy", fail_policy)
    env = DummyEnvAdapter()
    with pytest.raises(RuntimeError, match="user policy construction failed"):
        build_executor(_config(mode=mode, output_dir=str(tmp_path)), env=env)
    assert env.closed


def test_build_executor_realtime_rejects_injected_policy(tmp_path):
    with pytest.raises(ValueError, match="cannot cross the process boundary"):
        build_executor(_config(mode="realtime", output_dir=str(tmp_path)), policy=object())


def test_realtime_rejects_live_task_backend(tmp_path):
    """Native GPU environments must be constructed by their owning process."""
    config = _config(mode="realtime", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="cannot cross the process boundary"):
        build_executor(config, env_backend=object(), inference_devices=["cpu"])


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
        executor.close()


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
    finally:
        executor.close()


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
    config["executor"]["parallel_envs"] = 1

    executor = build_executor(config)

    try:
        assert isinstance(executor, SimulatedLatencyExecutor)
    finally:
        executor.close()


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
    config["evaluation"]["episodes"] = 3
    config["executor"]["parallel_envs"] = 4

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
    config["evaluation"]["episodes"] = 3
    config["executor"]["parallel_envs"] = 4
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
            executor.close()


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
    serial_config["evaluation"]["episodes"] = 3
    serial_config["executor"]["parallel_envs"] = 1
    serial_config["evaluation"]["max_env_steps"] = 4
    serial_config["latency"] = {
        "method": "normal",
        "mean_latency_ms": 10.0,
        "std_latency_ms": 1.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 20.0,
        "seed": 7,
    }

    parallel_config = _config(output_dir=str(tmp_path / "parallel"))
    parallel_config["evaluation"]["episodes"] = 3
    parallel_config["executor"]["parallel_envs"] = 3
    parallel_config["evaluation"]["max_env_steps"] = 4
    parallel_config["latency"] = dict(serial_config["latency"])

    serial_metrics = run_from_config(serial_config)
    parallel_metrics = run_from_config(parallel_config)

    assert [(metric.episode_id, metric.mean_latency_ms) for metric in serial_metrics] == [
        (metric.episode_id, metric.mean_latency_ms) for metric in parallel_metrics
    ]


def test_iid_latency_eval_is_parallel_invariant_and_loads_distribution_once(tmp_path, monkeypatch):
    import latency_bench.latency.samplers as latency_samplers

    profile_dir = _write_latency_profile(tmp_path)
    distribution_path = Path(profile_dir) / "latency_distribution.json"
    read_paths = []
    read_json = latency_samplers.read_json

    def record_read(path):
        if Path(path) == distribution_path:
            read_paths.append(Path(path))
        return read_json(path)

    monkeypatch.setattr(latency_samplers, "read_json", record_read)

    serial_config = _config(output_dir=str(tmp_path / "serial"))
    serial_config["evaluation"]["episodes"] = 3
    serial_config["executor"]["parallel_envs"] = 1
    serial_config["evaluation"]["max_env_steps"] = 4
    serial_config["latency"] = {
        "method": "iid",
        "profile_path": profile_dir,
        "seed": 7,
    }

    parallel_config = _config(output_dir=str(tmp_path / "parallel"))
    parallel_config["evaluation"]["episodes"] = 3
    parallel_config["executor"]["parallel_envs"] = 3
    parallel_config["evaluation"]["max_env_steps"] = 4
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
    config["evaluation"]["episodes"] = 3
    config["executor"]["parallel_envs"] = 4
    config["executor"]["env_backend"] = "subproc"
    config["env"]["name"] = "dummy"
    config["env"]["noop_action"] = "noop"
    config["env"]["action_map"] = {"noop": "noop", "jump": "jump"}

    executor = build_executor(config)

    try:
        assert isinstance(executor, BatchedSimulatedLatencyExecutor)
        assert isinstance(executor.env_backend, SubprocEnvStepBackend)
        assert executor.num_slots == 3
        metrics = executor.run_episodes(episode_ids=range(3), seeds=[1, 2, 3], max_env_steps=1)
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1, 2]


def test_latency_bench_env_backend_auto_selects_serial_for_demon_attack_timing_eval(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["executor"]["env_backend"] = "auto"
    config["env"]["name"] = "demon_attack"
    config["policy"]["type"] = "sample_factory"
    config["logging"]["save_step_records"] = False
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == "serial"


def test_latency_bench_env_backend_defaults_to_auto_for_parallel_cpu_bound_envs(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["evaluation"].pop("latency_bench_env_backend", None)
    config["env"]["name"] = "deadly_corridor"

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == "subproc"


@pytest.mark.parametrize(
    "backend",
    [
        "demon_attack_gpu_batched",
        "atlantis_gpu_batched",
        "asterix_gpu_batched",
    ],
)
def test_latency_bench_env_backend_accepts_explicit_atari_gpu_batched(tmp_path, backend):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["executor"]["env_backend"] = backend

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == backend


def test_latency_bench_env_backend_auto_selects_flappy_gpu_batched_backend(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["executor"]["env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = True
    config["policy"]["type"] = "sample_factory"

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == "flappy_gpu_batched"


def test_latency_bench_env_backend_auto_selects_flappy_gpu_batched_for_starvla_without_gpu_render(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["executor"]["env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = False
    config["policy"]["type"] = "starvla"

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == "flappy_gpu_batched"


def test_realtime_flappy_starvla_ignores_env_backend_selection_and_keeps_cpu_render(monkeypatch, tmp_path):
    """Realtime semantics: the env steps at its own fixed cadence in a dedicated
    process and must never share resources with policy inference. The batched
    GPU backend auto-selection is simulated-only; realtime must not consult it
    and must keep the CPU (pygame) render adapter from its config."""


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
        executor.env.reset(seed=0)
        task = executor.env.warmup_observation()
        assert task.observation.env_step == 0
        executor.env.release_observation(task.shared_obs_id)
        assert config["env"]["use_gpu_render"] is False
    finally:
        executor.close()


def test_latency_bench_env_backend_auto_keeps_serial_for_flappy_sf_without_gpu_render(tmp_path):
    from latency_bench.executors.backend_selection import resolve_latency_bench_env_backend

    config = _config(output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["executor"]["env_backend"] = "auto"
    config["env"]["name"] = "flappy"
    config["env"]["use_gpu_render"] = False
    config["policy"]["type"] = "sample_factory"

    assert resolve_latency_bench_env_backend(config, parallel_envs=2) == "serial"


@pytest.mark.parametrize("mode", ["paused", "realtime"])
def test_non_simulated_modes_reject_parallel_eval(tmp_path, mode):
    config = _config(mode=mode, output_dir=str(tmp_path))
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2

    with pytest.raises(ValueError, match="parallel_envs.*simulated"):
        build_executor(config)


def test_paused_mode_builds_paused_executor(tmp_path):
    executor = build_executor(_config(mode="paused", output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, PausedLatencyExecutor)
    finally:
        executor.close()


def test_realtime_mode_builds_realtime_executor(monkeypatch, tmp_path):

    executor = build_executor(_config(mode="realtime", output_dir=str(tmp_path)))

    try:
        assert isinstance(executor, RealtimeLatencyExecutor)
        assert executor.inference_devices == ["cpu"]
    finally:
        executor.close()


def test_realtime_mode_owns_environment_in_another_process(tmp_path):
    import os
    from latency_bench.executors.realtime.env_process import RealtimeEnvProcess

    config = _config(mode="realtime", output_dir=str(tmp_path))
    executor = build_executor(config, inference_devices=["cuda:1"])
    try:
        assert isinstance(executor.env, RealtimeEnvProcess)
        assert executor.env.process.pid != os.getpid()
        assert executor.inference_devices == ["cuda:1"]
        executor.env.reset(seed=7)
        task = executor.env.warmup_observation()
        assert task.observation.env_step == 0
        executor.env.release_observation(task.shared_obs_id)
        assert executor.env.step(executor.env.noop_action).observation.env_step == 1
    finally:
        executor.close()
    assert not executor.env.process.is_alive()


def test_realtime_factory_releases_environment_after_setup_failure(monkeypatch, tmp_path):
    """The factory owns worker cleanup until executor construction succeeds."""
    from latency_bench.executors.realtime.env_process import RealtimeEnvProcess

    environments = []

    def create(config, **kwargs):
        environment = RealtimeEnvProcess(config, **kwargs)
        environments.append(environment)
        return environment

    def fail_logger(config):
        raise OSError("cannot create log")

    monkeypatch.setattr(executor_factory, "RealtimeEnvProcess", create)
    monkeypatch.setattr(executor_factory, "_build_trajectory_logger", fail_logger)
    with pytest.raises(OSError, match="cannot create log"):
        build_executor(_config(mode="realtime", output_dir=str(tmp_path)), inference_devices=["cpu"])
    assert not environments[0].process.is_alive()


def test_factory_rejects_unsupported_latency_method(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["latency"] = {"method": "unsupported_method"}

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
        executor.close()


def test_parallel_simulated_tuple_action_space_builds_batched_executor_after_phase_3d(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["episodes"] = 2
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
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
    config["evaluation"]["episodes"] = 2
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
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
    from latency_bench.latency.samplers import build_episode_sampler_factory

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
        assert isinstance(executor.latency_source, JointBurstSampler)
    finally:
        executor.close()


def test_parallel_simulated_temporal_keeps_one_timeline_per_replica(tmp_path):
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor

    profile_dir = _write_latency_profile(tmp_path)
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["episodes"] = 3
    config["executor"]["parallel_envs"] = 3
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
        assert isinstance(executor.slots[0].latency_source, JointBurstSampler)
        # Identical seeds in distinct RNG objects would still couple replicas.
        streams = [tuple(slot.latency_source.rng.random() for _ in range(8)) for slot in executor.slots]
        assert len(set(streams)) == 3
    finally:
        executor.close()


def test_parallel_simulated_nested_tuple_action_space_still_fails_fast(tmp_path):
    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["episodes"] = 2
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2
    config["env"]["name"] = "nested_tuple_action_parallel_test"
    config["env"]["noop_action"] = ((0, 0), 0)
    config["env"]["action_map"] = {"noop": ((0, 0), 0), "jump": ((1, 0), 1)}
    config["policy"]["action"] = "jump"

    with pytest.raises(ValueError, match="nested|unsupported|Tuple"):
        build_executor(config)


def test_parallel_simulated_policy_without_predict_batch_fails_fast(tmp_path):
    class SingleObservationPolicy(PolicyRunner):
        """Scalar-only fixture for the executor's batch prediction contract."""

        def predict(self, observation):
            return PolicyOutput(action=Action(value="jump", name="jump"))

    config = _config(output_dir=str(tmp_path))
    config["evaluation"]["episodes"] = 2
    config["executor"]["env_backend"] = "auto"
    config["executor"]["parallel_envs"] = 2

    with pytest.raises(ValueError, match="does not support safe batched simulated eval"):
        build_executor(config, policy=SingleObservationPolicy())


@pytest.mark.parametrize("mode", ["simulated", "paused", "realtime"])
def test_executor_scheduler_uses_env_normalized_noop(monkeypatch, mode: str, tmp_path):
    if mode == "realtime":
        monkeypatch.setattr(executor_factory, "RealtimeEnvProcess", lambda config, **kwargs:
                            _build_normalized_noop_env(config, Action(value="env-noop", is_noop=True)))
    executor = build_executor(_config(mode=mode, output_dir=str(tmp_path)))

    try:
        assert executor.env.noop_action.value == "env-noop"
        assert executor.action_scheduler.noop_action is executor.env.noop_action
    finally:
        executor.close()
