"""Contracts for fitted alternatives used by training and seeded collection."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.latency.samplers import (
    build_sampler, build_episode_sampler_factory, temporal_profile_sampler_for_reset,
)
from latency_bench.latency.sequence_profiles import build_sequence_model, fit_hmm
from latency_bench.latency.sequence_samplers import HmmSampler, BlockResamplingSampler
from training.common.latency_config import sample_factory_latency_params_from_flat_config


@pytest.fixture(scope="module")
def fitted_profiles(tmp_path_factory):
    root = tmp_path_factory.mktemp("sequence_profiles")
    rng = np.random.default_rng(821)
    sessions, pairs = [], []
    for session in range(2):
        busy = np.exp(rng.normal(4 + .4 * session, .15, 150))
        overhead = np.exp(rng.normal(1, .2, 150))
        values = np.column_stack((busy + overhead, busy))
        rows = [{"obs_id": i, "episode_id": i // 50, "obs_env_step": i,
                 "submitted": True, "worker_id": 0,
                 "observation_to_action_latency_ms": float(a), "worker_service_time_ms": float(b)}
                for i, (a, b) in enumerate(values)]
        # A dropped request has no emission and must not enter either model.
        rows.append({"obs_id": 150, "episode_id": 2, "submitted": False})
        path = root / f"session_{session}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sessions.append({"measure_session_id": session, "observation_attempts": path})
        pairs.append(values)
    profiles = {}
    for method in ("hmm", "block", "ar1", "arma"):
        folder = root / method
        manifest = build_profile_from_traces(
            sessions=sessions, output_dir=folder, model_id="qwenoft", gpu_class="1x-rtx3090",
            workload_id="flappy", instance_id="test", source_run_id="measurement", env_fps=30,
            metadata={"session_ids": [0, 1]}, method=method, hmm_max_iterations=50, block_mean_length=16,
        )
        profiles[method] = folder / "profile.json"
        assert manifest["n_admitted_observations"] == 300
        assert manifest["n_capacity_drops"] == 2
        model = json.loads((folder / manifest["sequence_model_path"]).read_text())
        assert model["fit_session_lengths"] == [150, 150]
    return profiles, pairs


def _draw(sampler, n, episode=0):
    return [sampler.sample({"worker_slot": 0, "episode_id": episode}) for _ in range(n)]


@pytest.mark.parametrize("method", ["hmm", "block", "ar1", "arma"])
def test_clock_state_and_seeded_collection_contract(fitted_profiles, method):
    paths, _ = fitted_profiles
    cfg = {"method": method, "profile_path": str(paths[method]), "seed": 42}
    slow = build_sampler(cfg, clock=EnvClock(env_fps=10, obs_fps=10))
    fast = build_sampler(cfg, clock=EnvClock(env_fps=30, obs_fps=30))
    a, b = _draw(slow, 70), _draw(fast, 70)
    assert [x.latency_ms for x in a] == [x.latency_ms for x in b]
    assert all(x.action_ready_raw_frames == int(np.ceil(x.latency_ms / 100)) for x in a)
    assert any(x.action_ready_raw_frames < y.action_ready_raw_frames for x, y in zip(a, b))
    assert all(x.latency_ms >= x.worker_service_ms >= 0 for x in a)
    assert build_episode_sampler_factory(cfg, slow.clock) is None
    clone = deepcopy(slow)
    # An episode boundary preserves machine state. Deep copies replay the same
    # stream independently, including a partially consumed resampling block.
    assert _draw(slow, 100, episode=1) == _draw(clone, 100, episode=0)
    reset_a = temporal_profile_sampler_for_reset(slow, 123)
    reset_b = temporal_profile_sampler_for_reset(clone, 123)
    assert _draw(reset_a, 100) == _draw(reset_b, 100)
    assert reset_a.profile_metadata == slow.profile_metadata
    with pytest.raises(ValueError, match="worker_slot"):
        slow.sample({"worker_slot": 1, "episode_id": 2})


def test_block_stream_matches_offline_stationary_bootstrap(fitted_profiles):
    from scripts.experiments.temporal_profiles.compare import sample_blocks

    paths, pairs = fitted_profiles
    sampler = build_sampler({"method": "block", "profile_path": str(paths["block"]), "seed": 42},
                            clock=EnvClock(env_fps=10, obs_fps=10))
    result = np.array([[s.latency_ms, s.worker_service_ms] for s in _draw(sampler, 1000)])
    np.testing.assert_array_equal(result, sample_blocks(pairs, 1000, 42, 16))


def test_hmm_stream_matches_fitted_gaussian_hmm_without_clipping(fitted_profiles):
    _, pairs = fitted_profiles
    model, center, scale, params = fit_hmm([{"pairs": p} for p in pairs], max_iter=20)
    sampler = HmmSampler(params, clock=EnvClock(env_fps=10, obs_fps=10), seed=91)
    values, _ = model.sample(n_samples=200, random_state=91)
    busy_and_overhead = np.exp(values * scale + center)
    expected = np.column_stack((busy_and_overhead.sum(axis=1), busy_and_overhead[:, 0]))
    actual = np.array([[s.latency_ms, s.worker_service_ms] for s in _draw(sampler, 200)])
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("method", ["hmm", "block", "ar1", "arma"])
def test_training_adapter_retains_method_and_replay_capability(fitted_profiles, method):
    paths, _ = fitted_profiles
    cfg = SimpleNamespace(latency_type=method, simulated_worker_capacity=1,
                          latency_profile_path=str(paths[method]), latency_profile_worker_slot=0)
    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=10)
    assert params.profile_method == method
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1, fixed_latency_raw_frames=0, latency_distribution_spec=None,
        env_fps=10, obs_fps=10, seeds=[42], env_reset_seed_bases=[100], add_latency_info=False,
        profile_path=params.profile_path, profile_method=method, profile_worker_slot=0,
        simulated_worker_capacity=1, profile_seed_by_reset=True,
        hold_policy="one_frame_then_noop", ordering_policy="latest_ready",
    )
    results = []
    for repeat in range(2):
        scheduler.reset(seeds=[1000])
        run = []
        for _ in range(50):
            scheduler.begin_decision([1], [True])
            run.append((scheduler.current_latency_raw_frames.copy(), scheduler.raw_actions([True]).copy()))
            scheduler.record_raw_step([True])
        results.append(run)
    np.testing.assert_array_equal(results[0], results[1])


def test_unbounded_delays_grow_ring_storage_and_keep_pending_commands():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1, fixed_latency_raw_frames=0, latency_distribution_spec=None,
        env_fps=10, obs_fps=10, seeds=[1], env_reset_seed_bases=[10], add_latency_info=False,
        hold_policy="one_frame_then_noop", ordering_policy="latest_ready",
    )
    scheduler.reset(seeds=[1])
    # Force a tail emission much longer than the initial allocation estimate.
    from latency_bench.latency.sample import LatencySample
    class TailSampler:
        def sample(self, context):
            return LatencySample.from_latency_ms(10000, clock=scheduler.clock)
    scheduler.profile_samplers = [TailSampler()]
    for _ in range(20):
        scheduler.begin_decision([1], [True])
        scheduler.record_raw_step([True])
    assert scheduler.pending_count[0] == 20
    assert scheduler.pending_actions.shape[1] >= 20
    np.testing.assert_array_equal(scheduler.pending_action_ids[0, :20], np.arange(20))
    np.testing.assert_array_equal(scheduler.pending_ready_raw_frames[0, :20], np.arange(100, 120))


def test_sequence_fit_rejects_unsupported_worker_and_invalid_block_length():
    sample = {"worker_id": 1, "observation_to_action_latency_ms": 10., "worker_service_time_ms": 8.}
    with pytest.raises(ValueError, match="one inference worker"):
        build_sequence_model([[sample]], method="block")
    sample["worker_id"] = 0
    with pytest.raises(ValueError, match="mean length"):
        build_sequence_model([[sample]], method="block", block_mean_length=0)


@pytest.mark.parametrize("method", ["hmm", "block", "ar1", "arma"])
def test_collection_config_uses_selected_checkpoint_method_and_7200_cap(tmp_path, fitted_profiles, method):
    from scripts.experiments.temporal_profiles.prepare_collection import prepare
    from latency_bench.core.config import load_config

    paths, _ = fitted_profiles
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text(json.dumps({
        "env": "latency_flappy", "env_fps": 10, "obs_fps": 10, "action_history_decisions": 8,
        "latency_type": method, "simulated_worker_capacity": 1, "latency_profile_worker_slot": 0,
        "hold_policy": "one_frame_then_noop", "ordering_policy": "latest_ready",
    }))
    checkpoint = teacher / "best.pth"
    checkpoint.touch()
    output = tmp_path / "dataset"
    prepare(checkpoint, paths[method], output)
    config = load_config(output / "collect.yaml")
    assert config["latency"]["method"] == method
    assert config["evaluation"] == {"episodes": 200, "max_env_steps": 7200}
    assert config["collection"]["two_pass"] is True
    assert config["collection"]["filter_preset"] == "none"
    assert config["policy"]["checkpoint"] == "best.pth"
    assert load_config(output / "convert.yaml")["conversion"]["action_carrier"] == "native"


def test_torch_queue_growth_preserves_wrapped_order_without_cuda():
    import torch
    from training.common.gpu_latency import CudaLatencyScheduler

    # Exercise the actual tensor-storage operation on CPU; CUDA kernels are
    # covered by the GPU suite when a CUDA device is available.
    scheduler = CudaLatencyScheduler.__new__(CudaLatencyScheduler)
    scheduler.device = torch.device("cpu")
    scheduler.batch_size, scheduler.queue_capacity, scheduler.max_pending_actions = 1, 3, None
    scheduler.pending_head = torch.tensor([2])
    scheduler.pending_count = torch.tensor([3])
    for name in ("pending_actions", "pending_ready_raw_frames", "pending_action_ids"):
        setattr(scheduler, name, torch.tensor([[11, 12, 10]]))
    scheduler._reserve_pending_actions(torch.tensor([True]))
    assert scheduler.pending_action_ids.tolist() == [[10, 11, 12, 0, 0, 0]]
    assert scheduler.pending_head.tolist() == [0]
    assert scheduler.scratch_actions.shape == (1, 6)


@pytest.mark.parametrize("method", ["hmm", "block"])
def test_history_evaluation_selects_alternative_and_retains_game_clock(tmp_path, method):
    from scripts.tasks.decision_history.eval_vla_hist8 import evaluation_config

    (tmp_path / "provenance.json").write_text(json.dumps({"training_run_id": method, "training_step": 5000}))
    config = evaluation_config(
        "flappy", "qwenoft", tmp_path, tmp_path / "profile.json", tmp_path / "eval",
        tmp_path / "backbone", tmp_path / "python", device="cuda:0", episodes=100,
        latency_method=method,
    )
    assert config["latency"]["method"] == method
    assert config["executor"]["mode"] == "simulated"
    assert config["env"]["env_fps"] == config["env"]["obs_fps"] == 10
    assert config["evaluation"]["max_env_steps"] == 3600


@pytest.mark.parametrize("method", ["hmm", "block"])
def test_simulated_executor_runs_alternative_profiles(fitted_profiles, tmp_path, method):
    from latency_bench.executors.factory import build_executor

    paths, _ = fitted_profiles
    config = {
        "experiment": {"name": method, "seed": 42},
        "executor": {"mode": "simulated", "inference_devices": ["cpu"], "simulated_worker_capacity": 1},
        "env": {"name": "dummy", "env_fps": 10, "obs_fps": 10, "noop_action": "noop",
                "action_map": {"noop": "noop", "jump": "jump"}},
        "latency": {"method": method, "profile_path": str(paths[method]), "seed": 42},
        "scheduler": {"hold_policy": "one_frame_then_noop", "ordering_policy": "latest_ready"},
        "policy": {"type": "constant", "action": "jump"},
        "evaluation": {"episodes": 2, "max_env_steps": 30},
        "logging": {"output_dir": str(tmp_path / method)},
    }
    executor = build_executor(config)
    try:
        for episode in range(2):
            result = executor.run_episode(episode, seed=episode, max_env_steps=30)
            assert result.survival_steps == 30
            assert result.episode_return == 30
    finally:
        executor.close()
