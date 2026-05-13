from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.core.types import LatencyRecord
from latency_bench.latency.sources import (
    EmpiricalSampleLatencySource,
    EmpiricalTraceReplayLatencySource,
    FixedLatencySource,
    NormalDistributionLatencySource,
    TraceReplayLatencySource,
    UniformDistributionLatencySource,
    ZeroLatencySource,
)
from latency_bench.latency.profile import save_profile_manifest, write_latency_samples
from latency_bench.latency.trace import load_latency_trace, save_latency_trace


def _write_latency_profile(
    tmp_path: Path,
    samples: list[float] | None = None,
    *,
    trace_path: str = "latencies.jsonl",
    write_trace: bool = True,
) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    samples = [10.0, 20.0, 30.0] if samples is None else samples
    if write_trace:
        write_latency_samples(samples, profile_dir / trace_path)
    save_profile_manifest(
        {
            "schema_version": 1,
            "profile_id": "L_test",
            "profile_type": "rollout_profile",
            "policy_id": "pi_test",
            "env_id": "env_test",
            "target_device": "test_gpu",
            "fps": 25.0,
            "frame_ms": 40.0,
            "n_warmup": 0,
            "n_samples": len(samples),
            "batch_size": 1,
            "trace_path": trace_path,
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def test_zero_latency_source_returns_zero():
    assert ZeroLatencySource().next_latency_ms() == 0.0


def test_fixed_latency_source_returns_constant():
    source = FixedLatencySource(42.5)

    assert source.next_latency_ms() == 42.5
    assert source.next_latency_ms() == 42.5


def test_fixed_latency_source_preserves_contract_value_without_range_validation():
    source = FixedLatencySource(-4.5)

    assert source.fixed_latency_ms == -4.5
    assert source.next_latency_ms() == -4.5


def test_normal_distribution_samples_deterministically_with_seed():
    first = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=20.0, seed=0)
    second = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=20.0, seed=0)

    assert [first.next_latency_ms() for _ in range(5)] == [second.next_latency_ms() for _ in range(5)]


def test_normal_distribution_clamps_to_zero_by_default():
    source = NormalDistributionLatencySource(mean_latency_ms=-10.0, std_latency_ms=0.0)

    assert source.next_latency_ms() == 0.0


def test_normal_distribution_respects_min_and_max_bounds():
    low = NormalDistributionLatencySource(mean_latency_ms=5.0, std_latency_ms=0.0, min_latency_ms=10.0)
    high = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=0.0, max_latency_ms=50.0)

    assert low.next_latency_ms() == 10.0
    assert high.next_latency_ms() == 50.0


def test_normal_distribution_preserves_contract_values_without_range_validation():
    source = NormalDistributionLatencySource(
        mean_latency_ms=100.0,
        std_latency_ms=-1.0,
        min_latency_ms=-5.0,
        max_latency_ms=1.0,
    )

    assert source.mean_latency_ms == 100.0
    assert source.std_latency_ms == -1.0
    assert source.min_latency_ms == -5.0
    assert source.max_latency_ms == 1.0


def test_uniform_distribution_samples_deterministically_with_seed_and_bounds():
    first = UniformDistributionLatencySource(min_latency_ms=10.0, max_latency_ms=20.0, seed=0)
    second = UniformDistributionLatencySource(min_latency_ms=10.0, max_latency_ms=20.0, seed=0)

    first_values = [first.next_latency_ms() for _ in range(10)]
    second_values = [second.next_latency_ms() for _ in range(10)]

    assert first_values == second_values
    assert all(10.0 <= value <= 20.0 for value in first_values)


def test_uniform_distribution_preserves_contract_values_without_range_validation():
    source = UniformDistributionLatencySource(min_latency_ms=20.0, max_latency_ms=10.0)

    assert source.min_latency_ms == 20.0
    assert source.max_latency_ms == 10.0


def test_trace_replay_returns_values_in_order(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"obs_id": 0, "latency_ms": 35.2}\n'
        '{"obs_id": 1, "latency_ms": 41.8}\n',
        encoding="utf-8",
    )
    source = TraceReplayLatencySource(trace_path)

    assert source.next_latency_ms() == pytest.approx(35.2)
    assert source.next_latency_ms() == pytest.approx(41.8)
    with pytest.raises(StopIteration):
        source.next_latency_ms()


def test_trace_replay_loops_when_enabled(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"obs_id": 0, "latency_ms": 1.0}\n'
        '{"obs_id": 1, "latency_ms": 2.0}\n',
        encoding="utf-8",
    )
    source = TraceReplayLatencySource(trace_path, loop_trace=True)

    assert [source.next_latency_ms() for _ in range(5)] == [1.0, 2.0, 1.0, 2.0, 1.0]


def test_latency_trace_load_and_save(tmp_path: Path):
    trace_path = tmp_path / "saved.jsonl"
    save_latency_trace(
        [LatencyRecord(obs_id=0, latency_ms=12.0)],
        trace_path,
    )

    assert load_latency_trace(trace_path) == [12.0]


def test_empirical_sample_samples_deterministically_with_seed(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = EmpiricalSampleLatencySource(profile_dir / "profile.json", seed=0)
    second = EmpiricalSampleLatencySource(profile_dir / "profile.json", seed=0)

    assert [first.next_latency_ms() for _ in range(20)] == [second.next_latency_ms() for _ in range(20)]
    assert first.profile_id == "L_test"
    assert first.profile_type == "rollout_profile"
    assert first.target_device == "test_gpu"
    assert first.last_sample_index is not None


def test_empirical_sample_values_come_from_profile(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = EmpiricalSampleLatencySource(profile_dir / "profile.json", seed=1)

    assert {source.next_latency_ms() for _ in range(100)} <= {10.0, 20.0, 30.0}


def test_empirical_trace_replay_sequential_loop(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = EmpiricalTraceReplayLatencySource(profile_dir / "profile.json", loop_trace=True)

    assert [source.next_latency_ms() for _ in range(5)] == [10.0, 20.0, 30.0, 10.0, 20.0]
    assert source.last_trace_index == 1


def test_empirical_trace_replay_no_loop_exhaustion(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = EmpiricalTraceReplayLatencySource(profile_dir / "profile.json", loop_trace=False)

    assert [source.next_latency_ms() for _ in range(3)] == [10.0, 20.0, 30.0]
    with pytest.raises(StopIteration, match="Empirical latency trace exhausted"):
        source.next_latency_ms()


def test_empirical_trace_replay_random_order_is_seeded(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = EmpiricalTraceReplayLatencySource(profile_dir / "profile.json", trace_replay_order="random", seed=3)
    second = EmpiricalTraceReplayLatencySource(profile_dir / "profile.json", trace_replay_order="random", seed=3)

    assert [first.next_latency_ms() for _ in range(6)] == [second.next_latency_ms() for _ in range(6)]


def test_empirical_latency_sources_accept_profile_directory(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = EmpiricalSampleLatencySource(profile_dir, seed=0)

    assert source.profile_path == profile_dir / "profile.json"
    assert source.next_latency_ms() in {10.0, 20.0, 30.0}


def test_empirical_latency_sources_reject_missing_profile_path():
    with pytest.raises(ValueError, match="profile_path is required"):
        EmpiricalSampleLatencySource(None)


def test_empirical_latency_sources_reject_missing_trace_file(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, trace_path="missing.jsonl", write_trace=False)

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        EmpiricalSampleLatencySource(profile_dir / "profile.json")


def test_empirical_trace_replay_rejects_invalid_order(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)

    with pytest.raises(ValueError, match="trace_replay_order"):
        EmpiricalTraceReplayLatencySource(profile_dir / "profile.json", trace_replay_order="reverse")


def test_empirical_latency_sources_pass_seed_through_to_rng(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = EmpiricalSampleLatencySource(profile_dir / "profile.json", seed=True)

    assert source.next_latency_ms() in {10.0, 20.0, 30.0}
