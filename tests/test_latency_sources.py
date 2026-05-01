from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.core.types import LatencyRecord
from latency_bench.latency.sources import (
    FixedLatencySource,
    NormalDistributionLatencySource,
    TraceReplayLatencySource,
    ZeroLatencySource,
)
from latency_bench.latency.trace import load_latency_trace, save_latency_trace


def test_zero_latency_source_returns_zero():
    assert ZeroLatencySource().next_latency_ms() == 0.0


def test_fixed_latency_source_returns_constant():
    source = FixedLatencySource(42.5)

    assert source.next_latency_ms() == 42.5
    assert source.next_latency_ms() == 42.5


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


def test_normal_distribution_rejects_negative_std():
    with pytest.raises(ValueError, match="std_latency_ms must be non-negative"):
        NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=-1.0)


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
        [LatencyRecord(obs_id=0, latency_ms=12.0, source="test")],
        trace_path,
    )

    assert load_latency_trace(trace_path) == [12.0]
