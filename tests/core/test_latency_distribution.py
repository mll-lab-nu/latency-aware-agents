from __future__ import annotations

import pytest

from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
    parse_latency_distribution,
    sample_latency_frame,
)


def test_parse_normal_distribution_and_builds_stable_label():
    spec = parse_latency_distribution(
        {
            "type": "normal",
            "mean": 5.0,
            "std": 2.25,
            "min": 0,
            "max": 15.0,
        }
    )

    assert spec == LatencyDistributionSpec(type="normal", mean=5.0, std=2.25, min=0, max=15)
    assert latency_distribution_label(spec) == "normal_mean_5_std_2p25_min_0_max_15"
    parsed = parse_latency_distribution('{"type":"normal","mean":5,"std":2.25,"min":0,"max":15}')
    assert latency_distribution_label(parsed) == (
        "normal_mean_5_std_2p25_min_0_max_15"
    )
    assert distribution_max_frame_latency(spec) == 15


def test_parse_uniform_distribution_and_builds_stable_label():
    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 0})

    assert spec == LatencyDistributionSpec(type="uniform", mean=None, std=None, min=0, max=0)
    assert latency_distribution_label(spec) == "uniform_min_0_max_0"


def test_sample_latency_frame_clamps_then_uses_ceil_conversion():
    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 15})

    assert sample_latency_frame(spec, 0.01) == 1
    assert sample_latency_frame(spec, 3.0) == 3
    assert sample_latency_frame(spec, 3.01) == 4
    assert sample_latency_frame(spec, 99.0) == 15


def test_sample_latency_frame_supports_fractional_bounds():
    spec = parse_latency_distribution({"type": "uniform", "min": 1.2, "max": 3.2})

    assert spec.to_dict() == {"type": "uniform", "min": 1.2, "max": 3.2}
    assert latency_distribution_label(spec) == "uniform_min_1p2_max_3p2"
    assert distribution_max_frame_latency(spec) == 4
    assert sample_latency_frame(spec, 0.0) == 2
    assert sample_latency_frame(spec, 99.0) == 4


def test_parse_distribution_rejects_unknown_type():
    with pytest.raises(ValueError, match="type"):
        parse_latency_distribution({"type": "unknown", "min": 0, "max": 1})


def test_samplers_are_deterministic_without_global_random_state():
    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 15})
    first = make_latency_distribution_sampler(spec, seed=derive_seed(7, worker_index=1, vector_index=2))
    second = make_latency_distribution_sampler(spec, seed=derive_seed(7, worker_index=1, vector_index=2))
    other = make_latency_distribution_sampler(spec, seed=derive_seed(7, worker_index=1, vector_index=3))

    assert [first.sample_frame() for _ in range(20)] == [second.sample_frame() for _ in range(20)]
    assert [second.sample_frame() for _ in range(20)] != [other.sample_frame() for _ in range(20)]
