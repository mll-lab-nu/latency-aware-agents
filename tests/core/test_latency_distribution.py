from __future__ import annotations

import pytest

from latency_bench.core.latency_distribution import (
    LatencyDistributionSampler,
    LatencyDistributionSpec,
    derive_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    parse_latency_distribution,
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


@pytest.mark.parametrize(
    "lower,upper,label,max_frames",
    [(0, 0, "uniform_min_0_max_0", 0), (1.2, 3.2, "uniform_min_1p2_max_3p2", 4)],
)
def test_parse_uniform_distribution_and_builds_stable_label(lower, upper, label, max_frames):
    spec = parse_latency_distribution({"type": "uniform", "min": lower, "max": upper})

    assert spec == LatencyDistributionSpec(type="uniform", mean=None, std=None, min=lower, max=upper)
    assert spec.to_dict() == {"type": "uniform", "min": lower, "max": upper}
    assert latency_distribution_label(spec) == label
    assert distribution_max_frame_latency(spec) == max_frames


@pytest.mark.parametrize(
    "lower,upper,sampled,expected",
    [
        (0, 15, 0.01, 1),
        (0, 15, 3.0, 3),
        (0, 15, 3.01, 4),
        (0, 15, 99.0, 15),
        (1.2, 3.2, 0.0, 2),
        (1.2, 3.2, 99.0, 4),
    ],
)
def test_sampler_clamps_then_rounds_up_to_raw_frames(lower, upper, sampled, expected):
    """Long-lived core contract: sampled latency respects bounds and frame rounding."""
    spec = LatencyDistributionSpec(type="normal", mean=sampled, std=0.0, min=lower, max=upper)
    assert LatencyDistributionSampler(spec, seed=0).sample_frame() == expected


def test_parse_distribution_rejects_unknown_type():
    with pytest.raises(ValueError, match="type"):
        parse_latency_distribution({"type": "unknown", "min": 0, "max": 1})


def test_samplers_are_deterministic_without_global_random_state():
    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 15})
    first = LatencyDistributionSampler(spec, seed=derive_seed(7, worker_index=1, vector_index=2))
    second = LatencyDistributionSampler(spec, seed=derive_seed(7, worker_index=1, vector_index=2))
    other = LatencyDistributionSampler(spec, seed=derive_seed(7, worker_index=1, vector_index=3))

    assert [first.sample_frame() for _ in range(20)] == [second.sample_frame() for _ in range(20)]
    assert [second.sample_frame() for _ in range(20)] != [other.sample_frame() for _ in range(20)]
