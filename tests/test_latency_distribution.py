from __future__ import annotations

import pytest

from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
    parse_latency_distribution,
    round_half_up_frame,
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
    assert latency_distribution_label({"type": "normal", "mean": 5, "std": 2.25, "min": 0, "max": 15}) == (
        "normal_mean_5_std_2p25_min_0_max_15"
    )
    assert distribution_max_frame_latency(spec) == 15


def test_parse_uniform_distribution_and_builds_stable_label():
    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 0})

    assert spec == LatencyDistributionSpec(type="uniform", mean=None, std=None, min=0, max=0)
    assert latency_distribution_label(spec) == "uniform_min_0_max_0"


def test_round_half_up_clamps_before_integer_conversion():
    assert round_half_up_frame(-10.0, min_frame=0, max_frame=15) == 0
    assert round_half_up_frame(3.49, min_frame=0, max_frame=15) == 3
    assert round_half_up_frame(3.5, min_frame=0, max_frame=15) == 4
    assert round_half_up_frame(99.0, min_frame=0, max_frame=15) == 15
    assert round_half_up_frame(1.2, min_frame=3, max_frame=15) == 3


@pytest.mark.parametrize(
    ("raw", "match"),
    (
        ({"type": "normal", "mean": 5, "min": 0, "max": 15}, "std"),
        ({"type": "normal", "mean": 5, "std": -1, "min": 0, "max": 15}, "std"),
        ({"type": "uniform", "min": 0.5, "max": 15}, "integer"),
        ({"type": "uniform", "min": 2, "max": 1}, "min.*max"),
        ({"type": "unknown", "min": 0, "max": 1}, "type"),
    ),
)
def test_distribution_validation_rejects_invalid_specs(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_latency_distribution(raw)


def test_normal_mean_may_be_outside_bounds_but_samples_are_clamped_and_rounded():
    sampler = make_latency_distribution_sampler(
        {"type": "normal", "mean": -10.0, "std": 0.0, "min": 0, "max": 15},
        seed=123,
    )

    assert [sampler.sample_frame() for _ in range(3)] == [0, 0, 0]


def test_samplers_are_deterministic_without_global_random_state():
    spec = {"type": "uniform", "min": 0, "max": 15}
    first = make_latency_distribution_sampler(spec, seed=derive_latency_seed(7, worker_index=1, vector_index=2))
    second = make_latency_distribution_sampler(spec, seed=derive_latency_seed(7, worker_index=1, vector_index=2))
    other = make_latency_distribution_sampler(spec, seed=derive_latency_seed(7, worker_index=1, vector_index=3))

    assert [first.sample_frame() for _ in range(20)] == [second.sample_frame() for _ in range(20)]
    assert [second.sample_frame() for _ in range(20)] != [other.sample_frame() for _ in range(20)]
