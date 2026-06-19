from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.burst_profile import build_latency_burst_model
from latency_bench.latency.profile import build_inverse_cdf_distribution
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import (
    FixedSampler,
    IidInverseCdfSampler,
    QueueConditionedSampler,
    SemiMarkovSampler,
)


def _clock() -> EnvClock:
    return EnvClock(env_fps=25.0, obs_fps=25.0)


def test_from_latency_ms_fills_derived_fields():
    clock = _clock()
    sample = LatencySample.from_latency_ms(40.0, clock=clock)
    assert sample == LatencySample(
        latency_ms=40.0,
        latency_raw_frames=1,
        action_ready_raw_frames=1,
        worker_service_raw_frames=1,
        worker_service_ms=40.0,
    )


def test_from_action_ready_derives_latency_ms_from_action_ready():
    clock = _clock()
    sample = LatencySample.from_action_ready(
        action_ready_raw_frames=4,
        worker_service_raw_frames=5,
        worker_service_ms=110.0,
        clock=clock,
    )
    assert sample.latency_ms == pytest.approx(160.0)
    assert sample.latency_raw_frames == 4
    assert sample.worker_service_raw_frames == 5
    assert sample.worker_service_ms == 110.0


def test_iid_inverse_cdf_matches_golden_numbers():
    clock = _clock()
    sampler = IidInverseCdfSampler(build_inverse_cdf_distribution([10.0, 20.0, 30.0]), clock=clock, seed=0)
    values = [sampler.sample({"episode_id": 0}).latency_ms for _ in range(3)]
    assert values == pytest.approx([26.888437030500963, 25.15908805880605, 18.4114316166169])


def test_semi_markov_v1_reproduces_run_length_states():
    clock = _clock()
    burst_samples = [
        {"episode_id": 0, "sample_index": index, "latency_ms": float(value), "latency_raw_frames": index + 1}
        for index, value in enumerate([10.0, 20.0, 30.0])
    ]
    model = build_latency_burst_model(burst_samples)
    sampler = SemiMarkovSampler(model, clock=clock, seed=1)
    samples = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.latency_raw_frames for s in samples] == [1, 2, 3]
    assert [s.latency_ms for s in samples] == [10.0, 20.0, 30.0]


def test_queue_conditioned_v2_conditions_on_recent_drop_count():
    clock = EnvClock(env_fps=25.0, obs_fps=25.0)
    model = {
        "schema_version": 2,
        "model_type": "queue_conditioned",
        "state_key": "recent_drop_count_capped",
        "recent_drop_count_cap": 1,
        "samples_by_state": {
            "0": [{"action_ready_raw_frames": 1, "worker_service_raw_frames": 2, "worker_service_ms": 20.0}],
            "1": [{"action_ready_raw_frames": 4, "worker_service_raw_frames": 5, "worker_service_ms": 110.0}],
        },
    }
    sampler = QueueConditionedSampler(model, clock=clock, seed=0)
    clean = sampler.sample({"episode_id": 0, "recent_drop_count": 0})
    after_drop = sampler.sample({"episode_id": 0, "recent_drop_count": 1})
    assert (clean.action_ready_raw_frames, clean.worker_service_raw_frames, clean.worker_service_ms) == (1, 2, 20.0)
    assert (after_drop.action_ready_raw_frames, after_drop.worker_service_raw_frames, after_drop.worker_service_ms) == (4, 5, 110.0)


def test_fixed_sampler_returns_constant():
    clock = _clock()
    sampler = FixedSampler(42.5, clock=clock)
    assert sampler.sample().latency_ms == 42.5
    assert sampler.latency_type == "fixed"
