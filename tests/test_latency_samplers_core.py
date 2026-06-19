from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.burst_profile import build_bayes_markov_burst_model
from latency_bench.latency.profile import build_inverse_cdf_distribution
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import (
    BayesMarkovSampler,
    FixedSampler,
    IidInverseCdfSampler,
)


def _model_sample(episode_id: int, sample_index: int, worker_service_raw_frames: int, action_ready_raw_frames: int) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "action_ready_raw_frames": action_ready_raw_frames,
        "worker_service_ms": float(worker_service_raw_frames) * 40.0,
        "worker_service_raw_frames": worker_service_raw_frames,
    }


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


def test_bayes_markov_walks_worker_service_states_in_episode_order():
    clock = _clock()
    # Single deterministic chain 1 -> 2 -> 3 with one emission record per state.
    samples = [
        _model_sample(0, 0, 1, action_ready_raw_frames=1),
        _model_sample(0, 1, 2, action_ready_raw_frames=2),
        _model_sample(0, 2, 3, action_ready_raw_frames=4),
    ]
    model = build_bayes_markov_burst_model(samples)
    sampler = BayesMarkovSampler(
        model, clock=clock, seed=1, transition_alpha=0.0, initial_alpha=0.0, emission_bootstrap=False
    )
    drawn = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.worker_service_raw_frames for s in drawn] == [1, 2, 3]
    assert [s.action_ready_raw_frames for s in drawn] == [1, 2, 4]


def test_bayes_markov_ignores_recent_drop_count_context():
    clock = _clock()
    samples = [_model_sample(0, 0, 2, action_ready_raw_frames=2)]
    model = build_bayes_markov_burst_model(samples)
    sampler = BayesMarkovSampler(model, clock=clock, seed=0)
    # No recent_drop_count key in the context; must still produce a sample.
    sample = sampler.sample({"episode_id": 0})
    assert sample.worker_service_raw_frames == 2


def test_fixed_sampler_returns_constant():
    clock = _clock()
    sampler = FixedSampler(42.5, clock=clock)
    assert sampler.sample().latency_ms == 42.5
    assert sampler.latency_type == "fixed"
