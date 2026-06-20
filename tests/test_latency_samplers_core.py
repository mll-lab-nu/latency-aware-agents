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


def _model_sample(episode_id: int, sample_index: int, e2e_latency_ms: float, worker_service_ms: float) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "e2e_latency_ms": e2e_latency_ms,
        "worker_service_ms": worker_service_ms,
        "e2e_latency_ms_bin": int(e2e_latency_ms // 1.0),
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


def test_bayes_markov_walks_e2e_ms_bin_states_in_episode_order():
    clock = _clock()
    # Single deterministic chain 40 -> 80 -> 160 with one fitted distribution per state.
    samples = [
        _model_sample(0, 0, 40.0, 40.0),
        _model_sample(0, 1, 80.0, 120.0),
        _model_sample(0, 2, 160.0, 80.0),
    ]
    model = build_bayes_markov_burst_model(samples)
    sampler = BayesMarkovSampler(
        model, clock=clock, seed=1, transition_alpha=0.0, initial_alpha=0.0, emission_bootstrap=False
    )
    drawn = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.latency_ms for s in drawn] == pytest.approx([40.0, 80.0, 160.0])
    assert [s.action_ready_raw_frames for s in drawn] == [1, 2, 4]
    assert [s.worker_service_raw_frames for s in drawn] == [1, 3, 2]


def test_bayes_markov_ignores_recent_drop_count_context():
    clock = _clock()
    samples = [_model_sample(0, 0, 80.0, 120.0)]
    model = build_bayes_markov_burst_model(samples)
    sampler = BayesMarkovSampler(model, clock=clock, seed=0)
    # No recent_drop_count key in the context; must still produce a sample.
    sample = sampler.sample({"episode_id": 0})
    assert sample.action_ready_raw_frames == 2
    assert sample.worker_service_raw_frames == 3


def test_bayes_markov_defaults_to_small_transition_smoothing():
    sampler = BayesMarkovSampler(
        build_bayes_markov_burst_model([_model_sample(0, 0, 80.0, 120.0)]),
        clock=_clock(),
        seed=0,
    )

    assert sampler.transition_alpha == pytest.approx(0.01)
    assert sampler.initial_alpha == pytest.approx(0.01)


def test_bayes_markov_terminal_transition_returns_to_initial_distribution():
    clock = EnvClock(env_fps=100.0, obs_fps=100.0)
    model = build_bayes_markov_burst_model(
        [
            _model_sample(0, 0, 20.0, 20.0),
            _model_sample(0, 1, 310.0, 310.0),
        ]
    )
    sampler = BayesMarkovSampler(
        model,
        clock=clock,
        seed=0,
        transition_alpha=0.0,
        initial_alpha=0.0,
        emission_bootstrap=False,
    )

    drawn = [sampler.sample({"episode_id": 0}).worker_service_raw_frames for _ in range(6)]

    assert drawn == [2, 31, 2, 31, 2, 31]


def test_bayes_markov_fitted_emission_interpolates_within_state():
    clock = EnvClock(env_fps=1000.0, obs_fps=1000.0)
    model = build_bayes_markov_burst_model(
        [
            _model_sample(0, 0, 40.0, 50.0),
            _model_sample(0, 1, 40.9, 50.9),
            _model_sample(0, 2, 40.4, 50.4),
            _model_sample(0, 3, 40.7, 50.7),
        ]
    )
    sampler = BayesMarkovSampler(
        model, clock=clock, seed=0, transition_alpha=0.0, initial_alpha=0.0
    )

    values = [sampler.sample({"episode_id": 0}).latency_ms for _ in range(20)]

    assert any(value not in {40.0, 40.4, 40.7, 40.9} for value in values)


def test_fixed_sampler_returns_constant():
    clock = _clock()
    sampler = FixedSampler(42.5, clock=clock)
    assert sampler.sample().latency_ms == 42.5
    assert sampler.latency_type == "fixed"
