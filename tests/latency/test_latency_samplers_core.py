from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.burst_profile import build_hidden_regime_burst_model
from latency_bench.latency.profile import build_inverse_cdf_distribution
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import (
    FixedSampler,
    HiddenRegimeSampler,
    IidInverseCdfSampler,
)


def _model_sample(episode_id: int, sample_index: int, action_ready_ms: float, worker_service_ms: float) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "action_ready_ms": action_ready_ms,
        "worker_service_ms": worker_service_ms,
        "worker_service_ms_bin": int(worker_service_ms // 1.0),
    }


def _model(worker_service_values: list[float], *, action_ready_offset: float = 0.0) -> dict:
    return build_hidden_regime_burst_model(
        [
            _model_sample(0, index, value + action_ready_offset, value)
            for index, value in enumerate(worker_service_values)
        ]
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


def test_hidden_regime_clean_model_stays_calm_and_constant():
    # No spike in the trace -> burst emission is None -> the sampler must never
    # leave calm, and a constant calm emission stays constant.
    sampler = HiddenRegimeSampler(_model([40.0] * 8), clock=_clock(), seed=0)
    drawn = [sampler.sample({"episode_id": 0}) for _ in range(50)]
    assert {s.latency_ms for s in drawn} == {40.0}
    assert {s.worker_service_raw_frames for s in drawn} == {1}


def test_hidden_regime_emits_burst_values_at_fitted_rate():
    # Half the fitted steps are burst-labeled with a constant 120ms emission, so
    # generated sequences must mix both regimes' values.
    values = [40.0] * 40 + [120.0, 120.0, 120.0, 120.0] + [40.0] * 40
    sampler = HiddenRegimeSampler(_model(values), clock=_clock(), seed=3)
    drawn = [sampler.sample({"episode_id": episode_id}).worker_service_ms for episode_id in range(200) for _ in range(84)]
    burst_share = sum(1 for value in drawn if value == 120.0) / len(drawn)
    assert 0.01 < burst_share < 0.20  # fitted share is 4/84 with posterior spread


def test_hidden_regime_burst_persists_within_an_episode():
    # Once inside burst, the exit probability posterior (~1/dwell) keeps the
    # regime alive for several steps: bursts must contain runs, not only
    # isolated 120ms draws.
    values = ([40.0] * 40 + [120.0, 120.0, 120.0, 120.0, 120.0, 120.0]) * 3 + [40.0] * 40
    sampler = HiddenRegimeSampler(_model(values), clock=_clock(), seed=5)
    longest_run = 0
    for episode_id in range(300):
        run = 0
        for _ in range(150):
            if sampler.sample({"episode_id": episode_id}).worker_service_ms == 120.0:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0
    assert longest_run >= 3


def test_hidden_regime_episode_start_resets_to_calm():
    values = [40.0] * 4 + [120.0] * 4
    sampler = HiddenRegimeSampler(_model(values), clock=_clock(), seed=0)
    first_draws = [sampler.sample({"episode_id": episode_id}).worker_service_ms for episode_id in range(100)]
    assert first_draws == pytest.approx([40.0] * 100)


def test_hidden_regime_is_deterministic_with_seed():
    values = [40.0] * 20 + [120.0, 40.0, 120.0] + [40.0] * 20
    first = HiddenRegimeSampler(_model(values), clock=_clock(), seed=7)
    second = HiddenRegimeSampler(_model(values), clock=_clock(), seed=7)
    assert [first.sample({"episode_id": 0}) for _ in range(50)] == [
        second.sample({"episode_id": 0}) for _ in range(50)
    ]


def test_hidden_regime_quantizes_worker_service_ms_on_current_clock():
    sampler = HiddenRegimeSampler(_model([75.0], action_ready_offset=-35.0), clock=_clock(), seed=0)
    sample = sampler.sample({"episode_id": 0})
    assert sample.latency_ms == 40.0
    assert sample.worker_service_ms == 75.0
    assert sample.action_ready_raw_frames == 1
    assert sample.worker_service_raw_frames == 2


def test_fixed_sampler_returns_constant():
    clock = _clock()
    sampler = FixedSampler(42.5, clock=clock)
    assert sampler.sample().latency_ms == 42.5
    assert sampler.latency_type == "fixed"
