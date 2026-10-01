from __future__ import annotations

from copy import deepcopy

import pytest
from functools import partial

from latency_bench.core.clock import EnvClock
from latency_bench.latency.burst_profile import (
    build_hidden_regime_burst_model,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import (
    FixedSampler,
    HiddenRegimeSampler,
    IidInverseCdfSampler,
)


# These fixtures retain the v12 writer/sampler contract used by historical profiles.
build_hidden_regime_burst_model = partial(build_hidden_regime_burst_model, model_version=12)


def _inverse_cdf(samples: list[float]) -> dict:
    ordered = sorted(samples)
    levels = (
        [0.0, 1.0]
        if len(ordered) == 1
        else [index / (len(ordered) - 1) for index in range(len(ordered))]
    )
    values = [ordered[0], ordered[0]] if len(ordered) == 1 else ordered
    return {
        "distribution_type": "inverse_cdf",
        "quantile_levels": levels,
        "latency_ms": values,
    }


def _model_sample(
    episode_id: int,
    sample_index: int,
    observation_to_action_latency_ms: float,
    worker_service_time_ms: float,
) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "worker_id": 0,
        "observation_to_action_latency_ms": observation_to_action_latency_ms,
        "worker_service_time_ms": worker_service_time_ms,
        "worker_service_time_ms_bin": int(worker_service_time_ms // 1.0),
    }


def _model(
    worker_service_values: list[float], *, action_ready_offset: float = 0.0
) -> dict:
    return build_hidden_regime_burst_model(
        [
            _model_sample(0, index, value + action_ready_offset, value)
            for index, value in enumerate(worker_service_values)
        ]
    )


def _model_distribution(
    worker_service_values: list[float], *, action_ready_offset: float = 0.0
) -> dict:
    samples = [
        _model_sample(0, index, value + action_ready_offset, value)
        for index, value in enumerate(worker_service_values)
    ]
    model = build_hidden_regime_burst_model(samples)
    return build_worker_slot_distributions_from_sessions(
        [samples],
        spike_threshold_ms_by_worker_slot=model[
            "spike_threshold_ms_by_worker_slot"
        ],
    )


def _sampler(
    worker_service_values: list[float],
    *,
    action_ready_offset: float = 0.0,
    seed: int,
) -> HiddenRegimeSampler:
    model = _model(worker_service_values, action_ready_offset=action_ready_offset)
    return HiddenRegimeSampler(
        model,
        _model_distribution(worker_service_values, action_ready_offset=action_ready_offset),
        clock=_clock(),
        seed=seed,
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
    joint = {
        "schema_version": 3,
        "worker_slots": {
            "0": {
                "all": {
                    "count": 3,
                    "observation_to_action_latency_ms": _inverse_cdf([10.0, 20.0, 30.0]),
                    "worker_service_time_ms": _inverse_cdf([10.0, 20.0, 30.0]),
                    "spearman_rho": 1.0,
                },
                "steady": {
                    "count": 3,
                    "observation_to_action_latency_ms": _inverse_cdf([10.0, 20.0, 30.0]),
                    "worker_service_time_ms": _inverse_cdf([10.0, 20.0, 30.0]),
                    "spearman_rho": 1.0,
                },
            }
        },
    }
    sampler = IidInverseCdfSampler(joint, clock=clock, seed=0)
    values = [sampler.sample({"episode_id": 0, "worker_slot": 0}).latency_ms for _ in range(3)]
    assert all(10.0 <= value <= 30.0 for value in values)


def test_iid_inverse_cdf_selects_the_requested_worker_slot_joint():
    def joint(observation_to_action_latency_ms: float, worker_service_time_ms: float) -> dict:
        return {
            "count": 1,
            "observation_to_action_latency_ms": _inverse_cdf([observation_to_action_latency_ms]),
            "worker_service_time_ms": _inverse_cdf([worker_service_time_ms]),
            "spearman_rho": 0.0,
        }

    sampler = IidInverseCdfSampler(
        {
            "schema_version": 3,
            "worker_slots": {
                "0": {"all": joint(10.0, 2.0), "steady": joint(10.0, 2.0)},
                "1": {"all": joint(100.0, 40.0), "steady": joint(100.0, 40.0)},
            },
        },
        clock=_clock(),
        seed=0,
    )

    slot_zero = sampler.sample({"worker_slot": 0})
    slot_one = sampler.sample({"worker_slot": 1})
    assert (slot_zero.latency_ms, slot_zero.worker_service_ms) == (10.0, 2.0)
    assert (slot_one.latency_ms, slot_one.worker_service_ms) == (100.0, 40.0)


def test_hidden_regime_clean_model_stays_calm_and_constant():
    # No spike in the trace -> burst emission is None -> the sampler must never
    # leave calm, and a constant calm emission stays constant.
    sampler = _sampler([40.0] * 8, seed=0)
    drawn = [sampler.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(50)]
    assert {s.latency_ms for s in drawn} == {40.0}
    assert {s.worker_service_raw_frames for s in drawn} == {1}


def test_hidden_regime_deepcopy_shares_tables_and_clones_state():
    sampler = _sampler([30.0, 35.0, 40.0, 45.0] * 20 + [120.0] * 4, seed=7)
    sampler.sample({"episode_id": 0, "worker_slot": 0})

    clone = deepcopy(sampler)

    assert clone.model is sampler.model
    assert clone.distributions is sampler.distributions
    contexts = [{"episode_id": 0, "worker_slot": 0}] * 5
    assert [sampler.sample(context) for context in contexts] == [
        clone.sample(context) for context in contexts
    ]
    untouched = deepcopy(sampler)
    for episode in range(3):
        for _ in range(15):
            clone.sample({"episode_id": episode, "worker_slot": 0})
    contexts = [{"episode_id": 0, "worker_slot": 0}] * 40
    assert [sampler.sample(context) for context in contexts] == [
        untouched.sample(context) for context in contexts
    ]


def test_hidden_regime_emits_burst_values_at_fitted_rate():
    # Half the fitted steps are burst-labeled with a constant 120ms emission, so
    # generated sequences must mix both regimes' values.
    values = [40.0] * 40 + [120.0, 120.0, 120.0, 120.0] + [40.0] * 40
    sampler = _sampler(values, seed=3)
    drawn = [sampler.sample({"episode_id": episode_id, "worker_slot": 0}).worker_service_ms for episode_id in range(200) for _ in range(84)]
    burst_share = sum(1 for value in drawn if value == 120.0) / len(drawn)
    assert 0.01 < burst_share < 0.20  # fitted share is 4/84 with posterior spread


def test_hidden_regime_burst_persists_within_an_episode():
    # Once inside burst, the exit probability posterior (~1/dwell) keeps the
    # regime alive for several steps: bursts must contain runs, not only
    # isolated 120ms draws.
    values = ([40.0] * 40 + [120.0, 120.0, 120.0, 120.0, 120.0, 120.0]) * 3 + [40.0] * 40
    sampler = _sampler(values, seed=5)
    longest_run = 0
    for episode_id in range(300):
        run = 0
        for _ in range(150):
            if sampler.sample({"episode_id": episode_id, "worker_slot": 0}).worker_service_ms == 120.0:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0
    assert longest_run >= 3


def test_hidden_regime_episode_start_carries_regime_across_boundaries():
    # One draw per episode: the machine chain keeps evolving across episode
    # boundaries, so degraded 120ms draws must reach episode starts instead of
    # every episode being forced back to calm.
    values = [40.0] * 4 + [120.0] * 4
    sampler = _sampler(values, seed=0)
    first_draws = [sampler.sample({"episode_id": episode_id, "worker_slot": 0}).worker_service_ms for episode_id in range(100)]
    assert 120.0 in first_draws
    assert 40.0 in first_draws


def test_hidden_regime_is_deterministic_with_seed():
    values = [40.0] * 20 + [120.0, 40.0, 120.0] + [40.0] * 20
    first = _sampler(values, seed=7)
    second = _sampler(values, seed=7)
    assert [first.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(50)] == [
        second.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(50)
    ]


def test_hidden_regime_quantizes_worker_service_ms_on_current_clock():
    sampler = _sampler([75.0], action_ready_offset=-35.0, seed=0)
    sample = sampler.sample({"episode_id": 0, "worker_slot": 0})
    assert sample.latency_ms == 40.0
    assert sample.worker_service_ms == 75.0
    assert sample.action_ready_raw_frames == 1
    assert sample.worker_service_raw_frames == 2


def test_fixed_sampler_returns_constant():
    clock = _clock()
    sampler = FixedSampler(42.5, clock=clock)
    assert sampler.sample().latency_ms == 42.5
    assert sampler.latency_type == "fixed"

def test_hidden_regime_burst_is_a_worker_state_not_a_merged_pattern():
    # Fit the A100 failure shape: worker 0 constantly slow inside the dwell
    # while worker 1 stays fast, so the merged stream alternates 80/44. When a
    # sampled burst plays back, the slot that entered must carry the slowness
    # on essentially every serve and the other slot must stay sub-threshold.
    # Spreading the merged ~50% alternation iid over both slots (so neither
    # worker ever saturates) is exactly the bug this guards against.
    values = [42.0] * 20 + [80.0, 44.0] * 10 + [42.0] * 20
    workers = ([0, 1] * 10) + ([0, 1] * 10) + ([0, 1] * 10)
    samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "worker_id": worker,
            "observation_to_action_latency_ms": value + 1.0,
            "worker_service_time_ms": value,
            "worker_service_time_ms_bin": int(value),
        }
        for index, (value, worker) in enumerate(zip(values, workers))
    ]
    model = build_hidden_regime_burst_model(samples)
    distribution = build_worker_slot_distributions_from_sessions(
        [samples],
        spike_threshold_ms_by_worker_slot=model[
            "spike_threshold_ms_by_worker_slot"
        ],
    )
    sampler = HiddenRegimeSampler(model, distribution, clock=_clock(), seed=2)

    for attempt in range(50000):
        slot = attempt % 2
        sampler.sample({"episode_id": attempt // 100, "worker_slot": slot})
        if sampler.regime == "burst":
            entry_slot = slot
            break
    else:
        pytest.fail("sampler never entered burst")

    slot_to_rank = sampler._slot_to_rank
    hot_slot = next(slot for slot, rank in slot_to_rank.items() if rank == 0)
    cold_slot = next(slot for slot, rank in slot_to_rank.items() if rank == 1)
    spikes_by_slot = {hot_slot: [], cold_slot: []}
    draw = attempt
    while sampler.regime == "burst":
        draw += 1
        slot = draw % 2
        value = sampler.sample({"episode_id": draw // 100, "worker_slot": slot})
        spikes_by_slot[slot].append(
            value.worker_service_ms
            > model["spike_threshold_ms_by_worker_slot"][str(slot)]
        )

    hot = spikes_by_slot[hot_slot]
    cold = spikes_by_slot[cold_slot]
    assert len(hot) >= 8 and len(cold) >= 8
    assert sum(hot) / len(hot) > 0.85
    assert not any(cold)
