from __future__ import annotations

import pytest
from functools import partial
from copy import deepcopy
from statistics import mean

from latency_bench.core.clock import EnvClock
from latency_bench.latency.samplers import JointBurstSampler

from latency_bench.latency.burst_profile import (
    _fit_inverse_cdf,
    build_burst_samples,
    build_hidden_regime_burst_model,
    build_hidden_regime_burst_model_from_sessions,
    build_worker_slot_distributions_from_sessions,
)


# These fixtures retain the v12 writer/sampler contract used by historical profiles.
build_hidden_regime_burst_model = partial(build_hidden_regime_burst_model, model_version=12)
build_hidden_regime_burst_model_from_sessions = partial(build_hidden_regime_burst_model_from_sessions, model_version=12)


def _model_sample(
    episode_id: int,
    sample_index: int,
    observation_to_action_latency_ms: float,
    worker_service_time_ms: float,
    worker_id: int = 0,
) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "worker_id": worker_id,
        "observation_to_action_latency_ms": observation_to_action_latency_ms,
        "worker_service_time_ms": worker_service_time_ms,
        "worker_service_time_ms_bin": int(worker_service_time_ms // 1.0),
    }


@pytest.mark.parametrize("levels", [(60.0, 100.0), (100.0, 60.0), (80.0, 80.0)])
def test_joint_bursts_preserve_request_weighted_levels(levels):
    """Long-lived regression: duration/level coupling survives fit and sampling."""
    short, long = levels
    values = [30.0] * 40 + [short] + [30.0] * 40 + [long] * 9 + [30.0] * 40
    samples = _episode(0, values)
    model = build_hidden_regime_burst_model(samples, model_version=13)
    distribution = build_worker_slot_distributions_from_sessions(
        [samples], spike_threshold_ms_by_worker_slot=model["spike_threshold_ms_by_worker_slot"],
    )
    sampler = JointBurstSampler(model, distribution, clock=EnvClock(env_fps=25, obs_fps=25), seed=20261001)
    slow = []
    for i in range(100000):
        value = sampler.sample({"episode_id": i // 250, "worker_slot": 0}).worker_service_ms
        if value > 37.5:
            slow.append(value)
    assert mean(slow) == pytest.approx((short + 9 * long) / 10, abs=2.0)


def test_joint_burst_initial_state_and_episode_continuation():
    """Initial bursts belong to sessions; resetting an episode must not erase them."""
    sessions = [_episode(0, [120.0] * 2 + [40.0] * 40), [], _episode(0, [40.0] * 42)]
    model = build_hidden_regime_burst_model_from_sessions(sessions, model_version=13)
    distribution = build_worker_slot_distributions_from_sessions(
        sessions, spike_threshold_ms_by_worker_slot=model["spike_threshold_ms_by_worker_slot"],
    )
    initial = []
    for seed in range(1000):
        sampler = JointBurstSampler(model, distribution, clock=EnvClock(env_fps=25, obs_fps=25), seed=seed)
        first = sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_ms
        initial.append(first)
        cloned = deepcopy(sampler)
        context = {"episode_id": 1, "worker_slot": 0}
        assert sampler.sample(context) == cloned.sample(context)
        assert cloned.sample(context).worker_service_ms == 40.0
        for _ in range(20):
            assert sampler.sample(context).worker_service_ms == 40.0
    assert 0.45 < initial.count(120.0) / len(initial) < 0.55


def _episode(
    episode_id: int,
    worker_service_values: list[float],
    start_index: int = 0,
    worker_ids: list[int] | None = None,
) -> list[dict]:
    return [
        _model_sample(
            episode_id,
            start_index + offset,
            value + 1.0,
            value,
            worker_id=0 if worker_ids is None else worker_ids[offset],
        )
        for offset, value in enumerate(worker_service_values)
    ]


def _action_record(
    *,
    action_id: int,
    obs_env_step: int,
    ready_raw_frame: int,
    latency_ms: float,
    applied: bool,
    dropped: bool,
    episode_id: int | None = None,
    worker_id: int = 0,
) -> dict:
    metadata: dict = {
        "frame_ms": 40.0,
        "realtime_profile": {
            "env_observe_ms": 1.0,
            "env_to_parent_ipc_ms": 2.0,
            "parent_observation_handle_ms": 3.0,
            "parent_to_worker_ipc_ms": 0.0,
            "worker_recv_overhead_ms": 0.0,
            "worker_to_parent_ipc_ms": 0.0,
            "parent_release_ms": 4.0,
            "parent_to_env_ipc_ms": 5.0,
            "env_action_recv_to_enqueue_ms": 6.0,
            "env_scheduler_update_ms": 7.0,
        }
    }
    if episode_id is not None:
        metadata["episode_id"] = episode_id
    return {
        "action_id": action_id,
        "obs_env_step": obs_env_step,
        "ready_raw_frame": ready_raw_frame,
        "ready_time_ms": float(ready_raw_frame) * 40.0,
        "applied": applied,
        "dropped": dropped,
        "latency": {"latency_ms": latency_ms, "metadata": {"worker_id": worker_id}},
        "metadata": metadata,
    }


def test_hidden_regime_model_labels_first_to_last_spike_as_one_burst():
    # Median 42 -> spike threshold 52.5. The real burst signature: spikes with
    # normal-valued samples interleaved (116, 42, 42, 116, 42, 42, 116) must be
    # ONE burst covering the interleaved normals, not three isolated spikes.
    values = [42.0] * 10 + [116.0, 42.0, 42.0, 116.0, 42.0, 42.0, 116.0] + [42.0] * 10
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["schema_version"] == 12
    assert model["model_type"] == "hidden_regime"
    assert model["reset_scope"] == "session"
    assert model["worker_count"] == 1
    assert model["spike_threshold_ms_by_worker_slot"]["0"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [7]
    assert model["burst_dwell_distribution"]["latency_ms"][0] == pytest.approx(7.0)
    assert model["burst_dwell_distribution"]["latency_ms"][-1] == pytest.approx(7.0)
    assert model["regime_step_counts"] == {"calm": 20, "burst": 7}
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 18, "burst": 1},
        "burst": {"calm": 1, "burst": 6},
    }
    assert len(model["burst_rank_processes"]) == 1
    process = model["burst_rank_processes"][0]
    assert process["draw_count"] == 7
    assert process["spike_count"] == 3
    # One dwell -> one severity (3 of 7 serves spiked) and one level (116).
    assert process["severity"]["latency_ms"][0] == pytest.approx(3.0 / 7.0)
    assert process["severity"]["latency_ms"][-1] == pytest.approx(3.0 / 7.0)
    assert process["level_ms"]["latency_ms"][0] == pytest.approx(116.0)
    assert process["level_ms"]["latency_ms"][-1] == pytest.approx(116.0)
    # All three spikes sit exactly on the dwell level: residuals are zero.
    assert process["level_residual_ms"]["latency_ms"][0] == pytest.approx(0.0)
    assert process["level_residual_ms"]["latency_ms"][-1] == pytest.approx(0.0)
    assert model["burst_slot_rank_templates"] == [
        {"dwell_length": 7, "slot_to_rank": {"0": 0}}
    ]


def test_hidden_regime_model_isolated_spike_is_a_length_one_burst():
    values = [42.0] * 20 + [70.0] + [42.0] * 20
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["burst_dwell_lengths"] == [1]
    assert model["regime_step_counts"] == {"calm": 40, "burst": 1}
    assert model["regime_transition_counts"]["calm"] == {"calm": 38, "burst": 1}
    assert model["regime_transition_counts"]["burst"] == {"calm": 1, "burst": 0}
    process = model["burst_rank_processes"][0]
    # The burst's only record is the spike itself: severity one and level equal
    # to the spike. Calm emissions live in the per-slot companion distribution.
    assert process["severity"]["latency_ms"][0] == pytest.approx(1.0)
    assert process["severity"]["latency_ms"][-1] == pytest.approx(1.0)
    assert process["level_ms"]["latency_ms"][0] == pytest.approx(70.0)
    assert process["level_residual_ms"]["latency_ms"][-1] == pytest.approx(0.0)


def test_hidden_regime_model_splits_spikes_beyond_merge_gap():
    # Two spikes 40 records apart (> BURST_MERGE_GAP_RECORDS=30) are separate
    # bursts; the samples between them stay calm.
    values = [42.0] * 5 + [70.0] + [42.0] * 39 + [70.0] + [42.0] * 5
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["burst_dwell_lengths"] == [1, 1]
    assert model["regime_step_counts"]["burst"] == 2


def test_hidden_regime_model_bridges_bursts_across_episodes():
    # The regime is machine state: a slow condition straddling an env reset is
    # one burst on the machine timeline, not two per-episode bursts.
    spiky_tail = [42.0] * 5 + [70.0]
    spiky_head = [70.0] + [42.0] * 5
    model = build_hidden_regime_burst_model(
        _episode(0, spiky_tail) + _episode(1, spiky_head, start_index=6)
    )

    assert model["burst_dwell_lengths"] == [2]
    assert model["regime_transition_counts"]["burst"] == {"calm": 1, "burst": 1}


def test_hidden_regime_model_does_not_bridge_measurement_sessions():
    sessions = [
        _episode(0, [42.0] * 10 + [70.0]),
        _episode(0, [70.0] + [42.0] * 10),
    ]

    model = build_hidden_regime_burst_model_from_sessions(sessions)

    assert model["spike_threshold_ms_by_worker_slot"]["0"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [1, 1]
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 18, "burst": 1},
        "burst": {"calm": 1, "burst": 0},
    }
    process = model["burst_rank_processes"][0]
    assert process["draw_count"] == 2
    assert process["spike_count"] == 2
    assert process["severity"]["latency_ms"] == pytest.approx([1.0] * len(process["severity"]["latency_ms"]))


def test_episode_scoped_model_is_not_supported():
    segments = [
        _episode(0, [70.0] + [42.0] * 10),
        _episode(1, [42.0] * 5 + [70.0] + [42.0] * 5),
    ]

    with pytest.raises(TypeError):
        build_hidden_regime_burst_model_from_sessions(segments, reset_scope="episode")


def test_single_session_model_wrapper_matches_session_aware_fitter():
    samples = _episode(0, [42.0] * 10 + [70.0, 42.0, 70.0] + [42.0] * 10)

    assert build_hidden_regime_burst_model(samples) == build_hidden_regime_burst_model_from_sessions([samples])


def test_hidden_regime_model_clean_trace_has_no_burst():
    model = build_hidden_regime_burst_model(_episode(0, [42.0, 42.4, 41.8, 42.1]))

    assert model["burst_dwell_lengths"] == []
    assert model["burst_dwell_distribution"] is None
    assert model["regime_step_counts"] == {"calm": 4, "burst": 0}
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 3, "burst": 0},
        "burst": {"calm": 0, "burst": 0},
    }
    assert "emission_distributions_by_regime" not in model
    assert model["burst_rank_processes"] == []
    assert model["pre_worker_time_ms"] is None


def test_hidden_regime_uses_each_slot_steady_baseline():
    samples = []
    for episode_id in range(4):
        samples.extend(
            _episode(
                episode_id,
                [100.0, 10.0] * 10,
                start_index=episode_id * 20,
                worker_ids=[0, 1] * 10,
            )
        )

    model = build_hidden_regime_burst_model(samples)
    distribution = build_worker_slot_distributions_from_sessions(
        [samples],
        spike_threshold_ms_by_worker_slot=model[
            "spike_threshold_ms_by_worker_slot"
        ],
    )

    assert model["spike_threshold_ms_by_worker_slot"] == {"0": 125.0, "1": 12.5}
    assert model["regime_step_counts"] == {"calm": 80, "burst": 0}
    for slot, expected in (("0", 100.0), ("1", 10.0)):
        steady = distribution["worker_slots"][slot]["steady"]
        assert steady["count"] == 40
        assert set(steady["worker_service_time_ms"]["latency_ms"]) == {expected}


def test_all_burst_slot_falls_back_to_machine_wide_calm_baseline():
    samples = []
    for episode_id in range(4):
        samples.extend(
            _episode(
                episode_id,
                [10.0, 100.0] * 10,
                start_index=episode_id * 20,
                worker_ids=[0, 1] * 10,
            )
        )

    # Worker 0 serves at its calm 10 ms baseline; worker 1 is entirely above
    # the injected threshold and has no per-slot calm baseline. A hidden-regime
    # replay of worker 1's calm regime must fall back to the machine-wide calm
    # pool instead of carrying spike mass.
    distribution = build_worker_slot_distributions_from_sessions(
        [samples],
        spike_threshold_ms_by_worker_slot={"0": 50.0, "1": 50.0},
    )

    assert distribution["worker_slots"]["0"]["steady"]["count"] == 40
    steady_1 = distribution["worker_slots"]["1"]["steady"]
    assert steady_1["count"] == 40
    assert set(steady_1["worker_service_time_ms"]["latency_ms"]) == {10.0}


def test_all_burst_profile_without_any_calm_baseline_rejected():
    samples = _episode(
        0,
        [100.0] * 10,
        worker_ids=[0, 1] * 5,
    )

    with pytest.raises(ValueError, match="calm"):
        build_worker_slot_distributions_from_sessions(
            [samples],
            spike_threshold_ms_by_worker_slot={"0": 50.0, "1": 50.0},
        )


def test_slot_baseline_preserves_repeated_episode_ids_across_sessions():
    healthy = _episode(0, [10.0] * 5)
    degraded = _episode(0, [100.0] * 5)

    model = build_hidden_regime_burst_model_from_sessions([healthy, degraded])

    assert model["spike_threshold_ms_by_worker_slot"] == {"0": 12.5}


def test_hidden_regime_model_fits_per_dwell_severity_and_level():
    # One isolated high spike, then (beyond the merge gap) a dwell mixing a
    # mild 6-draw plateau, two normals, and a 130 spike. Two dwells -> two
    # (severity, level) pairs: (1.0, 150) for the blip and (7/9, 466/7) for
    # the mixed dwell.
    values = (
        [42.0] * 10
        + [150.0]
        + [42.0] * 40
        + [55.0, 56.0, 57.0, 55.0, 56.0, 57.0]
        + [42.0, 42.0]
        + [130.0]
        + [42.0] * 10
    )
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["spike_threshold_ms_by_worker_slot"]["0"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [1, 9]
    process = model["burst_rank_processes"][0]
    assert process["severity"]["latency_ms"][0] == pytest.approx(7.0 / 9.0)
    assert process["severity"]["latency_ms"][-1] == pytest.approx(1.0)
    assert process["level_ms"]["latency_ms"][0] == pytest.approx(466.0 / 7.0)
    assert process["level_ms"]["latency_ms"][-1] == pytest.approx(150.0)
    # Residuals live around each spike's own dwell level: the mixed dwell's
    # 130 sits far above its 466/7 mean, the blip contributes exactly zero.
    assert process["level_residual_ms"]["latency_ms"][0] == pytest.approx(55.0 - 466.0 / 7.0)
    assert process["level_residual_ms"]["latency_ms"][-1] == pytest.approx(130.0 - 466.0 / 7.0)
    # _episode() builds action_ready = worker_service + 1.
    assert model["pre_worker_time_ms"]["latency_ms"][0] == pytest.approx(1.0)
    assert model["pre_worker_time_ms"]["latency_ms"][-1] == pytest.approx(1.0)


def test_hidden_regime_model_bridges_dwells_across_episodes():
    # Episode 0 ends above threshold and episode 1 starts above threshold:
    # one 2-draw dwell on the machine timeline, severity one at level 70.
    model = build_hidden_regime_burst_model(
        _episode(0, [42.0] * 5 + [70.0]) + _episode(1, [70.0] + [42.0] * 5, start_index=6)
    )

    process = model["burst_rank_processes"][0]
    assert process["severity"]["latency_ms"][-1] == pytest.approx(1.0)
    assert process["level_ms"]["latency_ms"][0] == pytest.approx(70.0)


def test_hidden_regime_model_fits_degraded_worker_as_one_constant_excursion():
    # The A100 failure shape: one of two round-robin workers degrades for
    # three consecutive short episodes. The merged stream alternates 80/44, so
    # a merged-stream fit would see fifteen D=1 excursions with one-draw gaps
    # (the interleaving artifact) and a sampler replaying it would spread the
    # slow draws over both workers, so neither saturates. Per-worker fitting
    # recovers the truth: the slow worker's sub-stream is ONE 15-serve
    # excursion at 80ms and the healthy worker never spikes.
    degraded = [80.0, 44.0] * 5
    alternating = [0, 1] * 5
    model = build_hidden_regime_burst_model(
        _episode(0, [42.0] * 10, worker_ids=[0, 1] * 5)
        + _episode(1, degraded, start_index=10, worker_ids=alternating)
        + _episode(2, degraded, start_index=20, worker_ids=alternating)
        + _episode(3, degraded, start_index=30, worker_ids=alternating)
        + _episode(4, [42.0] * 10, start_index=40, worker_ids=[0, 1] * 5)
    )

    # Median 44 -> threshold 55; the merged dwell runs first 80 to last 80.
    assert model["spike_threshold_ms_by_worker_slot"] == {
        "0": pytest.approx(52.5),
        "1": pytest.approx(52.5),
    }
    assert model["worker_count"] == 2
    assert model["burst_dwell_lengths"] == [29]
    assert len(model["burst_rank_processes"]) == 2
    slow = model["burst_rank_processes"][0]
    assert slow["draw_count"] == 15
    assert slow["spike_count"] == 15
    # Constant-slow is severity one at the worker's level, not a run taxonomy.
    assert slow["severity"]["latency_ms"][0] == pytest.approx(1.0)
    assert slow["severity"]["latency_ms"][-1] == pytest.approx(1.0)
    assert slow["level_ms"]["latency_ms"][0] == pytest.approx(80.0)
    assert slow["level_residual_ms"]["latency_ms"][-1] == pytest.approx(0.0)
    healthy = model["burst_rank_processes"][1]
    assert healthy["draw_count"] == 14
    assert healthy["spike_count"] == 0
    # The healthy worker never spiked inside the dwell: severity zero, no
    # level; its calm distribution is fitted separately for slot 1.
    assert healthy["severity"]["latency_ms"][0] == pytest.approx(0.0)
    assert healthy["severity"]["latency_ms"][-1] == pytest.approx(0.0)
    assert healthy["level_ms"] is None
    assert model["burst_slot_rank_templates"] == [
        {"dwell_length": 29, "slot_to_rank": {"0": 0, "1": 1}}
    ]


def test_hidden_regime_model_ranks_workers_by_in_dwell_severity():
    # A host-wide wobble: both workers elevated inside one dwell, one worse.
    # Rank 0 must be the worse worker regardless of worker id.
    values = [42.0] * 10 + [80.0, 80.0, 44.0, 80.0, 80.0, 80.0] + [42.0] * 10
    worker_ids = [0, 1] * 5 + [1, 0, 1, 0, 1, 0] + [0, 1] * 5
    model = build_hidden_regime_burst_model(_episode(0, values, worker_ids=worker_ids))

    # Worker 1 spikes 3/3 of its in-dwell serves, worker 0 spikes 2/3.
    ranks = model["burst_rank_processes"]
    assert ranks[0]["draw_count"] == 3
    assert ranks[0]["spike_count"] == 3
    assert ranks[1]["draw_count"] == 3
    assert ranks[1]["spike_count"] == 2


def test_emission_quantile_grid_resolves_rare_tail():
    # 999/1000 draws take 41ms and one takes 100ms (fast service, one late
    # delivery). A uniform 1% grid would smear a full percent of mass evenly
    # from 41ms to 100ms; the densified tail keeps the bulk pinned at 41ms
    # through p99.8 while the outlier owns its 1/1000 share: a 100ms plateau
    # above the last plotting position (999.5 / 1000).
    samples = [
        _model_sample(0, index, 41.0 if index < 999 else 100.0, 40.0)
        for index in range(1000)
    ]
    model = build_hidden_regime_burst_model(samples)
    distribution = build_worker_slot_distributions_from_sessions(
        [samples], spike_threshold_ms_by_worker_slot={"0": 50.0}
    )
    table = distribution["worker_slots"]["0"]["all"]["observation_to_action_latency_ms"]

    assert table["latency_ms"][table["quantile_levels"].index(0.998)] == pytest.approx(41.0)
    assert table["latency_ms"][table["quantile_levels"].index(0.9995)] == pytest.approx(100.0)
    assert table["latency_ms"][-1] == pytest.approx(100.0)


def _table_mean(table: dict) -> float:
    levels = table["quantile_levels"]
    values = table["latency_ms"]
    return sum(
        (levels[i + 1] - levels[i]) * (values[i] + values[i + 1]) / 2.0
        for i in range(len(levels) - 1)
    )


def test_fit_inverse_cdf_preserves_heavy_tail_mass():
    # The A100 dwell shape: 44 short bursts and one 2631-draw degradation.
    # Each observed sample carries 1/45 of the probability mass, so the
    # table's integral must sit at the sample mean; pinning the maximum on
    # the q=1.0 endpoint instead smears its mass down the 34->2631 ramp and
    # the resampled mean collapses to ~2/3 of the observed one.
    values = [34.0] * 44 + [2631.0]
    table = _fit_inverse_cdf(values)

    sample_mean = sum(values) / len(values)
    assert _table_mean(table) == pytest.approx(sample_mean, rel=0.05)


def test_fit_inverse_cdf_gives_the_maximum_its_own_quantile_mass():
    # Above the last plotting position ((n - 0.5) / n) the inverse CDF is the
    # largest observed sample, so the rarest event keeps a real plateau
    # instead of being reachable only in the limit u -> 1.
    values = [34.0] * 44 + [2631.0]
    table = _fit_inverse_cdf(values)

    for level, value in zip(table["quantile_levels"], table["latency_ms"]):
        if level >= (len(values) - 0.5) / len(values):
            assert value == pytest.approx(2631.0)


def _attempt_record(
    *, obs_id: int, episode_id: int, obs_env_step: int,
    worker_id: int | None, service_ms: float | None, latency_ms: float | None,
    submitted: bool = True,
) -> dict:
    record = {
        "episode_id": episode_id,
        "obs_id": obs_id,
        "obs_env_step": obs_env_step,
        "submitted": submitted,
    }
    if submitted:
        record["worker_id"] = worker_id
        record["worker_service_time_ms"] = service_ms
        record["observation_to_action_latency_ms"] = latency_ms
    return record


def test_build_burst_samples_filters_capacity_drops_and_keeps_attempt_order():
    attempts = [
        _attempt_record(episode_id=3, obs_id=12, obs_env_step=8,
                        worker_id=1, service_ms=70.0, latency_ms=None,
                        submitted=False),
        _attempt_record(episode_id=3, obs_id=10, obs_env_step=4,
                        worker_id=0, service_ms=25.0, latency_ms=53.0),
        _attempt_record(episode_id=3, obs_id=11, obs_env_step=6,
                        worker_id=1, service_ms=30.0, latency_ms=58.0),
    ]

    samples = build_burst_samples(attempts)

    assert [s["obs_id"] for s in samples] == [10, 11]
    assert samples[0]["observation_to_action_latency_ms"] == pytest.approx(53.0)
    assert samples[0]["worker_service_time_ms"] == pytest.approx(25.0)


def test_build_burst_samples_records_measure_session_id():
    attempts = [_attempt_record(
        episode_id=0, obs_id=0, obs_env_step=0,
        worker_id=0, service_ms=10.0, latency_ms=11.0,
    )]

    samples = build_burst_samples(attempts, measure_session_id=4)

    assert samples[0]["measure_session_id"] == 4

def test_hidden_regime_model_fits_negative_dwell_length_severity_rho():
    # A short 100%-severity blip and a long mild dwell: the fitted
    # dwell-length/severity Spearman rho must be negative so sampling does not
    # hand long degradations a blip's severity.
    values = (
        [42.0] * 10
        + [80.0]                       # 1-draw blip, severity 1.0
        + [42.0] * 40
        + [80.0, 44.0, 44.0, 44.0] * 5  # dwell runs first 80 to last 80: 17 draws, 5 spikes
        + [42.0] * 10
    )
    model = build_hidden_regime_burst_model(_episode(0, values))

    process = model["burst_rank_processes"][0]
    assert process["dwell_length_spearman_rho"] < 0.0
    assert process["severity"]["latency_ms"][0] == pytest.approx(5.0 / 17.0)
    assert process["severity"]["latency_ms"][-1] == pytest.approx(1.0)
