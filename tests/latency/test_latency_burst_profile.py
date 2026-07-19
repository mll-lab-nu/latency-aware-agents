from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    _fit_inverse_cdf,
    build_burst_samples,
    build_hidden_regime_burst_model,
    build_hidden_regime_burst_model_from_sessions,
)


def _model_sample(
    episode_id: int,
    sample_index: int,
    action_ready_ms: float,
    worker_service_ms: float,
    worker_id: int = 0,
) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "worker_id": worker_id,
        "action_ready_ms": action_ready_ms,
        "worker_service_ms": worker_service_ms,
        "worker_service_ms_bin": int(worker_service_ms // 1.0),
    }


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

    assert model["schema_version"] == 9
    assert model["model_type"] == "hidden_regime"
    assert model["reset_scope"] == "session"
    assert model["worker_count"] == 1
    assert model["spike_threshold_ms"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [7]
    assert model["burst_dwell_distribution"]["latency_ms"][0] == pytest.approx(7.0)
    assert model["burst_dwell_distribution"]["latency_ms"][-1] == pytest.approx(7.0)
    assert model["regime_step_counts"] == {"calm": 20, "burst": 7}
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 18, "burst": 1},
        "burst": {"calm": 1, "burst": 6},
    }
    calm_emission = model["emission_distributions_by_regime"]["calm"]
    assert calm_emission["count"] == 20
    assert calm_emission["worker_service_ms"]["latency_ms"][-1] == pytest.approx(42.0)
    assert len(model["burst_rank_processes"]) == 1
    process = model["burst_rank_processes"][0]
    assert process["draw_count"] == 7
    assert process["spike_count"] == 3
    # Above-threshold records carry the dwell's (severity, level); the rank
    # emission keeps only the interleaved normals.
    assert process["emission"]["count"] == 4
    assert process["emission"]["worker_service_ms"]["latency_ms"][0] == pytest.approx(42.0)
    # One dwell -> one severity (3 of 7 serves spiked) and one level (116).
    assert process["severity"]["latency_ms"][0] == pytest.approx(3.0 / 7.0)
    assert process["severity"]["latency_ms"][-1] == pytest.approx(3.0 / 7.0)
    assert process["level_ms"]["latency_ms"][0] == pytest.approx(116.0)
    assert process["level_ms"]["latency_ms"][-1] == pytest.approx(116.0)
    # All three spikes sit exactly on the dwell level: residuals are zero.
    assert process["level_residual_ms"]["latency_ms"][0] == pytest.approx(0.0)
    assert process["level_residual_ms"]["latency_ms"][-1] == pytest.approx(0.0)


def test_hidden_regime_model_isolated_spike_is_a_length_one_burst():
    values = [42.0] * 20 + [70.0] + [42.0] * 20
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["burst_dwell_lengths"] == [1]
    assert model["regime_step_counts"] == {"calm": 40, "burst": 1}
    assert model["regime_transition_counts"]["calm"] == {"calm": 38, "burst": 1}
    assert model["regime_transition_counts"]["burst"] == {"calm": 1, "burst": 0}
    process = model["burst_rank_processes"][0]
    # The burst's only record is the spike itself: severity one, level equal
    # to the spike, and the rank emission has no records left.
    assert process["emission"] is None
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

    assert model["spike_threshold_ms"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [1, 1]
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 18, "burst": 1},
        "burst": {"calm": 1, "burst": 0},
    }
    process = model["burst_rank_processes"][0]
    assert process["draw_count"] == 2
    assert process["spike_count"] == 2
    assert process["severity"]["latency_ms"] == pytest.approx([1.0] * len(process["severity"]["latency_ms"]))


def test_episode_scoped_model_fits_initial_burst_separately():
    segments = [
        _episode(0, [70.0] + [42.0] * 10),
        _episode(1, [42.0] * 5 + [70.0] + [42.0] * 5),
    ]

    model = build_hidden_regime_burst_model_from_sessions(
        segments, reset_scope="episode"
    )

    assert model["schema_version"] == 10
    assert model["reset_scope"] == "episode"
    assert model["initial_regime_counts"] == {"calm": 1, "burst": 1}
    assert model["initial_burst_residual_dwell_distribution"]["latency_ms"] == pytest.approx(
        [1.0] * len(model["initial_burst_residual_dwell_distribution"]["latency_ms"])
    )
    assert model["burst_dwell_lengths"] == [1]
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 17, "burst": 1},
        "burst": {"calm": 2, "burst": 0},
    }


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
    assert model["emission_distributions_by_regime"]["calm"]["count"] == 4
    assert model["burst_rank_processes"] == []
    assert model["ready_delta_ms"] is None


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

    assert model["spike_threshold_ms"] == pytest.approx(52.5)
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
    assert model["ready_delta_ms"]["latency_ms"][0] == pytest.approx(1.0)
    assert model["ready_delta_ms"]["latency_ms"][-1] == pytest.approx(1.0)
    # The rank emission keeps only the two interleaved normals.
    assert process["emission"]["count"] == 2
    assert process["emission"]["worker_service_ms"]["latency_ms"][-1] == pytest.approx(42.0)


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
    assert model["spike_threshold_ms"] == pytest.approx(55.0)
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
    # level, and its serves all live in the sub-threshold emission.
    assert healthy["severity"]["latency_ms"][0] == pytest.approx(0.0)
    assert healthy["severity"]["latency_ms"][-1] == pytest.approx(0.0)
    assert healthy["level_ms"] is None
    assert healthy["emission"]["count"] == 14
    assert healthy["emission"]["worker_service_ms"]["latency_ms"][0] == pytest.approx(44.0)


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
    table = model["emission_distributions_by_regime"]["calm"]["action_ready_ms"]

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


def test_build_burst_samples_uses_all_completed_worker_service_records():
    action_records = [
        _action_record(action_id=10, obs_env_step=4, ready_raw_frame=6, latency_ms=25.0, applied=True, dropped=False),
        _action_record(action_id=11, obs_env_step=8, ready_raw_frame=11, latency_ms=70.0, applied=False, dropped=False, worker_id=1),
    ]
    metrics = [{"episode_id": 3, "num_actions": 2, "metadata": {"frame_ms": 40.0}}]

    samples = build_burst_samples(action_records, metrics)

    assert samples == [
        {
            "episode_id": 3,
            "sample_index": 0,
            "action_id": 10,
            "obs_env_step": 4,
            "ready_raw_frame": 6,
            "applied": True,
            "dropped": False,
            "worker_id": 0,
            "action_ready_ms": 80.0,
            "worker_service_ms_bin": 25,
            "worker_service_ms": 25.0,
        },
        {
            "episode_id": 3,
            "sample_index": 1,
            "action_id": 11,
            "obs_env_step": 8,
            "ready_raw_frame": 11,
            "applied": False,
            "dropped": False,
            "worker_id": 1,
            "action_ready_ms": 120.0,
            "worker_service_ms_bin": 70,
            "worker_service_ms": 70.0,
        },
    ]


def test_build_burst_samples_assigns_episodes_from_metrics():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
        _action_record(action_id=1, obs_env_step=1, ready_raw_frame=2, latency_ms=11.0, applied=True, dropped=False),
        _action_record(action_id=2, obs_env_step=0, ready_raw_frame=3, latency_ms=30.0, applied=True, dropped=False),
    ]
    metrics = [
        {"episode_id": 7, "num_actions": 2, "metadata": {"frame_ms": 40.0}},
        {"episode_id": 8, "num_actions": 1, "metadata": {"frame_ms": 40.0}},
    ]

    samples = build_burst_samples(action_records, metrics)

    assert [(s["episode_id"], s["action_id"]) for s in samples] == [(7, 0), (7, 1), (8, 2)]


def test_build_burst_samples_records_measure_session_id():
    action_records = [
        _action_record(
            action_id=0,
            obs_env_step=0,
            ready_raw_frame=1,
            latency_ms=10.0,
            applied=True,
            dropped=False,
        ),
    ]

    samples = build_burst_samples(
        action_records,
        [{"episode_id": 0, "num_actions": 1, "metadata": {"frame_ms": 40.0}}],
        measure_session_id=4,
    )

    assert samples[0]["measure_session_id"] == 4


def test_build_burst_samples_rejects_metric_action_count_mismatch():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
    ]

    with pytest.raises(ValueError, match="episode metric num_actions sum"):
        build_burst_samples(action_records, [{"episode_id": 0, "num_actions": 2, "metadata": {"frame_ms": 40.0}}])

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
