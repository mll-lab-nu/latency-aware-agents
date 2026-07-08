from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    build_burst_samples,
    build_hidden_regime_burst_model,
)


def _model_sample(episode_id: int, sample_index: int, action_ready_ms: float, worker_service_ms: float) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "action_ready_ms": action_ready_ms,
        "worker_service_ms": worker_service_ms,
        "worker_service_ms_bin": int(worker_service_ms // 1.0),
    }


def _episode(episode_id: int, worker_service_values: list[float], start_index: int = 0) -> list[dict]:
    return [
        _model_sample(episode_id, start_index + offset, value + 1.0, value)
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
        "latency": {"latency_ms": latency_ms},
        "metadata": metadata,
    }


def test_hidden_regime_model_labels_first_to_last_spike_as_one_burst():
    # Median 42 -> spike threshold 52.5. The real burst signature: spikes with
    # normal-valued samples interleaved (116, 42, 42, 116, 42, 42, 116) must be
    # ONE burst covering the interleaved normals, not three isolated spikes.
    values = [42.0] * 10 + [116.0, 42.0, 42.0, 116.0, 42.0, 42.0, 116.0] + [42.0] * 10
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["schema_version"] == 6
    assert model["model_type"] == "hidden_regime"
    assert model["reset_scope"] == "episode"
    assert model["spike_threshold_ms"] == pytest.approx(52.5)
    assert model["burst_dwell_lengths"] == [7]
    assert model["regime_step_counts"] == {"calm": 20, "burst": 7}
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 18, "burst": 1},
        "burst": {"calm": 1, "burst": 6},
    }
    burst_emission = model["emission_distributions_by_regime"]["burst"]
    assert burst_emission["count"] == 7
    # Burst emission is a mixture of the spikes and the interleaved normals.
    assert burst_emission["worker_service_ms"]["latency_ms"][0] == pytest.approx(42.0)
    assert burst_emission["worker_service_ms"]["latency_ms"][-1] == pytest.approx(116.0)
    calm_emission = model["emission_distributions_by_regime"]["calm"]
    assert calm_emission["count"] == 20
    assert calm_emission["worker_service_ms"]["latency_ms"][-1] == pytest.approx(42.0)


def test_hidden_regime_model_isolated_spike_is_a_length_one_burst():
    values = [42.0] * 20 + [70.0] + [42.0] * 20
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["burst_dwell_lengths"] == [1]
    assert model["regime_step_counts"] == {"calm": 40, "burst": 1}
    assert model["regime_transition_counts"]["calm"] == {"calm": 38, "burst": 1}
    assert model["regime_transition_counts"]["burst"] == {"calm": 1, "burst": 0}


def test_hidden_regime_model_splits_spikes_beyond_merge_gap():
    # Two spikes 40 records apart (> BURST_MERGE_GAP_RECORDS=30) are separate
    # bursts; the samples between them stay calm.
    values = [42.0] * 5 + [70.0] + [42.0] * 39 + [70.0] + [42.0] * 5
    model = build_hidden_regime_burst_model(_episode(0, values))

    assert model["burst_dwell_lengths"] == [1, 1]
    assert model["regime_step_counts"]["burst"] == 2


def test_hidden_regime_model_does_not_bridge_bursts_across_episodes():
    spiky_tail = [42.0] * 5 + [70.0]
    spiky_head = [70.0] + [42.0] * 5
    model = build_hidden_regime_burst_model(
        _episode(0, spiky_tail) + _episode(1, spiky_head, start_index=6)
    )

    # One length-1 burst per episode, no burst->burst transition across the boundary.
    assert model["burst_dwell_lengths"] == [1, 1]
    assert model["regime_transition_counts"]["burst"] == {"calm": 1, "burst": 0}


def test_hidden_regime_model_clean_trace_has_no_burst():
    model = build_hidden_regime_burst_model(_episode(0, [42.0, 42.4, 41.8, 42.1]))

    assert model["burst_dwell_lengths"] == []
    assert model["regime_step_counts"] == {"calm": 4, "burst": 0}
    assert model["regime_transition_counts"] == {
        "calm": {"calm": 3, "burst": 0},
        "burst": {"calm": 0, "burst": 0},
    }
    assert model["emission_distributions_by_regime"]["burst"] is None
    assert model["emission_distributions_by_regime"]["calm"]["count"] == 4


def test_build_burst_samples_uses_all_completed_worker_service_records():
    action_records = [
        _action_record(action_id=10, obs_env_step=4, ready_raw_frame=6, latency_ms=25.0, applied=True, dropped=False),
        _action_record(action_id=11, obs_env_step=8, ready_raw_frame=11, latency_ms=70.0, applied=False, dropped=False),
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


def test_build_burst_samples_rejects_metric_action_count_mismatch():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
    ]

    with pytest.raises(ValueError, match="episode metric num_actions sum"):
        build_burst_samples(action_records, [{"episode_id": 0, "num_actions": 2, "metadata": {"frame_ms": 40.0}}])
