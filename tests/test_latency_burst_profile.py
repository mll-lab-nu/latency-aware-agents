from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    build_bayes_markov_burst_model,
    build_bayes_markov_burst_samples,
)


def _model_sample(episode_id: int, sample_index: int, worker_service_raw_frames: int) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "action_ready_raw_frames": worker_service_raw_frames,
        "worker_service_ms": float(worker_service_raw_frames) * 40.0,
        "worker_service_raw_frames": worker_service_raw_frames,
    }


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
        "realtime_profile": {
            "parent_to_worker_ipc_ms": 0.0,
            "worker_recv_overhead_ms": 0.0,
            "worker_to_parent_ipc_ms": 0.0,
        }
    }
    if episode_id is not None:
        metadata["episode_id"] = episode_id
    return {
        "action_id": action_id,
        "obs_env_step": obs_env_step,
        "ready_raw_frame": ready_raw_frame,
        "applied": applied,
        "dropped": dropped,
        "latency": {"latency_ms": latency_ms},
        "metadata": metadata,
    }


def test_build_bayes_markov_burst_model_uses_per_submitted_obs_transitions():
    # Single episode worker_service_raw_frames sequence: [1, 1, 2, 3, 3]
    samples = [
        _model_sample(0, 0, 1),
        _model_sample(0, 1, 1),
        _model_sample(0, 2, 2),
        _model_sample(0, 3, 3),
        _model_sample(0, 4, 3),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["schema_version"] == 3
    assert model["model_type"] == "bayes_markov"
    assert model["reset_scope"] == "episode"
    assert model["state_key"] == "worker_service_raw_frames"
    assert model["initial_state_counts"] == {"1": 1}
    # Per-adjacent-pair transitions (NOT per-run): (1,1)(1,2)(2,3)(3,3).
    assert model["transition_counts"] == {
        "1": {"1": 1, "2": 1},
        "2": {"3": 1},
        "3": {"3": 1},
    }
    assert model["run_lengths_by_state"] == {"1": [2], "2": [1], "3": [2]}
    assert model["state_counts"] == {"1": 2, "2": 1, "3": 2}
    assert model["emissions_by_state"]["1"] == [
        {"action_ready_raw_frames": 1, "worker_service_raw_frames": 1, "worker_service_ms": 40.0},
        {"action_ready_raw_frames": 1, "worker_service_raw_frames": 1, "worker_service_ms": 40.0},
    ]
    assert model["emissions_by_state"]["3"][0]["worker_service_raw_frames"] == 3


def test_build_bayes_markov_burst_model_counts_initial_state_per_episode():
    samples = [
        _model_sample(0, 0, 1),
        _model_sample(0, 1, 2),
        _model_sample(1, 2, 2),
        _model_sample(1, 3, 1),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["initial_state_counts"] == {"1": 1, "2": 1}
    assert model["transition_counts"] == {"1": {"2": 1}, "2": {"1": 1}}


def test_build_bayes_markov_burst_samples_includes_applied_and_dropped_records():
    action_records = [
        _action_record(action_id=10, obs_env_step=4, ready_raw_frame=6, latency_ms=25.0, applied=True, dropped=False),
        _action_record(action_id=11, obs_env_step=8, ready_raw_frame=11, latency_ms=70.0, applied=False, dropped=True),
    ]
    metrics = [{"episode_id": 3, "num_actions": 2, "metadata": {"frame_ms": 40.0}}]

    samples = build_bayes_markov_burst_samples(action_records, metrics)

    assert samples == [
        {
            "episode_id": 3,
            "sample_index": 0,
            "action_id": 10,
            "obs_env_step": 4,
            "ready_raw_frame": 6,
            "applied": True,
            "dropped": False,
            "action_ready_raw_frames": 2,
            "worker_service_ms": 25.0,
            "worker_service_raw_frames": 1,
        },
        {
            "episode_id": 3,
            "sample_index": 1,
            "action_id": 11,
            "obs_env_step": 8,
            "ready_raw_frame": 11,
            "applied": False,
            "dropped": True,
            "action_ready_raw_frames": 3,
            "worker_service_ms": 70.0,
            "worker_service_raw_frames": 2,
        },
    ]


def test_build_bayes_markov_burst_samples_assigns_episodes_from_metrics():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
        _action_record(action_id=1, obs_env_step=1, ready_raw_frame=2, latency_ms=11.0, applied=False, dropped=True),
        _action_record(action_id=2, obs_env_step=0, ready_raw_frame=3, latency_ms=30.0, applied=True, dropped=False),
    ]
    metrics = [
        {"episode_id": 7, "num_actions": 2, "metadata": {"frame_ms": 40.0}},
        {"episode_id": 8, "num_actions": 1, "metadata": {"frame_ms": 40.0}},
    ]

    samples = build_bayes_markov_burst_samples(action_records, metrics)

    assert [(s["episode_id"], s["action_id"]) for s in samples] == [(7, 0), (7, 1), (8, 2)]


def test_build_bayes_markov_burst_samples_rejects_metric_action_count_mismatch():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
    ]

    with pytest.raises(ValueError, match="episode metric num_actions sum"):
        build_bayes_markov_burst_samples(action_records, [{"episode_id": 0, "num_actions": 2, "metadata": {"frame_ms": 40.0}}])
