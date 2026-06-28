from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    build_bayes_markov_burst_model,
    build_bayes_markov_burst_samples,
)


def _model_sample(episode_id: int, sample_index: int, action_ready_ms: float, worker_service_ms: float) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "action_ready_ms": action_ready_ms,
        "worker_service_ms": worker_service_ms,
        "worker_service_ms_bin": int(worker_service_ms // 1.0),
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


def test_build_bayes_markov_burst_model_uses_worker_service_ms_bin_transitions():
    # Single episode worker-service bin sequence: [39, 39, 79, 118, 119]
    samples = [
        _model_sample(0, 0, 40.1, 39.5),
        _model_sample(0, 1, 40.8, 39.9),
        _model_sample(0, 2, 81.2, 79.0),
        _model_sample(0, 3, 120.0, 118.0),
        _model_sample(0, 4, 120.9, 119.0),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["schema_version"] == 5
    assert model["model_type"] == "bayes_markov"
    assert model["reset_scope"] == "episode"
    assert model["state_key"] == "worker_service_ms_bin"
    assert model["bin_width_ms"] == 1.0
    assert model["initial_state_counts"] == {"39": 1}
    # Per-adjacent-pair transitions (NOT per-run): (39,39)(39,79)(79,118)(118,119).
    assert model["transition_counts"] == {
        "39": {"39": 1, "79": 1},
        "79": {"118": 1},
        "118": {"119": 1},
    }
    assert model["run_lengths_by_state"] == {"39": [2], "79": [1], "118": [1], "119": [1]}
    assert model["state_counts"] == {"39": 2, "79": 1, "118": 1, "119": 1}
    assert "emissions_by_state" not in model
    assert model["emission_distributions_by_state"]["39"]["count"] == 2
    assert model["emission_distributions_by_state"]["39"]["action_ready_ms"]["latency_ms"][0] == pytest.approx(40.1)
    assert model["emission_distributions_by_state"]["119"]["worker_service_ms"]["latency_ms"][-1] == pytest.approx(119.0)


def test_build_bayes_markov_burst_model_counts_initial_state_per_episode():
    samples = [
        _model_sample(0, 0, 40.0, 39.0),
        _model_sample(0, 1, 80.0, 78.0),
        _model_sample(1, 2, 80.0, 78.0),
        _model_sample(1, 3, 40.0, 39.0),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["initial_state_counts"] == {"39": 1, "78": 1}
    assert model["transition_counts"] == {"39": {"78": 1}, "78": {"39": 1}}


def test_build_bayes_markov_burst_samples_uses_all_completed_worker_service_records():
    action_records = [
        _action_record(action_id=10, obs_env_step=4, ready_raw_frame=6, latency_ms=25.0, applied=True, dropped=False),
        _action_record(action_id=11, obs_env_step=8, ready_raw_frame=11, latency_ms=70.0, applied=False, dropped=False),
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


def test_build_bayes_markov_burst_samples_assigns_episodes_from_metrics():
    action_records = [
        _action_record(action_id=0, obs_env_step=0, ready_raw_frame=1, latency_ms=10.0, applied=True, dropped=False),
        _action_record(action_id=1, obs_env_step=1, ready_raw_frame=2, latency_ms=11.0, applied=True, dropped=False),
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
