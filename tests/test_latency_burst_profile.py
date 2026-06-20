from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    build_bayes_markov_burst_model,
    build_bayes_markov_burst_samples,
)


def _model_sample(episode_id: int, sample_index: int, e2e_latency_ms: float, worker_service_ms: float) -> dict:
    return {
        "episode_id": episode_id,
        "sample_index": sample_index,
        "e2e_latency_ms": e2e_latency_ms,
        "worker_service_ms": worker_service_ms,
        "e2e_latency_ms_bin": int(e2e_latency_ms // 1.0),
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
        "applied": applied,
        "dropped": dropped,
        "latency": {"latency_ms": latency_ms},
        "metadata": metadata,
    }


def test_build_bayes_markov_burst_model_uses_e2e_ms_bin_transitions():
    # Single episode e2e bin sequence: [40, 40, 81, 120, 120]
    samples = [
        _model_sample(0, 0, 40.1, 39.5),
        _model_sample(0, 1, 40.8, 39.9),
        _model_sample(0, 2, 81.2, 79.0),
        _model_sample(0, 3, 120.0, 118.0),
        _model_sample(0, 4, 120.9, 119.0),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["schema_version"] == 4
    assert model["model_type"] == "bayes_markov"
    assert model["reset_scope"] == "episode"
    assert model["state_key"] == "e2e_latency_ms_bin"
    assert model["bin_width_ms"] == 1.0
    assert model["initial_state_counts"] == {"40": 1}
    # Per-adjacent-pair transitions (NOT per-run): (40,40)(40,81)(81,120)(120,120).
    assert model["transition_counts"] == {
        "40": {"40": 1, "81": 1},
        "81": {"120": 1},
        "120": {"120": 1},
    }
    assert model["run_lengths_by_state"] == {"40": [2], "81": [1], "120": [2]}
    assert model["state_counts"] == {"40": 2, "81": 1, "120": 2}
    assert "emissions_by_state" not in model
    assert model["emission_distributions_by_state"]["40"]["count"] == 2
    assert model["emission_distributions_by_state"]["40"]["e2e_latency_ms"]["latency_ms"][0] == pytest.approx(40.1)
    assert model["emission_distributions_by_state"]["120"]["worker_service_ms"]["latency_ms"][-1] == pytest.approx(119.0)


def test_build_bayes_markov_burst_model_counts_initial_state_per_episode():
    samples = [
        _model_sample(0, 0, 40.0, 39.0),
        _model_sample(0, 1, 80.0, 78.0),
        _model_sample(1, 2, 80.0, 78.0),
        _model_sample(1, 3, 40.0, 39.0),
    ]

    model = build_bayes_markov_burst_model(samples)

    assert model["initial_state_counts"] == {"40": 1, "80": 1}
    assert model["transition_counts"] == {"40": {"80": 1}, "80": {"40": 1}}


def test_build_bayes_markov_burst_samples_uses_applied_e2e_records():
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
            "e2e_latency_ms": 53.0,
            "e2e_latency_ms_bin": 53,
            "worker_service_ms": 25.0,
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
