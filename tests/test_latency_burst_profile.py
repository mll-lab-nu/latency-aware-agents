from __future__ import annotations

import pytest

from latency_bench.latency.burst_profile import (
    SemiMarkovLatencyBurstSampler,
    build_latency_burst_model,
    build_latency_burst_samples,
)


def test_build_latency_burst_model_uses_raw_frame_runs_within_each_episode():
    samples = [
        {"episode_id": 0, "sample_index": 0, "latency_ms": 10.0, "latency_raw_frames": 1},
        {"episode_id": 0, "sample_index": 1, "latency_ms": 11.0, "latency_raw_frames": 1},
        {"episode_id": 0, "sample_index": 2, "latency_ms": 20.0, "latency_raw_frames": 2},
        {"episode_id": 1, "sample_index": 3, "latency_ms": 21.0, "latency_raw_frames": 2},
        {"episode_id": 1, "sample_index": 4, "latency_ms": 30.0, "latency_raw_frames": 3},
        {"episode_id": 1, "sample_index": 5, "latency_ms": 31.0, "latency_raw_frames": 3},
    ]

    model = build_latency_burst_model(samples)

    assert model["model_type"] == "semi_markov_latency_frames"
    assert model["reset_scope"] == "episode"
    assert model["state_key"] == "latency_raw_frames"
    assert model["initial_state_counts"] == {"1": 1, "2": 1}
    assert model["initial_state_probs"] == {"1": 0.5, "2": 0.5}
    assert model["transition_counts"] == {
        "1": {"2": 1},
        "2": {"3": 1},
    }
    assert model["transition_probs"] == {
        "1": {"2": 1.0},
        "2": {"3": 1.0},
    }
    assert model["run_lengths_by_state"] == {
        "1": [2],
        "2": [1, 1],
        "3": [2],
    }
    assert model["latency_ms_by_state"] == {
        "1": [10.0, 11.0],
        "2": [20.0, 21.0],
        "3": [30.0, 31.0],
    }


def test_latency_burst_sampler_resets_at_episode_boundary():
    model = {
        "model_type": "semi_markov_latency_frames",
        "reset_scope": "episode",
        "state_key": "latency_raw_frames",
        "initial_state_counts": {"2": 1},
        "transition_counts": {"2": {"4": 1}, "4": {"2": 1}},
        "run_lengths_by_state": {"2": [2], "4": [1]},
        "latency_ms_by_state": {"2": [20.0], "4": [40.0]},
        "terminal_transition_policy": "initial",
    }
    sampler = SemiMarkovLatencyBurstSampler(model, seed=0)

    assert [sampler.sample({"episode_id": 0})["latency_raw_frames"] for _ in range(4)] == [2, 2, 4, 2]
    assert [sampler.sample({"episode_id": 1})["latency_raw_frames"] for _ in range(3)] == [2, 2, 4]


def test_build_latency_burst_samples_reads_applied_non_dropped_action_records():
    action_records = [
        {
            "action_id": 10,
            "obs_env_step": 4,
            "ready_raw_frame": 6,
            "apply_raw_frame": 7,
            "latency_raw_frames": 2,
            "applied": True,
            "dropped": False,
            "latency": {"latency_ms": 25.0},
            "metadata": {"episode_id": 3},
        },
        {
            "action_id": 11,
            "obs_env_step": 8,
            "ready_raw_frame": 10,
            "apply_raw_frame": None,
            "latency_raw_frames": 2,
            "applied": False,
            "dropped": True,
            "latency": {"latency_ms": 26.0},
            "metadata": {"episode_id": 3},
        },
    ]

    assert build_latency_burst_samples(action_records) == [
        {
            "episode_id": 3,
            "sample_index": 0,
            "action_id": 10,
            "obs_env_step": 4,
            "ready_raw_frame": 6,
            "apply_raw_frame": 7,
            "latency_ms": 25.0,
            "latency_raw_frames": 2,
        }
    ]


def test_latency_burst_sampler_interpolates_latency_ms_within_state():
    model = {
        "model_type": "semi_markov_latency_frames",
        "reset_scope": "episode",
        "state_key": "latency_raw_frames",
        "initial_state_counts": {"2": 1},
        "transition_counts": {"2": {"2": 1}},
        "run_lengths_by_state": {"2": [100]},
        "latency_ms_by_state": {"2": [20.0, 40.0]},
        "terminal_transition_policy": "initial",
    }
    sampler = SemiMarkovLatencyBurstSampler(model, seed=3)

    value = sampler.sample({"episode_id": 0})

    assert value["latency_raw_frames"] == 2
    assert value["latency_ms"] == pytest.approx(27.399103330961587)
