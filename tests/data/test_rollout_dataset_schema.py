from __future__ import annotations

import numpy as np


def test_rollout_features_include_env_specific_action_fields():
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.rollout_schema import rollout_features

    flappy_features = rollout_features(action_spec_for_env("flappy"))
    deadly_features = rollout_features(action_spec_for_env("deadly_corridor"))

    assert "action_id" in flappy_features
    assert "action_id" in deadly_features
    for features in (flappy_features, deadly_features):
        assert "decision_step" in features
        assert "latency_raw_frames" in features
        assert "raw_reward" in features

    demon_features = rollout_features(action_spec_for_env("demon_attack"))
    assert "raw_reward" in demon_features


def test_sort_rows_uses_decision_step():
    from latency_bench.data.rollout_schema import sort_rows

    rows = [
        {"episode_idx": 1, "decision_step": 2},
        {"episode_idx": 0, "decision_step": 1},
        {"episode_idx": 0, "decision_step": 0},
    ]

    assert sort_rows(rows) == [
        {"episode_idx": 0, "decision_step": 0},
        {"episode_idx": 0, "decision_step": 1},
        {"episode_idx": 1, "decision_step": 2},
    ]


def test_generic_teacher_row_uses_explicit_action_spec_and_prompt():
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.data.rollout_rows import teacher_dataset_row

    action_spec = gymnasium_discrete_action_spec(["PUSH_LEFT", "COAST", "PUSH_RIGHT"])
    row = teacher_dataset_row(
        env_name="mountain_car",
        spec={"episode_idx": 2, "split": "train", "seed": 19},
        decision_step=4,
        image="frame.png",
        raw_action=2,
        state=np.asarray([1.0, 2.0], dtype=np.float64),
        info={"latency_raw_frames": 3, "latency_ms": 100.0},
        reward=-1.0,
        env_fps=30.0,
        obs_fps=10.0,
        action_spec=action_spec,
        base_prompt="Drive the car up the hill.",
    )

    assert row["env_name"] == "mountain_car"
    assert row["action_id"] == 2
    assert row["action"] == [0.0, 0.0, 1.0]
    assert row["action_text"] == "PUSH_RIGHT"
    assert row["state"] == [1.0, 2.0]
    assert row["prompt"].startswith("Drive the car up the hill.")
    assert "Current action latency is 3 raw frames (100.00 ms)." in row["prompt"]
