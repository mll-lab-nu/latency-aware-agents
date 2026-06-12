from __future__ import annotations


def test_canonical_rollout_fields_exclude_legacy_aliases():
    from latency_bench.data.rollout_schema import CANONICAL_ROW_FIELDS

    assert CANONICAL_ROW_FIELDS == (
        "image",
        "env_name",
        "episode_idx",
        "decision_step",
        "split",
        "seed",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
        "reward",
    )
    assert "latency" not in CANONICAL_ROW_FIELDS
    assert "deterministic" not in CANONICAL_ROW_FIELDS
    assert "done" not in CANONICAL_ROW_FIELDS
    assert "terminated" not in CANONICAL_ROW_FIELDS
    assert "truncated" not in CANONICAL_ROW_FIELDS
    assert "t" not in CANONICAL_ROW_FIELDS


def test_rollout_features_include_env_specific_action_fields():
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.rollout_schema import rollout_features

    flappy_features = rollout_features(action_spec_for_env("flappy"))
    deadly_features = rollout_features(action_spec_for_env("deadly_corridor"))

    assert "action_id" in flappy_features
    assert "action_tuple" not in flappy_features
    assert "action_tuple" in deadly_features
    assert "action_id" not in deadly_features
    for features in (flappy_features, deadly_features):
        assert "decision_step" in features
        assert "latency_raw_frames" in features
        assert "latency" not in features
        assert "done" not in features
        assert "t" not in features


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
