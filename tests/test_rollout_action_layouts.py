from __future__ import annotations


def test_deadly_corridor_default_action_layout_is_factorized_eleven_dim():
    from latency_bench.data.action_layouts import (
        ACTION_LAYOUT_DEADLY_CORRIDOR_FACTORIZED_11,
        DEADLY_CORRIDOR_FACTORIZED_11_LABELS,
        action_spec_for_env,
        encode_action_for_env,
    )

    spec = action_spec_for_env("deadly_corridor")
    row = encode_action_for_env("deadly_corridor", [2, 1, 2, 1])

    assert spec.layout == ACTION_LAYOUT_DEADLY_CORRIDOR_FACTORIZED_11
    assert spec.labels == DEADLY_CORRIDOR_FACTORIZED_11_LABELS
    assert row == {
        "action_tuple": [2, 1, 2, 1],
        "action": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "action_text": "TURN_RIGHT+MOVE_FORWARD+MOVE_RIGHT+ATTACK",
    }


def test_deadly_corridor_noop_components_stay_in_vector_not_text():
    from latency_bench.data.action_layouts import encode_action_for_env

    row = encode_action_for_env("deadly_corridor", [0, 0, 1, 0])

    assert row["action"] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert row["action_text"] == "MOVE_LEFT"


def test_discrete_env_action_layout_keeps_action_id():
    from latency_bench.data.action_layouts import encode_action_for_env

    row = encode_action_for_env("flappy", 1)

    assert row == {
        "action_id": 1,
        "action": [0.0, 1.0],
        "action_text": "FLAP",
    }
