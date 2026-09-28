from __future__ import annotations


def test_deadly_corridor_default_action_layout_is_joint_fifty_four_id():
    from latency_bench.data.action_layouts import (
        ACTION_LAYOUT_DEADLY_CORRIDOR_JOINT_54,
        DEADLY_CORRIDOR_JOINT_54_LABELS,
        action_spec_for_env,
        encode_action_for_env,
    )

    spec = action_spec_for_env("deadly_corridor")
    row = encode_action_for_env("deadly_corridor", [2, 1, 2, 1])

    assert spec.layout == ACTION_LAYOUT_DEADLY_CORRIDOR_JOINT_54
    assert spec.labels == DEADLY_CORRIDOR_JOINT_54_LABELS
    assert row == {
        "action_id": 47,
        "action_text": "TURN_RIGHT + MOVE_FORWARD + MOVE_RIGHT + ATTACK",
    }


def test_deadly_corridor_noop_components_stay_in_vector_not_text():
    from latency_bench.data.action_layouts import encode_action_for_env

    row = encode_action_for_env("deadly_corridor", [0, 0, 1, 0])

    assert row["action_id"] == 2
    assert row["action_text"] == "MOVE_LEFT"


def test_defend_the_line_default_action_layout_is_joint_six_id():
    from latency_bench.data.action_layouts import (
        ACTION_LAYOUT_DEFEND_THE_LINE_JOINT_6,
        DEFEND_THE_LINE_JOINT_6_LABELS,
        action_spec_for_env,
        encode_action_for_env,
    )

    spec = action_spec_for_env("defend_the_line")
    row = encode_action_for_env("defend_the_line", [2, 1])

    assert spec.layout == ACTION_LAYOUT_DEFEND_THE_LINE_JOINT_6
    assert spec.labels == DEFEND_THE_LINE_JOINT_6_LABELS
    assert row == {
        "action_id": 5,
        "action_text": "TURN_RIGHT + ATTACK",
    }


def test_discrete_env_action_layout_keeps_action_id():
    from latency_bench.data.action_layouts import encode_action_for_env

    row = encode_action_for_env("flappy", 1)

    assert row == {
        "action_id": 1,
        "action": [0.0, 1.0],
        "action_text": "FLAP",
    }
