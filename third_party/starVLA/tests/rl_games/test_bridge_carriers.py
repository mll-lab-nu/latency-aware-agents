import pytest

pytest.importorskip("datasets")
pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

from examples.rl_games.bash_scripts.gr00t.data_conversion import convert_deadly_corridor_to_starvla_lerobot as deadly
from examples.rl_games.bash_scripts.gr00t.data_conversion import convert_demon_attack_to_starvla_lerobot as demon
from examples.rl_games.bash_scripts.gr00t.data_conversion import convert_flappy_to_starvla_lerobot as flappy


def test_flappy_bridge_labels_pad_to_7d_carrier():
    assert flappy._action_dim("native") == 2
    assert flappy._action_labels("native") == ["NOOP", "FLAP"]
    assert flappy._state_dim("native") == 1

    assert flappy._action_dim("bridge") == 7
    assert flappy._action_labels("bridge") == [
        "NOOP",
        "FLAP",
        "BRIDGE_PAD_2",
        "BRIDGE_PAD_3",
        "BRIDGE_PAD_4",
        "BRIDGE_PAD_5",
        "BRIDGE_PAD_6",
    ]
    assert flappy._state_dim("bridge") == 7
    assert flappy._state_labels("bridge") == [f"BRIDGE_STATE_{idx}" for idx in range(7)]


def test_demon_attack_bridge_labels_pad_to_7d_carrier():
    assert demon._action_dim("native") == 6
    assert demon._action_labels("native") == ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    assert demon._state_dim("native") == 1

    assert demon._action_dim("bridge") == 7
    assert demon._action_labels("bridge") == [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
        "BRIDGE_PAD_6",
    ]
    assert demon._state_dim("bridge") == 7
    assert demon._state_labels("bridge") == [f"BRIDGE_STATE_{idx}" for idx in range(7)]


def test_deadly_corridor_bridge_uses_native_7d_semantic_carrier():
    expected = ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK"]

    assert deadly._action_dim("native") == 7
    assert deadly._action_labels("native") == expected
    assert deadly._state_dim("native") == 1

    assert deadly._action_dim("bridge") == 7
    assert deadly._action_labels("bridge") == expected
    assert deadly._state_dim("bridge") == 7
    assert deadly._state_labels("bridge") == [f"BRIDGE_STATE_{idx}" for idx in range(7)]


def test_deadly_corridor_factorized_11_native_layout():
    expected = [
        "TURN_NONE",
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_NONE",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "STRAFE_NONE",
        "STRAFE_LEFT",
        "STRAFE_RIGHT",
        "ATTACK_OFF",
        "ATTACK_ON",
    ]

    assert deadly._action_dim("native", action_layout="factorized_11") == 11
    assert deadly._action_labels("native", action_layout="factorized_11") == expected
    assert deadly._action_vector({"action_tuple": [2, 1, 0, 1]}, action_layout="factorized_11") == [
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
