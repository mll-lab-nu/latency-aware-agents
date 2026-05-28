from __future__ import annotations

import sys
from pathlib import Path


STARVLA_ROOT = Path(__file__).resolve().parents[1] / "starVLA"
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def test_deadly_corridor_converter_uses_factorized_eleven_dim_actions():
    from examples.rl_games.data_conversion import convert_deadly_corridor_to_starvla_lerobot as converter

    assert converter.ACTION_DIM == 11
    assert converter._action_vector({"action": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]}) == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]


def test_deadly_corridor_converter_encodes_joint_fifty_four_actions():
    from examples.rl_games.data_conversion import convert_deadly_corridor_to_starvla_lerobot as converter

    action = converter._action_vector(
        {"action": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]},
        action_layout="joint_54",
    )

    assert len(action) == 54
    assert sum(action) == 1.0
    assert action[11] == 1.0
    assert converter.joint_id_to_deadly_corridor_tuple(11) == [0, 1, 2, 1]
