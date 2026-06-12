from __future__ import annotations


def test_deadly_corridor_canonical_action_layout_is_factorized_eleven_dim():
    from latency_bench.data.action_layouts import (
        DEADLY_CORRIDOR_FACTORIZED_11_LABELS,
        action_spec_for_env,
        encode_action_for_env,
    )

    spec = action_spec_for_env("deadly_corridor")
    row = encode_action_for_env("deadly_corridor", [0, 1, 2, 1])

    assert spec.layout == "deadly_corridor_factorized_11"
    assert spec.labels == DEADLY_CORRIDOR_FACTORIZED_11_LABELS
    assert row["action_tuple"] == [0, 1, 2, 1]
    assert row["action"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert row["action_text"] == "MOVE_FORWARD+MOVE_RIGHT+ATTACK"


def test_deadly_corridor_factorized_action_matches_starvla_policy_decoder():
    import numpy as np
    from latency_bench.policy.starvla import decode_starvla_action

    action, metadata = decode_starvla_action(
        vector=np.asarray([1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.8, 0.0, 0.7]),
        env_name="deadly_corridor",
        action_by_raw_id={},
    )

    assert action.value == [0, 1, 2, 1]
    assert action.name == "MOVE_FORWARD+MOVE_RIGHT+ATTACK"
    assert metadata["action_layout"] == "deadly_corridor_factorized_11"
