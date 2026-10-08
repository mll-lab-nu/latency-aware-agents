"""Decode StarVLA outputs into benchmark actions using task action layouts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from latency_bench.core.types import Action
from latency_bench.data.action_layouts import (
    DEADLY_CORRIDOR_ATTACK_LABELS as _DEADLY_CORRIDOR_ATTACK_LABELS,
    DEADLY_CORRIDOR_MOVE_LABELS as _DEADLY_CORRIDOR_MOVE_LABELS,
    DEADLY_CORRIDOR_STRAFE_LABELS as _DEADLY_CORRIDOR_STRAFE_LABELS,
    DEADLY_CORRIDOR_TURN_LABELS as _DEADLY_CORRIDOR_TURN_LABELS,
    DEMON_ATTACK_ACTION_LABELS as _DEMON_ATTACK_ACTION_LABELS,
)


# Fixed semantic button order the StarVLA multibinary head is trained against.
# Mirrors starVLA.training.rl_games.eval_core._semantic_to_runtime_multibinary;
# the env adapter re-orders this to the live ViZDoom button layout.
DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
)


def action_from_starvla_payload(
    *,
    payload: npt.NDArray[Any],
    env_name: str,
    action_by_raw_id: Mapping[int, Action],
    action_output_type: str = "",
) -> tuple[Action, dict[str, Any]]:
    if str(action_output_type) == "rl_games_continuous":
        values = [float(item) for item in np.asarray(payload).reshape(-1).tolist()]
        return Action(
            value=values,
            name="continuous_torque",
            is_noop=all(value == 0.0 for value in values),
            is_oneshot=False,
        ), {"continuous_action": values}
    if str(env_name) == "demon_attack":
        return demon_attack_action_from_id(int(np.asarray(payload).reshape(-1)[0]))
    if str(env_name) == "deadly_corridor":
        # Multibinary heads emit an already-thresholded 7-dim button vector in
        # fixed semantic order; the env adapter re-orders it to the live ViZDoom
        # button layout. Keep it as-is rather than reinterpreting it as a
        # [turn, move, strafe, attack] categorical tuple.
        if str(action_output_type) == "rl_games_deadly_corridor_multibinary":
            buttons = [int(item) for item in np.asarray(payload).reshape(-1).tolist()]
            active = [
                DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER[idx]
                for idx, pressed in enumerate(buttons)
                if pressed
            ]
            action_name = "+".join(active) if active else "NOOP"
            return Action(
                value=buttons,
                name=action_name,
                is_noop=not any(buttons),
                is_oneshot=False,
            ), {
                "decoded_multibinary_buttons": buttons,
                "action_label": action_name,
                "action_layout": "deadly_corridor_multibinary_7",
            }
        return deadly_corridor_action_from_tuple(
            action_value=[int(item) for item in np.asarray(payload).reshape(-1).tolist()],
            metadata={"action_layout": "deadly_corridor_tuple"},
        )
    raw_action_id = int(np.asarray(payload).reshape(-1)[0])
    return action_by_raw_id[raw_action_id], {"raw_action_id": raw_action_id}


def to_jsonable_action_payload(payload: npt.NDArray[Any]) -> Any:
    value = np.asarray(payload).tolist()
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def demon_attack_action_from_id(action_id: int) -> tuple[Action, dict[str, Any]]:
    action = Action(
        value=action_id,
        name=_DEMON_ATTACK_ACTION_LABELS[action_id],
        is_noop=action_id == 0,
        is_oneshot=False,
    )
    return action, {"raw_action_id": action_id, "action_label": action.name}


def deadly_corridor_action_from_tuple(
    *,
    action_value: list[int],
    metadata: dict[str, Any],
) -> tuple[Action, dict[str, Any]]:
    turn, move, strafe, attack = action_value
    action_value = [turn, move, strafe, attack]
    turn_label = _DEADLY_CORRIDOR_TURN_LABELS[turn]
    move_label = _DEADLY_CORRIDOR_MOVE_LABELS[move]
    strafe_label = _DEADLY_CORRIDOR_STRAFE_LABELS[strafe]
    attack_label = _DEADLY_CORRIDOR_ATTACK_LABELS[attack]
    active_labels = [
        label
        for label in (turn_label, move_label, strafe_label, attack_label)
        if not label.endswith("_NOOP")
    ]
    action_name = "+".join(active_labels) if active_labels else "NOOP"
    return Action(
        value=action_value,
        name=action_name,
        is_noop=action_value == [0, 0, 0, 0],
        is_oneshot=False,
    ), {
        "decoded_action_tuple": action_value,
        "turn_label": turn_label,
        "move_label": move_label,
        "strafe_label": strafe_label,
        "attack_label": attack_label,
        "action_label": action_name,
        **metadata,
    }
