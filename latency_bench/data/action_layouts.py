"""Canonical rollout action layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_LAYOUT_FLAPPY_DISCRETE = "flappy_discrete"
ACTION_LAYOUT_DEMON_ATTACK_DISCRETE = "demon_attack_discrete"
ACTION_LAYOUT_DEADLY_CORRIDOR_FACTORIZED_11 = "deadly_corridor_factorized_11"
ACTION_LAYOUT_DEADLY_CORRIDOR_JOINT_54 = "deadly_corridor_joint_54"

FLAPPY_ACTION_LABELS = ("NOOP", "FLAP")
DEMON_ATTACK_ACTION_LABELS = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")
DEADLY_CORRIDOR_TURN_LABELS = ("TURN_NOOP", "TURN_LEFT", "TURN_RIGHT")
DEADLY_CORRIDOR_MOVE_LABELS = ("MOVE_NOOP", "MOVE_FORWARD", "MOVE_BACKWARD")
DEADLY_CORRIDOR_STRAFE_LABELS = ("STRAFE_NOOP", "MOVE_LEFT", "MOVE_RIGHT")
DEADLY_CORRIDOR_ATTACK_LABELS = ("ATTACK_NOOP", "ATTACK")
DEADLY_CORRIDOR_FACTORIZED_11_LABELS = (
    *DEADLY_CORRIDOR_TURN_LABELS,
    *DEADLY_CORRIDOR_MOVE_LABELS,
    *DEADLY_CORRIDOR_STRAFE_LABELS,
    *DEADLY_CORRIDOR_ATTACK_LABELS,
)
DEADLY_CORRIDOR_JOINT_54_LABELS = tuple(
    " + ".join(
        label
        for label in (
            DEADLY_CORRIDOR_TURN_LABELS[turn],
            DEADLY_CORRIDOR_MOVE_LABELS[move],
            DEADLY_CORRIDOR_STRAFE_LABELS[strafe],
            DEADLY_CORRIDOR_ATTACK_LABELS[attack],
        )
        if not label.endswith("_NOOP")
    )
    or "NOOP"
    for turn in range(len(DEADLY_CORRIDOR_TURN_LABELS))
    for move in range(len(DEADLY_CORRIDOR_MOVE_LABELS))
    for strafe in range(len(DEADLY_CORRIDOR_STRAFE_LABELS))
    for attack in range(len(DEADLY_CORRIDOR_ATTACK_LABELS))
)


@dataclass(frozen=True)
class ActionSpec:
    """Describe how an environment stores action fields in rollout rows.

    Args:
        layout: Stable layout identifier written to rollout metadata.
        labels: Canonical action labels used for vectors and metadata.
        row_fields: Action-related fields expected in each row.
        reward_field: Reward column written in canonical rollout rows.
        episode_return_field: Episode return field written in rollout reports.
        reward_semantics: Human-readable reward semantics stored in metadata.
    """
    layout: str
    labels: tuple[str, ...]
    row_fields: tuple[str, ...]
    reward_field: str
    episode_return_field: str
    reward_semantics: str


ACTION_SPECS_BY_ENV = {
    "flappy": ActionSpec(
        layout=ACTION_LAYOUT_FLAPPY_DISCRETE,
        labels=FLAPPY_ACTION_LABELS,
        row_fields=("action_id", "action", "action_text"),
        reward_field="raw_reward",
        episode_return_field="episode_raw_return",
        reward_semantics="raw_reward",
    ),
    "demon_attack": ActionSpec(
        layout=ACTION_LAYOUT_DEMON_ATTACK_DISCRETE,
        labels=DEMON_ATTACK_ACTION_LABELS,
        row_fields=("action_id", "action", "action_text"),
        reward_field="raw_reward",
        episode_return_field="episode_raw_return",
        reward_semantics="raw_reward",
    ),
    "deadly_corridor": ActionSpec(
        layout=ACTION_LAYOUT_DEADLY_CORRIDOR_JOINT_54,
        labels=DEADLY_CORRIDOR_JOINT_54_LABELS,
        row_fields=("action_id", "action_text"),
        reward_field="raw_reward",
        episode_return_field="episode_raw_return",
        reward_semantics="raw_reward",
    ),
}


def action_spec_for_env(env_name: str) -> ActionSpec:
    """Return the canonical action spec for an environment.

    Args:
        env_name: Canonical benchmark environment name.

    Returns:
        The action spec that defines row fields and label order.
    """
    return ACTION_SPECS_BY_ENV[str(env_name)]


def _to_python_action(value: Any) -> Any:
    """Convert tensor/ndarray action values into Python scalars or lists.

    Args:
        value: Raw action value returned by policy or Sample Factory helpers.

    Returns:
        A Python scalar or list preserving the action semantics.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], list):
        return value[0]
    return value


def _one_hot(index: int, size: int) -> list[float]:
    """Encode an action index as a float one-hot vector.

    Args:
        index: Active index in the vector.
        size: Vector length.

    Returns:
        One-hot vector as floats.
    """
    return [1.0 if item == int(index) else 0.0 for item in range(int(size))]


def _action_id(raw_action: Any) -> int:
    """Extract a discrete action id from raw policy output.

    Args:
        raw_action: Scalar or singleton sequence action.

    Returns:
        Discrete action id.
    """
    action = _to_python_action(raw_action)
    if isinstance(action, (list, tuple)):
        return int(action[0])
    return int(action)


def _deadly_corridor_tuple(raw_action: Any) -> list[int]:
    """Extract the four factorized Deadly Corridor action components.

    Args:
        raw_action: Raw tuple action in turn/move/strafe/attack order.

    Returns:
        Four integer component ids.
    """
    return [int(item) for item in _to_python_action(raw_action)]


def _deadly_corridor_action_text(action_tuple: list[int]) -> str:
    """Render active Deadly Corridor tuple components as readable text.

    Args:
        action_tuple: Four component ids in turn/move/strafe/attack order.

    Returns:
        Plus-joined active labels, or ``NOOP`` when all components are noop.
    """
    turn, move, strafe, attack = action_tuple
    labels = (
        DEADLY_CORRIDOR_TURN_LABELS[turn],
        DEADLY_CORRIDOR_MOVE_LABELS[move],
        DEADLY_CORRIDOR_STRAFE_LABELS[strafe],
        DEADLY_CORRIDOR_ATTACK_LABELS[attack],
    )
    active_labels = [label for label in labels if not label.endswith("_NOOP")]
    return " + ".join(active_labels) if active_labels else "NOOP"


def _deadly_corridor_joint_action_id(action_tuple: list[int]) -> int:
    turn, move, strafe, attack = action_tuple
    return int(turn) * 18 + int(move) * 6 + int(strafe) * 2 + int(attack)


def deadly_corridor_joint_54_action(raw_action: Any) -> dict[str, Any]:
    """Encode a Deadly Corridor tuple action as the legacy joint action id."""
    action_tuple = _deadly_corridor_tuple(raw_action)
    return {
        "action_id": _deadly_corridor_joint_action_id(action_tuple),
        "action_text": _deadly_corridor_action_text(action_tuple),
    }


def discrete_action(env_name: str, raw_action: Any) -> dict[str, Any]:
    """Encode a discrete environment action into canonical row fields.

    Args:
        env_name: Canonical benchmark environment name.
        raw_action: Raw discrete action id from the policy.

    Returns:
        Row fields ``action_id``, ``action``, and ``action_text``.
    """
    action_id = _action_id(raw_action)
    action_labels = action_spec_for_env(env_name).labels
    return {
        "action_id": action_id,
        "action": _one_hot(action_id, len(action_labels)),
        "action_text": action_labels[int(action_id)],
    }


def encode_action_for_env(env_name: str, raw_action: Any) -> dict[str, Any]:
    """Encode an environment action using the canonical rollout layout.

    Args:
        env_name: Canonical benchmark environment name.
        raw_action: Raw action produced for the environment.

    Returns:
        Environment-specific action row fields.
    """
    if str(env_name) == "deadly_corridor":
        return deadly_corridor_joint_54_action(raw_action)
    return discrete_action(str(env_name), raw_action)
