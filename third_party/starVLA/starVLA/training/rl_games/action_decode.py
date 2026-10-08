from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np


DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
)

DEFEND_THE_LINE_ACTIONS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
)

_DEADLY_LEGACY_LAYOUT_BY_DIM = {
    7: "semantic_7",
    11: "factorized_11",
    54: "joint_54",
}

_DEADLY_ACTION_LAYOUT_ALIASES = {
    "multibinary_7": "multibinary_7",
    "semantic_7": "semantic_7",
    "factorized_11": "factorized_11",
    "joint_54": "joint_54",
    "deadly_corridor_multibinary_7": "multibinary_7",
    "deadly_corridor_factorized_11": "factorized_11",
    "deadly_corridor_joint_54": "joint_54",
}

_ZERO_LOGIT_THRESHOLD_LOSS_TYPES = {
    "current_multibinary_bce",
    "current_bce",
    "multibinary_bce",
    "multibinary_ce",
    "bce",
    "binary_cross_entropy",
}

_ASTERIX_ACTION_LAYOUT_ALIASES = {
    "discrete_9": "discrete_9",
    "joint_9": "discrete_9",
    "asterix_discrete_9": "discrete_9",
    "factorized_6": "factorized_6",
    "factorized6": "factorized_6",
    "asterix_factorized_6": "factorized_6",
    "asterix_factorized6": "factorized_6",
}

_ASTERIX_FACTORIZED_6_TO_ACTION_ID = np.asarray(
    (
        (0, 2, 3),
        (1, 5, 6),
        (4, 7, 8),
    ),
    dtype=np.int64,
)


def _optional_config_value(config: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def resolve_deadly_action_decode_spec(
    model_config: Mapping[str, Any],
    *,
    action_layout: str | None = None,
    multibinary_threshold: float | None = None,
) -> tuple[str | None, float]:
    if action_layout is not None:
        action_layout = _DEADLY_ACTION_LAYOUT_ALIASES[str(action_layout).strip().lower()]
    else:
        saved_layout = _optional_config_value(
            model_config,
            ("rl_games", "env_eval", "deadly", "action_layout"),
        )
        if saved_layout is None:
            saved_layout = _optional_config_value(
                model_config,
                ("framework", "action_model", "action_layout"),
            )
        if saved_layout is not None:
            action_layout = _DEADLY_ACTION_LAYOUT_ALIASES[str(saved_layout).strip().lower()]

    if multibinary_threshold is not None:
        multibinary_threshold = float(multibinary_threshold)
    else:
        saved_threshold = _optional_config_value(
            model_config,
            ("rl_games", "env_eval", "deadly", "multibinary_threshold"),
        )
        if saved_threshold is not None:
            multibinary_threshold = float(saved_threshold)

    if multibinary_threshold is None:
        loss_type = str(model_config["framework"]["action_model"]["loss_type"]).strip().lower()
        multibinary_threshold = 0.0 if loss_type in _ZERO_LOGIT_THRESHOLD_LOSS_TYPES else 0.5

    return action_layout, multibinary_threshold


def decode_rl_games_actions(
    *,
    normalized_actions: np.ndarray,
    env_name: str,
    deadly_action_layout: str | None = None,
    deadly_multibinary_threshold: float | None = None,
    asterix_action_layout: str | None = None,
    action_env_dim: int | None = None,
    gymnasium_action_space_type: str = "discrete",
    action_values: np.ndarray | None = None,
) -> dict[str, Any]:
    raw_scores = np.asarray(normalized_actions)
    values = raw_scores if action_values is None else np.asarray(action_values)
    decoder = {
        "flappy": lambda: (_decode_discrete(raw_scores, 2), "rl_games_discrete_id"),
        "demon_attack": lambda: (_decode_discrete(raw_scores, 6), "rl_games_discrete_id"),
        "air_raid": lambda: (_decode_discrete(raw_scores, 6), "rl_games_discrete_id"),
        "defend_the_line": lambda: (_decode_discrete(raw_scores, 6), "rl_games_defend_the_line_joint_6"),
        "asterix": lambda: decode_asterix_actions(raw_scores, action_layout=asterix_action_layout),
        "atlantis": lambda: (_decode_discrete(raw_scores, 4), "rl_games_discrete_id"),
        "gymnasium": lambda: (
            (values[..., :action_env_dim], "rl_games_continuous")
            if gymnasium_action_space_type == "box"
            else (_decode_discrete_explicit_dim(raw_scores, action_env_dim), "rl_games_discrete_id")
        ),
        "deadly_corridor": lambda: decode_deadly_corridor_actions(
            raw_scores,
            action_layout=deadly_action_layout,
            multibinary_threshold=deadly_multibinary_threshold,
        ),
    }[env_name]
    actions, action_output_type = decoder()
    return {
        "actions": actions,
        "raw_action_scores": raw_scores,
        "action_output_type": action_output_type,
    }


def resolve_asterix_action_decode_spec(
    model_config: Mapping[str, Any],
    *,
    action_layout: str | None = None,
) -> str:
    if action_layout is not None:
        return _ASTERIX_ACTION_LAYOUT_ALIASES[str(action_layout).strip().lower()]
    saved_layout = _optional_config_value(
        model_config,
        ("rl_games", "env_eval", "asterix", "action_layout"),
    )
    if saved_layout is None:
        saved_layout = _optional_config_value(
            model_config,
            ("framework", "action_model", "action_layout"),
        )
    if saved_layout is None:
        return "discrete_9"
    return _ASTERIX_ACTION_LAYOUT_ALIASES[str(saved_layout).strip().lower()]


def decode_asterix_actions(
    raw_scores: np.ndarray,
    *,
    action_layout: str | None,
) -> tuple[np.ndarray, str]:
    values = np.asarray(raw_scores)
    layout = "discrete_9" if action_layout is None else _ASTERIX_ACTION_LAYOUT_ALIASES[str(action_layout).strip().lower()]
    if layout == "discrete_9":
        return _decode_discrete(values, 9), "rl_games_discrete_id"
    if layout == "factorized_6":
        if values.shape[-1] < 6:
            raise ValueError(f"Asterix factorized_6 decode requires at least 6 logits, got {values.shape[-1]}")
        vertical = np.argmax(values[..., 0:3], axis=-1)
        horizontal = np.argmax(values[..., 3:6], axis=-1)
        return _ASTERIX_FACTORIZED_6_TO_ACTION_ID[vertical, horizontal][..., None].astype(np.int64), "rl_games_discrete_id"
    raise ValueError(f"Unsupported Asterix action layout: {action_layout!r}")


def decode_deadly_corridor_actions(
    raw_scores: np.ndarray,
    *,
    action_layout: str | None,
    multibinary_threshold: float | None,
) -> tuple[np.ndarray, str]:
    values = np.asarray(raw_scores)
    layout = action_layout or _DEADLY_LEGACY_LAYOUT_BY_DIM[values.shape[-1]]
    if layout == "multibinary_7":
        actions = (values[..., :7] >= multibinary_threshold).astype(np.int64)
        return actions, "rl_games_deadly_corridor_multibinary"
    if layout == "semantic_7":
        return _decode_deadly_semantic_7(values), "rl_games_deadly_corridor_tuple"
    if layout == "factorized_11":
        return _decode_deadly_factorized_11(values), "rl_games_deadly_corridor_tuple"
    decoders = {
        "joint_54": _decode_deadly_joint_54,
    }
    return decoders[layout](values), "rl_games_deadly_corridor_tuple"


def deadly_tuple_to_semantic_buttons(action_tuple: np.ndarray) -> np.ndarray:
    values = np.asarray(action_tuple)
    semantic = np.zeros((*values.shape[:-1], 7), dtype=np.int64)
    semantic[..., 4] = values[..., 0] == 1
    semantic[..., 5] = values[..., 0] == 2
    semantic[..., 0] = values[..., 1] == 1
    semantic[..., 1] = values[..., 1] == 2
    semantic[..., 2] = values[..., 2] == 1
    semantic[..., 3] = values[..., 2] == 2
    semantic[..., 6] = values[..., 3] == 1
    return semantic


def decode_discrete_argmax(action_values: Iterable[float], n_actions: int) -> int:
    decoded = _decode_discrete(np.asarray(action_values, dtype=np.float32), n_actions)
    return int(decoded[0])


def decode_defend_the_line_tuple(action_values: Iterable[float]) -> list[int]:
    action_id = decode_discrete_argmax(action_values, len(DEFEND_THE_LINE_ACTIONS))
    return list(DEFEND_THE_LINE_ACTIONS[action_id])


def decode_defend_the_line_multibinary(action_values: Iterable[float]) -> list[int]:
    turn, attack = decode_defend_the_line_tuple(action_values)
    return [int(turn == 1), int(turn == 2), int(attack == 1)]


def decode_deadly_multibinary_7(
    action_values: Iterable[float],
    threshold: float = 0.5,
) -> list[int]:
    decoded, _ = decode_deadly_corridor_actions(
        np.asarray(action_values, dtype=np.float32),
        action_layout="multibinary_7",
        multibinary_threshold=threshold,
    )
    return decoded.tolist()


def decode_deadly_factorized_11(action_values: Iterable[float]) -> list[int]:
    decoded, _ = decode_deadly_corridor_actions(
        np.asarray(action_values, dtype=np.float32),
        action_layout="factorized_11",
        multibinary_threshold=None,
    )
    return decoded.tolist()


def _decode_discrete_explicit_dim(raw_scores: np.ndarray, n_actions: int | None) -> np.ndarray:
    active_scores = np.take(raw_scores, np.arange(n_actions), axis=-1)
    return np.argmax(active_scores, axis=-1)[..., None].astype(np.int64)


def _decode_discrete(raw_scores: np.ndarray, n_actions: int) -> np.ndarray:
    return np.argmax(raw_scores[..., :n_actions], axis=-1)[..., None].astype(np.int64)


def _decode_deadly_factorized_11(raw_scores: np.ndarray) -> np.ndarray:
    turn = np.argmax(raw_scores[..., 0:3], axis=-1)
    move = np.argmax(raw_scores[..., 3:6], axis=-1)
    strafe = np.argmax(raw_scores[..., 6:9], axis=-1)
    attack = np.argmax(raw_scores[..., 9:11], axis=-1)
    return np.stack((turn, move, strafe, attack), axis=-1).astype(np.int64)


def _decode_deadly_joint_54(raw_scores: np.ndarray) -> np.ndarray:
    joint_id = np.argmax(raw_scores, axis=-1)
    attack = joint_id % 2
    strafe = (joint_id // 2) % 3
    move = (joint_id // 6) % 3
    turn = (joint_id // 18) % 3
    return np.stack((turn, move, strafe, attack), axis=-1).astype(np.int64)


def _decode_deadly_semantic_7(raw_scores: np.ndarray) -> np.ndarray:
    action_id = np.argmax(raw_scores, axis=-1)
    semantic_actions = np.asarray(
        (
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 2, 0),
            (1, 0, 0, 0),
            (2, 0, 0, 0),
            (0, 0, 0, 1),
        ),
        dtype=np.int64,
    )
    return semantic_actions[action_id]
