from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_DISCRETE_V1 as GYMNASIUM_DISCRETE_ACTION_LAYOUT,
    gymnasium_discrete_action_spec,
)


def gymnasium_discrete_action_labels(env_config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(env_config["action_order"])


def gymnasium_discrete_action_values(env_config: Mapping[str, Any]) -> tuple[Any, ...]:
    action_map = env_config["action_map"]
    return tuple(action_map[label] for label in gymnasium_discrete_action_labels(env_config))


def gymnasium_noop_action_id(env_config: Mapping[str, Any]) -> int:
    return gymnasium_discrete_action_labels(env_config).index(env_config["noop_action"])


def gymnasium_task_contract(env_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable task semantics shared by training, data, and evaluation."""
    return {
        "task_name": env_config["task_name"],
        "env_id": env_config["env_id"],
        "make_kwargs": env_config["make_kwargs"],
        "registration_imports": env_config["registration_imports"],
        "action_labels": list(gymnasium_discrete_action_labels(env_config)),
        "action_values": list(gymnasium_discrete_action_values(env_config)),
        "noop_action_id": gymnasium_noop_action_id(env_config),
        "base_prompt": env_config["base_prompt"],
        "env_fps": env_config["env_fps"],
        "obs_fps": env_config["obs_fps"],
        "frame_stack": env_config["frame_stack"],
    }
