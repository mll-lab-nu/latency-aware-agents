from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_DISCRETE_V1 as GYMNASIUM_DISCRETE_ACTION_LAYOUT,
    gymnasium_discrete_action_spec,
)


_PRE_RELEASE_REGISTRATION_MODULES = {
    "latency_bench.envs.gymnasium_air_raid": "latency_bench.envs.tasks.air_raid",
    "latency_bench.envs.gymnasium_ant": "latency_bench.envs.tasks.ant",
    "latency_bench.envs.gymnasium_asterix": "latency_bench.envs.tasks.asterix",
    "latency_bench.envs.gymnasium_atlantis": "latency_bench.envs.tasks.atlantis",
    "latency_bench.envs.gymnasium_half_cheetah": "latency_bench.envs.tasks.half_cheetah",
    "latency_bench.envs.gymnasium_hopper": "latency_bench.envs.tasks.hopper",
    "latency_bench.envs.gymnasium_humanoid": "latency_bench.envs.tasks.humanoid",
    "latency_bench.envs.gymnasium_mujoco": "latency_bench.envs.tasks.inverted_pendulum",
    "latency_bench.envs.gymnasium_pendulum": "latency_bench.envs.tasks.pendulum",
    "latency_bench.envs.gymnasium_swimmer": "latency_bench.envs.tasks.swimmer",
    "latency_bench.envs.gymnasium_walker2d": "latency_bench.envs.tasks.walker2d",
    "latency_bench.envs.humanoidbench_sonic": "latency_bench.envs.tasks.humanoidbench.sonic",
    "latency_bench.envs.myochallenge_table_tennis": "latency_bench.envs.tasks.myochallenge_table_tennis",
    "latency_bench.envs.myosuite": "latency_bench.envs.tasks.myosuite",
}


def registration_module_name(module_name: str) -> str:
    """Resolve renamed task modules stored in pre-release checkpoint contracts.

    Only these module paths changed. Preserve custom modules and all task,
    action, state, and timing fields; unknown imports still fail normally.
    """
    if module_name in _PRE_RELEASE_REGISTRATION_MODULES:
        return _PRE_RELEASE_REGISTRATION_MODULES[module_name]
    return module_name


def gymnasium_discrete_action_labels(env_config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(env_config["action_order"])


def gymnasium_discrete_action_values(env_config: Mapping[str, Any]) -> tuple[Any, ...]:
    action_map = env_config["action_map"]
    return tuple(action_map[label] for label in gymnasium_discrete_action_labels(env_config))


def gymnasium_noop_action_id(env_config: Mapping[str, Any]) -> int:
    return gymnasium_discrete_action_labels(env_config).index(env_config["noop_action"])


def gymnasium_action_space_contract(env_config: Mapping[str, Any]) -> dict[str, Any]:
    if "action_space" in env_config:
        return dict(env_config["action_space"])
    return {
        "type": "discrete",
        "labels": list(gymnasium_discrete_action_labels(env_config)),
        "values": list(gymnasium_discrete_action_values(env_config)),
    }


def gymnasium_task_contract(env_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable task semantics shared by training, data, and evaluation."""
    contract = {
        "task_name": env_config["task_name"],
        "env_id": env_config["env_id"],
        "make_kwargs": env_config["make_kwargs"],
        "registration_imports": env_config["registration_imports"],
        "base_prompt": env_config["base_prompt"],
        "env_fps": env_config["env_fps"],
        "obs_fps": env_config["obs_fps"],
        "frame_stack": env_config["frame_stack"],
    }
    if "state_space" in env_config:
        contract["state_space"] = env_config["state_space"]
    action_space = gymnasium_action_space_contract(env_config)
    if action_space["type"] == "discrete":
        contract.update(
            {
                "action_labels": list(gymnasium_discrete_action_labels(env_config)),
                "action_values": list(gymnasium_discrete_action_values(env_config)),
                "noop_action_id": gymnasium_noop_action_id(env_config),
            }
        )
    else:
        contract.update(
            {
                "action_space": action_space,
                "noop_action": env_config["noop_action"],
            }
        )
    return contract
