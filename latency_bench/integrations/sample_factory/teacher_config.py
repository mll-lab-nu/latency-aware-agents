"""Load rollout descriptors from Sample Factory teacher checkpoints."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


from latency_bench.data.action_layouts import (
    ActionSpec,
    G1_SONIC_ACTION_SPEC,
    action_spec_for_env,
    gymnasium_discrete_action_spec,
    gymnasium_continuous_action_spec,
)
from latency_bench.data.prompts import BASE_PROMPTS
from latency_bench.integrations.sample_factory.runtime import resolve_sample_factory_device


SF_ENV_TO_BENCH_ENV = {
    "doom_deadly_corridor": "deadly_corridor",
    "doom_defend_the_line": "defend_the_line",
    "latency_flappy": "flappy",
    "latency_demon_attack": "demon_attack",
    "deadly_corridor": "deadly_corridor",
    "defend_the_line": "defend_the_line",
    "flappy": "flappy",
    "demon_attack": "demon_attack",
}


GYMNASIUM_SF_ENV = "latency_gymnasium"


def _infer_env_name_from_sf_env(sf_env_name: str) -> str:
    """Infer the benchmark environment name from a Sample Factory env id.

    Args:
        sf_env_name: Environment name stored in Sample Factory config.

    Returns:
        Canonical benchmark environment name.
    """
    normalized = str(sf_env_name).strip()
    if normalized in SF_ENV_TO_BENCH_ENV:
        return SF_ENV_TO_BENCH_ENV[normalized]
    raise ValueError(f"Could not infer benchmark env name from Sample Factory env {sf_env_name!r}")


def _rollout_task_contract(flat_cfg: Mapping[str, Any]) -> tuple[str, str, ActionSpec, str]:
    """Resolve task identity and semantics stored in a teacher checkpoint."""
    if flat_cfg["env"] == GYMNASIUM_SF_ENV:
        action_space = json.loads(flat_cfg["gym_action_space_json"])
        action_labels = action_space["labels"]
        action_spec = (
            G1_SONIC_ACTION_SPEC
            if str(flat_cfg["gym_task_name"]).startswith("g1_sonic_")
            else (
                gymnasium_continuous_action_spec(action_labels)
                if action_space["type"] == "box"
                else gymnasium_discrete_action_spec(action_labels)
            )
        )
        return (
            flat_cfg["gym_task_name"],
            "gymnasium",
            action_spec,
            flat_cfg["gym_base_prompt"],
        )
    env_name = _infer_env_name_from_sf_env(flat_cfg["env"])
    return env_name, env_name, action_spec_for_env(env_name), BASE_PROMPTS[env_name]


def _gymnasium_task_metadata(flat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build published Gymnasium task metadata from its checkpoint contract."""
    metadata = {
        "task_name": flat_cfg["gym_task_name"],
        "env_id": flat_cfg["gym_env_id"],
        "make_kwargs": json.loads(flat_cfg["gym_make_kwargs_json"]),
        "registration_imports": json.loads(flat_cfg["gym_registration_imports_json"]),
        "base_prompt": flat_cfg["gym_base_prompt"],
        "env_fps": flat_cfg["env_fps"],
        "obs_fps": flat_cfg["obs_fps"],
        "frame_stack": flat_cfg["frame_stack"],
    }
    if flat_cfg["gym_state_labels_json"]:
        metadata["state_space"] = {
            "labels": json.loads(flat_cfg["gym_state_labels_json"]),
        }
    action_space = json.loads(flat_cfg["gym_action_space_json"])
    if str(flat_cfg["gym_task_name"]).startswith("g1_sonic_"):
        metadata["action_space"] = {
            "type": action_space["type"],
            "labels": list(G1_SONIC_ACTION_SPEC.labels),
            "dtype": action_space["dtype"],
        }
    elif action_space["type"] == "discrete":
        metadata.update(
            {
                "action_labels": json.loads(flat_cfg["gym_action_labels_json"]),
                "action_values": json.loads(flat_cfg["gym_action_values_json"]),
                "noop_action_id": flat_cfg["gym_noop_action_id"],
            }
        )
    else:
        metadata.update(
            {
                "action_space": action_space,
                "noop_action": json.loads(flat_cfg["gym_noop_action_json"]),
            }
        )
    return metadata


def _resolve_checkpoint_root(checkpoint_root: str | Path) -> tuple[Path, Path | None]:
    """Resolve a checkpoint file or experiment directory.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.

    Returns:
        Experiment directory with optional explicit checkpoint path.
    """
    # HF checkpoints can be symlinks; the config belongs to the snapshot directory.
    path = Path(os.path.abspath(Path(checkpoint_root).expanduser()))
    if path.is_file():
        if path.parent.name.startswith("checkpoint_p"):
            return (path.parent.parent, path)
        return (path.parent, path)
    return (path, None)


def _read_checkpoint_config(experiment_dir: Path) -> dict[str, Any]:
    """Read Sample Factory's flat ``config.json``.

    Args:
        experiment_dir: Sample Factory experiment directory.

    Returns:
        Flat checkpoint config mapping.
    """
    config_path = experiment_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rollout_training_config(
    checkpoint_root: str | Path,
    *,
    device_override: str | None = None,
    gym_make_kwargs_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the rollout descriptor and metadata used for dataset export.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.
        device_override: Optional rollout device override.
        gym_make_kwargs_override: Optional Gymnasium reset/runtime overrides.

    Returns:
        Rollout descriptor and export metadata.
    """
    experiment_dir, checkpoint_path = _resolve_checkpoint_root(checkpoint_root)
    flat_cfg = _read_checkpoint_config(experiment_dir)
    if flat_cfg["env"] == GYMNASIUM_SF_ENV:
        make_kwargs = json.loads(flat_cfg["gym_make_kwargs_json"])
        if gym_make_kwargs_override is not None:
            make_kwargs.update(gym_make_kwargs_override)
        flat_cfg["gym_make_kwargs_json"] = json.dumps(make_kwargs)
    env_name, integration_name, action_spec, base_prompt = _rollout_task_contract(flat_cfg)
    config_source = str((experiment_dir / "config.json").resolve())
    latency_override = None
    latency_source = "checkpoint"

    descriptor = {
        "env_name": env_name,
        "integration_name": integration_name,
        "action_spec": action_spec,
        "base_prompt": base_prompt,
        "checkpoint_experiment_dir": str(experiment_dir),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "flat_cfg": deepcopy(flat_cfg),
        "device_override": None if device_override is None else resolve_sample_factory_device(device_override),
        "latency_override": latency_override,
    }
    metadata = {
        "checkpoint_experiment_dir": str(experiment_dir),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "config_source": config_source,
        "latency_source": latency_source,
        "checkpoint_config": deepcopy(flat_cfg),
        "latency_override": deepcopy(latency_override),
    }
    if integration_name == "gymnasium":
        metadata["gymnasium_task"] = _gymnasium_task_metadata(flat_cfg)
    return descriptor, metadata
