"""Generated config and command construction for protocol tasks.

The protocol always writes a concrete YAML config before running evaluation so
that measured and replayed runs can be audited after the fact.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.core.config import load_config
from scripts.cross_device.models import ProtocolTask


OPENVLA_POLICY_TYPES = {
    "flappy_openvla",
    "demon_attack_openvla",
    "deadly_corridor_openvla",
}
OPENVLA_CHECKPOINT_ENV_BY_POLICY = {
    "flappy_openvla": "FLAPPY_OPENVLA_CHECKPOINT_PATH",
    "demon_attack_openvla": "DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH",
    "deadly_corridor_openvla": "DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH",
}


def _generated_config(task: ProtocolTask, spec: dict[str, Any]) -> dict[str, Any]:
    """Create the exact evaluation config for one protocol task.

    Args:
        task: Concrete task with stage, seed, device, and output paths.
        spec: Loaded protocol spec.

    Returns:
        A config dictionary ready to dump to YAML and pass to `latency_bench.run`.
    """
    config = _row_base_config(task.row)
    config["experiment"]["seed"] = int(task.seed)
    config["evaluation"]["eval_episodes"] = int(task.n_episodes)
    if task.smoke_max_steps is not None:
        config["evaluation"]["eval_max_steps"] = int(task.smoke_max_steps)
    config.setdefault("logging", {})["output_dir"] = str(task.output_dir)

    if task.stage in {"measure", "realtime"}:
        _patch_realtime_measured_config(config, task)
    elif task.stage in {"sim_same", "sim_cross"}:
        _patch_simulated_replay_config(config, task, spec)
    else:
        raise ValueError(f"Unsupported stage: {task.stage}")
    return config


def _row_base_config(row: dict[str, Any]) -> dict[str, Any]:
    """Load a row base config and apply row-level protocol overrides.

    Args:
        row: Protocol row containing base config, policy/env ids, and FPS.

    Returns:
        A mutable config with only row-level overrides applied. Task-level
        fields such as seed, episode count, and output dir are patched by
        `_generated_config`.
    """
    base_config = _base_config_path(row)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config does not exist for {row.get('policy')} / {row.get('env')}: {base_config}")
    config = deepcopy(load_config(base_config))
    config.setdefault("policy", {})["type"] = row.get("policy_type", config.get("policy", {}).get("type"))
    config.setdefault("env", {})["name"] = row.get("env_name", config.get("env", {}).get("name"))
    if row.get("fps") is not None:
        fps = _config_fps_value(row["fps"])
        config["env"]["env_fps"] = fps
        config["env"]["obs_fps"] = fps
    _apply_openvla_env_overrides(config)
    return config


def _config_fps_value(raw: Any) -> int | float:
    fps = float(raw)
    return int(fps) if fps.is_integer() else fps


def _apply_openvla_env_overrides(config: dict[str, Any]) -> None:
    """Apply local OpenVLA model/checkpoint environment overrides.

    Args:
        config: Mutable evaluation config. `policy.base_model_path` and
        `policy.checkpoint_path` are updated when the matching env vars exist.

    Side effects:
        Mutates only OpenVLA policy configs; small-policy configs are untouched.
    """
    policy = config.setdefault("policy", {})
    policy_type = _policy_type(config)
    if policy_type not in OPENVLA_POLICY_TYPES:
        return
    base_model_path = os.environ.get("OPENVLA_BASE_MODEL_PATH")
    if base_model_path:
        policy["base_model_path"] = base_model_path
    checkpoint_env = OPENVLA_CHECKPOINT_ENV_BY_POLICY.get(policy_type)
    if checkpoint_env:
        checkpoint_path = os.environ.get(checkpoint_env)
        if checkpoint_path:
            policy["checkpoint_path"] = checkpoint_path


def _policy_type(config: dict[str, Any]) -> str:
    return str(config.get("policy", {}).get("type", "")).strip().lower()


def _patch_realtime_measured_config(config: dict[str, Any], task: ProtocolTask) -> None:
    config.setdefault("executor", {})["mode"] = "realtime"
    latency = config.setdefault("latency", {})
    latency.pop("type", None)
    latency["device"] = latency.get("device") or "cuda:0"
    latency["sync_cuda"] = latency.get("sync_cuda", True)
    latency["profile_id"] = task.target_device["profile_id"]
    latency["target_device"] = task.target_device["id"]
    latency.pop("fixed_latency_ms", None)
    latency.pop("profile_path", None)
    evaluation = config.setdefault("evaluation", {})
    # Realtime-vs-replay profiles use serial inference semantics.
    evaluation["eval_parallel_envs"] = 1
    evaluation["eval_latency_values"] = None


def _patch_simulated_replay_config(config: dict[str, Any], task: ProtocolTask, spec: dict[str, Any]) -> None:
    if task.profile_path is None:
        raise ValueError("Simulated replay task requires a profile_path")
    config.setdefault("executor", {})["mode"] = "simulated"
    latency = config.setdefault("latency", {})
    latency["type"] = "profile_sample"
    latency["profile_path"] = str(task.profile_path)
    latency["seed"] = int(task.seed)
    latency["profile_id"] = task.target_device["profile_id"]
    latency["target_device"] = task.target_device["id"]
    if task.replay_device is not None:
        latency["replay_host"] = task.replay_device["id"]
    for key in ("fixed_latency_ms", "mean_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms", "trace_path"):
        latency.pop(key, None)
    evaluation = config.setdefault("evaluation", {})
    # Realtime-vs-replay profiles use serial inference semantics.
    evaluation["eval_parallel_envs"] = 1
    evaluation["eval_latency_values"] = None


def _write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _eval_command(task: ProtocolTask) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.run",
        "--config",
        str(task.generated_config_path),
        "--checkpoint-kind",
        task.checkpoint_kind,
        "--output-dir",
        str(task.output_dir),
    ]


def _profile_command(task: ProtocolTask, spec: dict[str, Any], overwrite: bool) -> list[str]:
    """Build the command that converts measured latencies into a profile.

    Args:
        task: Measure-stage task that produced `latencies.jsonl`.
        spec: Loaded protocol spec used to reconstruct generated config metadata.
        overwrite: Whether profile artifacts may be overwritten.

    Returns:
        A subprocess argument vector for `build_latency_profile_from_trace`.
    """
    if task.profile_output_dir is None:
        raise ValueError("Measure task requires profile_output_dir")
    cmd = [
        sys.executable,
        "-m",
        "latency_bench.experiments.build_latency_profile_from_trace",
        "--latencies",
        str(task.output_dir / "latencies.jsonl"),
        "--output-dir",
        str(task.profile_output_dir),
        "--profile-id",
        str(task.target_device["profile_id"]),
        "--policy-id",
        str(task.row.get("policy")),
        "--env-id",
        str(task.row.get("env_name", task.row.get("env"))),
        "--target-device",
        str(task.target_device["id"]),
        "--env-fps",
        str(task.row["fps"]),
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _base_config_path(row: dict[str, Any]) -> Path:
    """Return the base config path required by one protocol row.

    Args:
        row: Protocol row containing `base_config`.

    Returns:
        Path to the row's base YAML config.
    """
    value = row.get("base_config")
    if value is None or str(value).strip().upper().startswith("TODO"):
        raise FileNotFoundError(f"Missing base_config for {row.get('policy')} / {row.get('env')}: {value}")
    return Path(str(value))
