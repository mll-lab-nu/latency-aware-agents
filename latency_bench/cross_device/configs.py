"""Generated config and command construction for protocol tasks.

The protocol always writes a concrete YAML config before running evaluation so
that measured and replayed runs can be audited after the fact.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.core.config import load_config
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.cross_device.instances import gpu_uuid_hash_from_instance_id
from latency_bench.cross_device.planning import ProtocolTask


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

    if task.stage == "measure":
        _patch_realtime_measured_config(config, task)
    elif task.stage in {"sim_same", "sim_cross"}:
        _patch_simulated_replay_config(config, task, spec)
    else:
        raise ValueError(f"Unsupported stage: {task.stage}")
    return config


def _row_base_config(row: dict[str, Any]) -> dict[str, Any]:
    """Load a row base config and apply row-level protocol overrides.

    Args:
        row: Protocol row containing base config and policy/env ids.

    Returns:
        A mutable config with only row-level overrides applied. Task-level
        fields such as seed, episode count, and output dir are patched by
        `_generated_config`.
    """
    base_config = _base_config_path(row)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config does not exist for {row.get('policy')} / {row.get('env')}: {base_config}")
    config = deepcopy(load_config(base_config))
    config.setdefault("policy", {})["type"] = row["policy_type"]
    config.setdefault("env", {})["name"] = row["env_name"]
    return config


def _patch_realtime_measured_config(config: dict[str, Any], task: ProtocolTask) -> None:
    config.setdefault("executor", {})["mode"] = "realtime"
    latency = {
        "device": "cuda:0",
        "sync_cuda": True,
        "profile_id": task.target_device["profile_id"],
        "target_device": task.target_device["id"],
    }
    if task.instance_id is not None:
        latency["instance_id"] = task.instance_id
        latency["instance_origin"] = task.instance_origin
    config["latency"] = latency
    evaluation = config.setdefault("evaluation", {})
    # Realtime-vs-replay profiles use serial inference semantics.
    evaluation["eval_parallel_envs"] = 1
    evaluation["eval_latency_values"] = None
    evaluation.pop("eval_suites", None)
    logging = config.setdefault("logging", {})
    logging["save_step_records"] = True
    logging["save_action_records"] = True
    logging["save_latency_records"] = True
    # Record per-stage pipeline timing so e2e_latencies.jsonl carries the
    # continuous e2e (inference + overhead) and bench_overhead_ms, instead of
    # falling back to the frame-quantized applied delay.
    logging["realtime_pipeline_profile"] = True


def _patch_simulated_replay_config(config: dict[str, Any], task: ProtocolTask, spec: dict[str, Any]) -> None:
    if task.target_profile_path is None:
        raise ValueError("Simulated replay task requires a profile_path")
    profile = load_profile_manifest(task.target_profile_path)
    _patch_env_from_profile(config, task.target_profile_path)
    if "sim_env" in task.row:
        config["env"].update(task.row["sim_env"])
    config.setdefault("executor", {})["mode"] = "simulated"
    config["executor"]["simulated_worker_capacity"] = profile["inference_worker_count"]
    latency = {
        "type": "profile_sample",
        "profile_path": str(task.target_profile_path),
        "seed": int(task.seed),
        "sampler": "profile_default",
        "profile_id": task.target_device["profile_id"],
        "target_device": task.target_device["id"],
    }
    if task.replay_device is not None:
        latency["replay_host"] = task.replay_device["id"]
    config["latency"] = latency
    evaluation = config.setdefault("evaluation", {})
    # Simulated replay supports batched multi-env inference (factory routes
    # eval_parallel_envs>1 to the batched executor for simulated mode only).
    evaluation["eval_parallel_envs"] = int(spec["sim_parallel_envs"])
    evaluation["eval_latency_values"] = None
    evaluation.pop("eval_suites", None)
    logging = config.setdefault("logging", {})
    logging["save_step_records"] = True
    logging["save_action_records"] = True
    logging["save_latency_records"] = True


def _patch_env_from_profile(config: dict[str, Any], profile_path: Path) -> None:
    profile = load_profile_manifest(profile_path)
    base_env_fps = float(config["env"]["env_fps"])
    base_obs_fps = float(config["env"]["obs_fps"])
    observation_stride = base_env_fps / base_obs_fps
    selected_env_fps = float(profile["env_fps"])
    config["env"]["env_fps"] = selected_env_fps
    config["env"]["obs_fps"] = selected_env_fps / observation_stride


def _write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _eval_command(task: ProtocolTask) -> list[str]:
    cmd = [
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
    return cmd


def _profile_command(
    task: ProtocolTask,
    spec: dict[str, Any],
    overwrite: bool,
    generated_config: dict[str, Any],
) -> list[str]:
    """Build the command that converts measured latencies into a profile.

    Args:
        task: Measure-stage task that produced `infer_latencies.jsonl` and `e2e_latencies.jsonl`.
        spec: Loaded protocol spec used to reconstruct generated config metadata.
        overwrite: Whether profile artifacts may be overwritten.
        generated_config: Exact config written for the measured run.

    Returns:
        A subprocess argument vector for `build_latency_profile_from_trace`.
    """
    if task.profile_output_dir is None:
        raise ValueError("Measure task requires profile_output_dir")
    metadata = {
        "policy_type": task.row["policy_type"],
        "model_type": task.row["model_type"],
    }
    policy_cfg = generated_config["policy"]
    if "checkpoint_path" in policy_cfg:
        metadata["checkpoint_path"] = str(policy_cfg["checkpoint_path"])
    if task.instance_id is not None:
        metadata["hardware_class"] = str(task.target_device["id"])
        metadata["instance_id"] = task.instance_id
        metadata["instance_origin"] = task.instance_origin
        if task.instance_origin == "gpu_uuid_hash":
            metadata["gpu_uuid_hash"] = gpu_uuid_hash_from_instance_id(task.instance_id)
    cmd = [
        sys.executable,
        "-m",
        "latency_bench.experiments.build_latency_profile_from_trace",
        "--latencies",
        str(task.output_dir / "e2e_latencies.jsonl"),
        "--latency-kind",
        "e2e_latency",
        "--inference-latencies",
        str(task.output_dir / "infer_latencies.jsonl"),
        "--episode-metrics",
        str(task.output_dir / "episode_metrics.jsonl"),
        "--actions",
        str(task.output_dir / "actions.jsonl"),
        "--steps",
        str(task.output_dir / "steps.jsonl"),
        "--output-dir",
        str(task.profile_output_dir),
        "--profile-id",
        str(task.target_device["profile_id"]),
        "--policy-id",
        str(task.row["policy"]),
        "--env-id",
        str(task.row["env_name"]),
        "--target-device",
        str(task.target_device["id"]),
        "--env-fps",
        str(generated_config["env"]["env_fps"]),
        "--metadata-json",
        json.dumps(metadata, sort_keys=True),
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _base_config_path(row: dict[str, Any]) -> Path:
    return Path(str(row["base_config"]))
