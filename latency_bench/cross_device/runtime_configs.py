from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.core.config import load_config


def calibration_config(
    workload: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    config = _base_config(workload, output_dir)
    _patch_realtime(config, device, instance_id)
    return config


def measure_config(
    workload: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    output_dir: Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    config = calibration_config(workload, device, instance_id, output_dir)
    config["env"]["env_fps"] = calibration["selected_env_fps"]
    config["env"]["obs_fps"] = calibration["selected_obs_fps"]
    config["measurement"] = deepcopy(workload["measurement"])
    return config


def simulation_config(
    workload: dict[str, Any],
    output_dir: Path,
    profile_path: Path,
    profile_ref: Path,
    profile: dict[str, Any],
    method: str,
    latency_seed: int,
) -> dict[str, Any]:
    config = _base_config(workload, output_dir)
    workload_config = workload["workload"]
    base_env_fps = config["env"]["env_fps"]
    observation_stride = base_env_fps / config["env"]["obs_fps"]
    config["env"]["env_fps"] = profile["env_fps"]
    config["env"]["obs_fps"] = profile["env_fps"] / observation_stride
    config["env"].update(workload_config["sim_env"])
    config["executor"] = {
        "mode": "simulated",
        "simulated_worker_capacity": profile["inference_worker_count"],
        "simulated_inference_pool": True,
    }
    config["latency"] = {
        "method": method,
        "profile_path": str(profile_path),
        "profile_ref": profile_ref.as_posix(),
        "seed": latency_seed,
        "profile_id": profile["profile_id"],
        "target_device": profile["device_id"],
        "instance_id": profile["instance_id"],
        "profile_revision": profile["profile_revision"],
    }
    if method == "fixed":
        config["latency"]["fixed_latency_ms"] = profile["summary"]["mean_ms"]
    elif method == "normal":
        config["latency"]["mean_latency_ms"] = profile["summary"]["mean_ms"]
        config["latency"]["std_latency_ms"] = profile["summary"]["std_ms"]
    config["evaluation"]["eval_parallel_envs"] = workload["sim_parallel_envs"]
    config["evaluation"]["eval_latency_values"] = None
    _enable_records(config)
    return config


def write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def eval_command(
    workload: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.run",
        "--config",
        str(config_path),
        "--checkpoint-kind",
        workload["checkpoint_kind"],
        "--output-dir",
        str(output_dir),
    ]


def _base_config(workload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    workload_config = workload["workload"]
    config = deepcopy(load_config(workload_config["base_config"]))
    config["experiment"]["seed"] = workload["evaluation_seed"]
    config["policy"]["type"] = workload_config["policy_type"]
    config["env"]["name"] = workload_config["env_name"]
    config["evaluation"]["eval_episodes"] = workload["n_episodes"]
    config["logging"]["output_dir"] = str(output_dir)
    return config


def _patch_realtime(
    config: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
) -> None:
    config["executor"] = {"mode": "realtime"}
    config["latency"] = {
        "method": "zero",
        "device": "cuda:0",
        "sync_cuda": True,
        "profile_id": device["profile_id"],
        "target_device": device["id"],
        "instance_id": instance_id,
    }
    config["evaluation"]["eval_parallel_envs"] = 1
    config["evaluation"]["eval_latency_values"] = None
    config["logging"]["realtime_pipeline_profile"] = True
    _enable_records(config)


def _enable_records(config: dict[str, Any]) -> None:
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = True
    config["logging"]["save_latency_records"] = True
