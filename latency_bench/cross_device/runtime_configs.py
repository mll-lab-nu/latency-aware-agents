from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.core.config import load_config


def calibration_config(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    config = _base_config(spec, output_dir)
    _patch_realtime(config, spec, deployment)
    return config


def measure_config(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    output_dir: Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    config = calibration_config(spec, deployment, output_dir)
    config["env"]["env_fps"] = calibration["selected_env_fps"]
    config["env"]["obs_fps"] = calibration["selected_obs_fps"]
    return config


def simulation_config(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    profile_path: Path,
    profile_ref: Path,
    profile: dict[str, Any],
    method: str,
    latency_seed: int,
    hf_commit: str,
) -> dict[str, Any]:
    config = _base_config(spec, output_dir)
    base_env_fps = config["env"]["env_fps"]
    observation_stride = base_env_fps / config["env"]["obs_fps"]
    config["env"]["env_fps"] = profile["env_fps"]
    config["env"]["obs_fps"] = profile["env_fps"] / observation_stride
    config["env"].update(spec["simulation_env"])
    config["executor"] = {
        "mode": "simulated",
        "simulated_worker_capacity": len(profile["per_slot_summary"]),
        "simulated_inference_pool": True,
    }
    config["latency"] = {
        "method": method,
        "profile_path": str(profile_path),
        "profile_ref": profile_ref.as_posix(),
        "seed": latency_seed,
        "model_id": profile["model_id"],
        "gpu_class": profile["gpu_class"],
        "workload_id": profile["workload_id"],
        "instance_id": profile["instance_id"],
        "source_run_id": profile["source_run_id"],
        "hf_commit": hf_commit,
    }
    if method == "fixed":
        config["latency"]["fixed_latency_ms"] = profile["summary"]["mean_ms"]
    elif method == "normal":
        config["latency"]["mean_latency_ms"] = profile["summary"]["mean_ms"]
        config["latency"]["std_latency_ms"] = profile["summary"]["std_ms"]
    config["evaluation"]["eval_parallel_envs"] = spec["simulation_parallel_envs"]
    config["evaluation"]["eval_latency_values"] = None
    _enable_records(config)
    return config


def write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def eval_command(spec: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.run",
        "--config",
        str(config_path),
        "--checkpoint-kind",
        spec["checkpoint_kind"],
        "--output-dir",
        str(output_dir),
    ]


def _base_config(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = deepcopy(load_config(spec["base_config"]))
    config["policy"]["checkpoint_path"] = spec["checkpoint_path"]
    config["experiment"]["seed"] = spec["evaluation_seed"]
    config["evaluation"]["eval_episodes"] = spec["n_episodes"]
    config["logging"]["output_dir"] = str(output_dir)
    return config


def _patch_realtime(
    config: dict[str, Any],
    spec: dict[str, Any],
    deployment: dict[str, Any],
) -> None:
    config["executor"] = {"mode": "realtime"}
    config["latency"] = {
        "method": "zero",
        "device": "cuda:0",
        "sync_cuda": True,
        "model_id": spec["model_id"],
        "gpu_class": deployment["gpu_class"],
        "workload_id": spec["workload_id"],
        "instance_id": deployment["instance_id"],
    }
    config["evaluation"]["eval_parallel_envs"] = 1
    config["evaluation"]["eval_latency_values"] = None
    config["logging"]["realtime_pipeline_profile"] = True
    _enable_records(config)


def _enable_records(config: dict[str, Any]) -> None:
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = True
    config["logging"]["save_latency_records"] = True
