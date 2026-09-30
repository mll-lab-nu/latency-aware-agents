from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.core.config import load_config
from latency_bench.eval.config import apply_runtime_overrides


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


def _set_env_clock_preserving_observation_stride(
    config: dict[str, Any], env_fps: float,
) -> None:
    """Change the simulation rate while keeping the configured observation stride."""
    observation_stride = config["env"]["env_fps"] / config["env"]["obs_fps"]
    config["env"]["env_fps"] = env_fps
    config["env"]["obs_fps"] = env_fps / observation_stride


def _apply_profile_latency_summary(
    latency_config: dict[str, Any], method: str, profile: dict[str, Any],
) -> None:
    """Add fixed or normal latency parameters from the selected profile summary."""
    if method == "fixed":
        latency_config["fixed_latency_ms"] = profile["summary"]["mean_ms"]
    elif method == "normal":
        latency_config["mean_latency_ms"] = profile["summary"]["mean_ms"]
        latency_config["std_latency_ms"] = profile["summary"]["std_ms"]


def simulation_config(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    profile_path: Path,
    profile_ref: Path,
    profile: dict[str, Any],
    method: str,
    latency_seed: int,
) -> dict[str, Any]:
    config = _base_config(spec, output_dir)
    _set_env_clock_preserving_observation_stride(config, profile["env_fps"])
    config["env"].update(spec["simulation_env"])
    config["executor"] = {
        "mode": "simulated",
        "simulated_worker_capacity": len(profile["per_slot_summary"]),
        "simulated_inference_pool": True,
        "inference_devices": _inference_devices(
            spec,
            len(profile["per_slot_summary"]),
        ),
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
    }
    _apply_profile_latency_summary(config["latency"], method, profile)
    config["evaluation"]["eval_parallel_envs"] = spec["simulation_parallel_envs"]
    if "inference_batch_size" in spec:
        config["executor"]["inference_batch_size"] = spec["inference_batch_size"]
    config["evaluation"]["eval_latency_values"] = None
    _enable_records(config)
    return config


def write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def eval_command(spec: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    if "eval_entrypoint" in spec:
        return [
            sys.executable,
            spec["eval_entrypoint"],
            "latency-eval",
            "--eval-config",
            str(config_path),
        ]
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
    apply_runtime_overrides(
        config,
        checkpoint_path=spec["checkpoint_path"],
        output_dir=output_dir,
        model_config_path=(
            spec["model_config_path"]
            if "model_config_path" in spec
            else None
        ),
    )
    if "backbone_path" in spec:
        config["policy"]["backbone_path"] = spec["backbone_path"]
    if "runtime_checkpoint_path" in spec:
        config["env"]["runtime_checkpoint_path"] = spec["runtime_checkpoint_path"]
    if spec["simulation_gpu"] is not None:
        config["env"]["simulator_device"] = spec["simulation_gpu"]
    if "model_config_path" in spec and "task_contract_path" in config["policy"]:
        config["policy"]["task_contract_path"] = str(
            Path(spec["model_config_path"]).parent
            / config["policy"]["task_contract_path"]
        )
    if "model_config_path" in spec and "task_manifest_path" in config["policy"]:
        config["policy"]["task_manifest_path"] = str(
            Path(spec["model_config_path"]).parent / config["policy"]["task_manifest_path"]
        )
    config["experiment"]["seed"] = spec["evaluation_seed"]
    config["evaluation"]["eval_episodes"] = spec["eval_episodes"]
    config["provenance"] = {
        key: spec[key]
        for key in ("model_provenance", "runtime_provenance")
        if key in spec
    }
    return config


def _patch_realtime(
    config: dict[str, Any],
    spec: dict[str, Any],
    deployment: dict[str, Any],
) -> None:
    inference_devices = _inference_devices(spec, len(deployment["gpus"]))
    config["executor"] = {
        "mode": "realtime",
        "inference_devices": inference_devices,
    }
    config["latency"] = {
        "method": "measured",
        "device": inference_devices[0],
        "sync_cuda": True,
        "model_id": spec["model_id"],
        "gpu_class": deployment["gpu_class"],
        "workload_id": spec["workload_id"],
        "instance_id": deployment["instance_id"],
    }
    config["evaluation"]["eval_parallel_envs"] = 1
    config["evaluation"]["eval_latency_values"] = None
    config["evaluation"].pop("eval_suites", None)
    config["logging"]["realtime_pipeline_profile"] = True
    _enable_records(config)


def _inference_devices(spec: dict[str, Any], count: int) -> list[str]:
    offset = 0 if spec["simulation_gpu"] is None else 1
    return [f"cuda:{index + offset}" for index in range(count)]


def _enable_records(config: dict[str, Any]) -> None:
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = True
    config["logging"]["save_latency_records"] = True
