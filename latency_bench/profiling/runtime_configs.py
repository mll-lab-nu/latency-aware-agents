"""Build measurement and replay configs from the user's evaluation configuration."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from latency_bench.executors.env_step_backend import START_METHOD_ENV


def measure_config(
    config: dict[str, Any],
    deployment: dict[str, Any],
    output_dir: Path,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select recording output without replacing the configured deployment."""
    config = deepcopy(config)
    if calibration is not None:
        config["env"]["env_fps"] = calibration["selected_env_fps"]
        config["env"]["obs_fps"] = calibration["selected_obs_fps"]
    config.setdefault("measurement", {}).update(
        model_id=config["profiling"]["model_id"],
        workload_id=config["profiling"]["workload_id"],
        gpu_class=deployment["gpu_class"],
        instance_id=deployment["instance_id"],
    )
    _enable_records(config, output_dir)
    return config


def set_env_clock_preserving_observation_stride(
    config: dict[str, Any], env_fps: float,
) -> None:
    """Change the simulation rate while keeping the configured observation stride."""
    observation_stride = config["env"]["env_fps"] / config["env"]["obs_fps"]
    config["env"]["env_fps"] = env_fps
    config["env"]["obs_fps"] = env_fps / observation_stride


def apply_profile_latency_summary(
    latency_config: dict[str, Any], method: str, profile: dict[str, Any],
) -> None:
    """Add fixed or normal latency parameters from the selected profile summary."""
    if method == "fixed":
        latency_config["fixed_latency_ms"] = profile["summary"]["mean_ms"]
    elif method == "normal":
        latency_config["mean_latency_ms"] = profile["summary"]["mean_ms"]
        latency_config["std_latency_ms"] = profile["summary"]["std_ms"]


def simulation_config(
    config: dict[str, Any],
    *,
    output_dir: Path,
    profile_path: Path,
    profile: dict[str, Any],
    method: str,
    latency_seed: int,
) -> dict[str, Any]:
    """Replay measured timing at the target task's explicitly configured clock."""
    config = deepcopy(config)
    config["executor"].update(
        mode="simulated",
        simulated_worker_capacity=len(profile["per_slot_summary"]),
        inference_pool=True,
    )
    config["latency"] = {
        "method": method,
        "profile_path": str(profile_path),
        "seed": latency_seed,
        **{key: profile[key] for key in (
            "model_id", "gpu_class", "workload_id", "instance_id", "source_run_id",
        )},
    }
    apply_profile_latency_summary(config["latency"], method, profile)
    _enable_records(config, output_dir)
    return config


def write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def evaluation_environment() -> dict[str, str]:
    """Use the profiling subprocess start method selected by the caller."""
    environment = dict(os.environ)
    environment.setdefault(START_METHOD_ENV, "spawn")
    return environment


def eval_command(options: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    """Use the normal evaluator or an explicitly configured task entrypoint."""
    if "eval_entrypoint" in options:
        return [sys.executable, options["eval_entrypoint"], "latency-eval", "--eval-config", str(config_path)]
    command = [sys.executable, "-m", "latency_bench.run", "--config", str(config_path),
               "--output-dir", str(output_dir)]
    if "checkpoint_kind" in options:
        command.extend(["--checkpoint-kind", options["checkpoint_kind"]])
    return command


def _enable_records(config: dict[str, Any], output_dir: Path) -> None:
    logging = config["logging"]
    logging.update(output_dir=str(output_dir), save_step_records=True,
                   save_action_records=True, save_latency_records=True)
