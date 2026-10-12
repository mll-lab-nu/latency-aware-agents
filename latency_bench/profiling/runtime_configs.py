"""Build measurement configs from the user's evaluation configuration."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    config["logging"].update(
        output_dir=str(output_dir), save_step_records=True,
        save_action_records=True, save_latency_records=True,
    )
    return config


def evaluation_environment() -> dict[str, str]:
    """Use the profiling subprocess start method selected by the caller."""
    environment = dict(os.environ)
    environment.setdefault(START_METHOD_ENV, "spawn")
    return environment


def eval_command(options: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    """Use the normal evaluator or an explicitly configured task entrypoint."""
    if "eval_entrypoint" in options:
        return [sys.executable, options["eval_entrypoint"], "latency-eval", "--eval-config", str(config_path)]
    command = [sys.executable, "-m", "latency_bench", "eval", "--config", str(config_path),
               "--output-dir", str(output_dir)]
    return command
