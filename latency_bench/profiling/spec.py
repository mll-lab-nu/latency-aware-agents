"""Resolve profiling options on an ordinary evaluation configuration."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.eval.config import resolve_evaluation_config


def path_segment(value: str) -> str:
    """Validate an artifact coordinate supplied by a config or command line."""
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Expected one path segment, got {value!r}")
    return value


def resolve_profile_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy and resolve a run config without constructing its environment or policy.

    Calibration and video recording are disabled by default.
    Filesystem paths follow evaluation's caller-working-directory convention.
    """
    config = deepcopy(config)
    resolve_evaluation_config(config)
    options = config["profiling"]
    for name in ("model_id", "workload_id"):
        path_segment(options[name])
    if options["repeats"] < 1:
        raise ValueError("profiling.repeats must be positive")
    if "calibration" not in options:
        options["calibration"] = None
    if "video" not in config["logging"]:
        config["logging"]["video"] = {"enabled": False}
    if "eval_entrypoint" in options:
        options["eval_entrypoint"] = os.path.abspath(Path(options["eval_entrypoint"]).expanduser())
    if not config["executor"]["inference_devices"]:
        raise ValueError("executor.inference_devices must select at least one worker")
    if config["evaluation"]["eval_parallel_envs"] != 1:
        raise ValueError("Profiling requires evaluation.eval_parallel_envs: 1")
    if config["evaluation"]["eval_latency_values"] is not None or config["evaluation"].get("eval_suites"):
        raise ValueError("Profiling requires one workload, without evaluation sweeps or suites")
    return config
