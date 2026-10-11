"""Resolve profiling options on an ordinary evaluation configuration."""

from __future__ import annotations

from latency_bench.core.config import resolve_runtime_config

import os
from copy import deepcopy
from pathlib import Path
from typing import Any




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
    resolve_runtime_config(config)
    options = config["profiling"]
    for name in ("model_id", "workload_id"):
        path_segment(options[name])
    if "calibration" not in options:
        options["calibration"] = None
    if "eval_entrypoint" in options:
        options["eval_entrypoint"] = os.path.abspath(Path(options["eval_entrypoint"]).expanduser())
    if config["executor"]["parallel_envs"] != 1:
        raise ValueError("Profiling requires executor.parallel_envs: 1")
    if config["evaluation"]["suites"]:
        raise ValueError("Profiling requires one workload, without evaluation sweeps or suites")
    return config
