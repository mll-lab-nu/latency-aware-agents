"""Resolve registered model/workload presets and GPU names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config


def load_profile_spec(
    path: str | Path,
    model_id: str,
    workload_id: str,
) -> dict[str, Any]:
    registry = load_config(path)
    preset = registry["models"][model_id]["workloads"][workload_id]
    return {
        "artifact_root": registry["artifact_root"],
        "hf_repo": registry["hf_repo"],
        "gpu_models": registry["gpu_models"],
        "model_id": model_id,
        "workload_id": workload_id,
        **preset,
    }


def gpu_name_registry(gpu_models: dict[str, Any]) -> dict[str, str]:
    return {
        alias: gpu_id
        for gpu_id, gpu in gpu_models.items()
        for alias in gpu["aliases"]
    }
