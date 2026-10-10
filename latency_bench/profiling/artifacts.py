"""Define local and published coordinates for profile, measurement and replay artifacts."""

from __future__ import annotations

from pathlib import Path


def profile_ref(model_id: str, gpu_class: str, workload_id: str, instance_id: str) -> Path:
    return Path(model_id) / gpu_class / workload_id / instance_id


def run_ref(profile: str | Path, run_id: str) -> Path:
    return Path(profile) / run_id


def run_dir(artifact_root: str | Path, run: str | Path) -> Path:
    return Path(artifact_root) / "runs" / run


def profile_dir(artifact_root: str | Path, profile: str | Path) -> Path:
    return Path(artifact_root) / "profiles" / profile


def simulation_seed_dir(
    artifact_root: str | Path,
    profile: str | Path,
    run_id: str,
    method: str,
    latency_seed: int,
) -> Path:
    return Path(artifact_root) / "simulations" / profile / run_id / method / str(latency_seed)
