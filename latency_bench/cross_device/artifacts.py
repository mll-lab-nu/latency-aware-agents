from __future__ import annotations

from pathlib import Path


def profile_ref(model_id: str, gpu_class: str, workload_id: str, instance_id: str) -> Path:
    return Path(model_id) / gpu_class / workload_id / instance_id


def run_ref(profile: str | Path, run_id: str) -> Path:
    return Path(profile) / run_id


def run_dir(artifact_root: str | Path, run: str | Path) -> Path:
    return Path(artifact_root) / "runs" / run


def run_session_dir(artifact_root: str | Path, run: str | Path, session_id: int) -> Path:
    return run_dir(artifact_root, run) / "sessions" / str(session_id)


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
