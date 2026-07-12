from __future__ import annotations

from pathlib import Path


def target_ref(device_id: str, instance_id: str, workload_id: str) -> Path:
    return Path(device_id) / instance_id / workload_id


def raw_workload_dir(artifact_root: str | Path, target: str | Path) -> Path:
    return Path(artifact_root) / "raw" / target


def raw_session_dir(
    artifact_root: str | Path,
    target: str | Path,
    session_id: int,
) -> Path:
    return raw_workload_dir(artifact_root, target) / "sessions" / str(session_id)


def profile_ref(target: str | Path, profile_revision: str) -> Path:
    return Path(target) / profile_revision


def profile_dir(artifact_root: str | Path, profile: str | Path) -> Path:
    return Path(artifact_root) / "profiles" / profile


def simulation_seed_dir(
    artifact_root: str | Path,
    profile: str | Path,
    method: str,
    latency_seed: int,
) -> Path:
    return Path(artifact_root) / "sim" / profile / method / str(latency_seed)
