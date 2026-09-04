from __future__ import annotations

import fcntl
import shutil
from pathlib import Path
from typing import Any


PROFILE_FILENAMES = (
    "profile.json",
    "latency_distribution.json",
    "latency_burst_model.json",
    "hardware.json",
    "DONE",
)
LOCAL_ONLY_RUN_FILES = {
    "calibration.generated.yaml",
    "generated.yaml",
    "hardware.json",
    "run_contract.json",
}


def publish_run_and_profile(
    spec: dict[str, Any],
    *,
    local_run_dir: Path,
    local_profile_dir: Path,
    run_ref: Path,
    profile_ref: Path,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [
        CommitOperationAdd(
            path_in_repo=f"runs/{run_ref.as_posix()}/{path.relative_to(local_run_dir).as_posix()}",
            path_or_fileobj=str(path),
        )
        for path in sorted(local_run_dir.rglob("*"))
        if path.is_file()
        and path.name not in LOCAL_ONLY_RUN_FILES
        and path.relative_to(local_run_dir).parts[0] != "profile"
    ]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=f"profiles/{profile_ref.as_posix()}/{name}",
            path_or_fileobj=str(local_profile_dir / name),
        )
        for name in PROFILE_FILENAMES
    )
    commit = HfApi().create_commit(
        repo_id=spec["hf_repo"],
        repo_type="dataset",
        operations=operations,
        commit_message=f"profile {profile_ref.as_posix()} from {run_ref.name}",
    )
    return commit.oid


def download_profile(
    spec: dict[str, Any],
    profile_ref: Path,
    *,
    hf_commit: str,
    local_dir: Path,
) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    remote_dir = Path("profiles") / profile_ref
    resolved_commit = HfApi().repo_info(
        repo_id=spec["hf_repo"],
        repo_type="dataset",
        revision=hf_commit,
    ).sha
    snapshot = Path(
        snapshot_download(
            repo_id=spec["hf_repo"],
            repo_type="dataset",
            revision=resolved_commit,
            allow_patterns=[f"{remote_dir.as_posix()}/{name}" for name in PROFILE_FILENAMES],
        )
    )
    source_dir = snapshot / remote_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = local_dir / resolved_commit
    with (local_dir / f".{resolved_commit}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (profile_dir / "DONE").exists():
            profile_dir.mkdir(exist_ok=True)
            for name in PROFILE_FILENAMES:
                shutil.copyfile(source_dir / name, profile_dir / name)
        (profile_dir / "DONE").read_bytes()
    return profile_dir, resolved_commit


def publish_simulation_dirs(
    spec: dict[str, Any],
    *,
    local_dirs: list[Path],
    remote_dirs: list[Path],
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [
        CommitOperationAdd(
            path_in_repo=f"{remote.as_posix()}/{path.relative_to(local).as_posix()}",
            path_or_fileobj=str(path),
        )
        for local, remote in zip(local_dirs, remote_dirs)
        for path in sorted(local.rglob("*"))
        if path.is_file() and path.name != "generated.yaml"
    ]
    commit = HfApi().create_commit(
        repo_id=spec["hf_repo"],
        repo_type="dataset",
        operations=operations,
        commit_message=f"simulations under {remote_dirs[0].parent.as_posix()}",
    )
    return commit.oid
