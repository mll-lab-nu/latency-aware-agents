"""Publish complete profiling artifacts and materialize exact remote profiles."""

from __future__ import annotations

import fcntl
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from latency_bench.utils.io import read_json


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
    "resolved_config.yaml",
    "hardware.json",
    "run_contract.json",
}


def publish_run_and_profile(
    repo_id: str,
    *,
    local_run_dir: Path,
    local_profile_dir: Path,
    run_ref: Path,
    profile_ref: Path,
) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    profile_filenames = _profile_filenames(local_profile_dir)
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
        for name in profile_filenames
    )
    HfApi().create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"profile {profile_ref.as_posix()} from {run_ref.name}",
    )


def download_profile(
    repo_id: str,
    profile_ref: Path,
    *,
    local_dir: Path,
) -> Path:
    from huggingface_hub import snapshot_download

    remote_dir = Path("profiles") / profile_ref
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{remote_dir.as_posix()}/{name}" for name in PROFILE_FILENAMES],
        )
    )
    source_dir = snapshot / remote_dir
    profile_filenames = _profile_filenames(source_dir)
    visualization_filenames = profile_filenames[len(PROFILE_FILENAMES) - 1 : -1]
    if visualization_filenames:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=snapshot.name,
            allow_patterns=[
                f"{remote_dir.as_posix()}/{name}"
                for name in visualization_filenames
            ],
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    revision_dir = local_dir / snapshot.name
    with (local_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not revision_dir.exists():
            with TemporaryDirectory(prefix=f".{snapshot.name}.", dir=local_dir) as temporary:
                bundle = Path(temporary) / "profile"
                bundle.mkdir()
                for name in profile_filenames:
                    shutil.copyfile(source_dir / name, bundle / name)
                bundle.rename(revision_dir)
        (revision_dir / "DONE").read_bytes()
    return revision_dir


def _profile_filenames(profile_dir: Path) -> tuple[str, ...]:
    manifest = read_json(profile_dir / "profile.json")
    if "visualization_path" not in manifest:
        return PROFILE_FILENAMES
    return (*PROFILE_FILENAMES[:-1], manifest["visualization_path"], PROFILE_FILENAMES[-1])


def publish_simulation_dirs(
    repo_id: str,
    *,
    local_dirs: list[Path],
    remote_dirs: list[Path],
) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [
        CommitOperationAdd(
            path_in_repo=f"{remote.as_posix()}/{path.relative_to(local).as_posix()}",
            path_or_fileobj=str(path),
        )
        for local, remote in zip(local_dirs, remote_dirs)
        for path in sorted(local.rglob("*"))
        if path.is_file() and path.name not in LOCAL_ONLY_RUN_FILES
    ]
    HfApi().create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"simulations under {remote_dirs[0].parent.as_posix()}",
    )
