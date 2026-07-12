from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any


def download_calibration(
    workload: dict[str, Any],
    target_ref: Path,
    local_dir: Path,
) -> bool:
    from huggingface_hub import HfApi, snapshot_download

    remote_dir = Path("raw") / target_ref
    api = HfApi()
    remote_done = f"{remote_dir.as_posix()}/calibration.DONE"
    if not api.file_exists(
        repo_id=workload["hf_repo"],
        filename=remote_done,
        repo_type=workload["hf_repo_type"],
    ):
        return False
    snapshot = Path(
        snapshot_download(
            repo_id=workload["hf_repo"],
            repo_type=workload["hf_repo_type"],
            allow_patterns=[
                f"{remote_dir.as_posix()}/calibration.generated.yaml",
                f"{remote_dir.as_posix()}/calibration.json",
                f"{remote_dir.as_posix()}/hardware.json",
                remote_done,
            ],
        )
    )
    source_dir = snapshot / remote_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "calibration.generated.yaml",
        "calibration.json",
        "hardware.json",
        "calibration.DONE",
    ):
        shutil.copyfile(source_dir / name, local_dir / name)
    return True


def download_profile(workload: dict[str, Any], profile_ref: Path) -> Path:
    from huggingface_hub import snapshot_download

    remote_dir = Path("profiles") / profile_ref
    snapshot = Path(
        snapshot_download(
            repo_id=workload["hf_repo"],
            repo_type=workload["hf_repo_type"],
            allow_patterns=f"{remote_dir.as_posix()}/**",
        )
    )
    profile_dir = snapshot / remote_dir
    (profile_dir / "DONE").read_bytes()
    return profile_dir


def publish_directory_leaf(
    workload: dict[str, Any],
    local_dir: Path,
    remote_dir: Path,
    *,
    resume: bool = False,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    remote_done = f"{remote_dir.as_posix()}/DONE"
    if api.file_exists(
        repo_id=workload["hf_repo"],
        filename=remote_done,
        repo_type=workload["hf_repo_type"],
    ):
        if resume:
            return
        raise FileExistsError(remote_done)
    done = local_dir / "DONE"
    done.read_bytes()
    api.upload_folder(
        repo_id=workload["hf_repo"],
        repo_type=workload["hf_repo_type"],
        folder_path=str(local_dir),
        path_in_repo=remote_dir.as_posix(),
        ignore_patterns=["DONE"],
    )
    api.upload_file(
        repo_id=workload["hf_repo"],
        repo_type=workload["hf_repo_type"],
        path_or_fileobj=str(done),
        path_in_repo=remote_done,
    )


def publish_calibration(
    workload: dict[str, Any],
    local_dir: Path,
    remote_dir: Path,
    *,
    resume: bool = False,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    remote_done = f"{remote_dir.as_posix()}/calibration.DONE"
    if api.file_exists(
        repo_id=workload["hf_repo"],
        filename=remote_done,
        repo_type=workload["hf_repo_type"],
    ):
        if resume:
            return
        raise FileExistsError(remote_done)
    done = local_dir / "calibration.DONE"
    done.read_bytes()
    for name in ("calibration.generated.yaml", "calibration.json", "hardware.json"):
        api.upload_file(
            repo_id=workload["hf_repo"],
            repo_type=workload["hf_repo_type"],
            path_or_fileobj=str(local_dir / name),
            path_in_repo=f"{remote_dir.as_posix()}/{name}",
        )
    api.upload_file(
        repo_id=workload["hf_repo"],
        repo_type=workload["hf_repo_type"],
        path_or_fileobj=str(done),
        path_in_repo=remote_done,
    )


def download_study(study: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    remote_root = f"cross_device/{study['study_id']}"
    snapshot = Path(
        snapshot_download(
            repo_id=study["hf_repo"],
            repo_type=study["hf_repo_type"],
            allow_patterns=f"{remote_root}/**",
        )
    )
    return snapshot / remote_root
