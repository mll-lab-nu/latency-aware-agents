from __future__ import annotations

from pathlib import Path
from typing import Any


def study_remote_root(study: dict[str, Any]) -> str:
    return f"cross_device/{study['study_id']}"


def download_measurements(study: dict[str, Any]) -> Path:
    return _download(study, "measure/**")


def download_study(study: dict[str, Any]) -> Path:
    return _download(study, "**")


def publish_study_unit(
    study: dict[str, Any],
    local_dir: Path,
    relative_dir: Path,
) -> None:
    from huggingface_hub import HfApi

    remote_dir = f"{study_remote_root(study)}/{relative_dir.as_posix()}"
    api = HfApi()
    api.upload_folder(
        repo_id=study["hf_repo"],
        repo_type=study["hf_repo_type"],
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
        ignore_patterns=["DONE"],
    )
    api.upload_file(
        repo_id=study["hf_repo"],
        repo_type=study["hf_repo_type"],
        path_or_fileobj=str(local_dir / "DONE"),
        path_in_repo=f"{remote_dir}/DONE",
    )


def _download(study: dict[str, Any], pattern: str) -> Path:
    from huggingface_hub import snapshot_download

    remote_root = study_remote_root(study)
    snapshot = Path(
        snapshot_download(
            repo_id=study["hf_repo"],
            repo_type=study["hf_repo_type"],
            allow_patterns=f"{remote_root}/{pattern}",
        )
    )
    return snapshot / remote_root
