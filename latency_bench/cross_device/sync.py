from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

from latency_bench.utils.io import read_json


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
                f"{remote_dir.as_posix()}/calibration.json",
                f"{remote_dir.as_posix()}/hardware.json",
                remote_done,
            ],
        )
    )
    source_dir = snapshot / remote_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in (
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


def download_raw_selection(
    workload: dict[str, Any],
    target_ref: Path,
    local_dir: Path,
    session_ids: list[int],
    *,
    overwrite: bool = False,
) -> None:
    from huggingface_hub import HfApi, snapshot_download

    remote_dir = Path("raw") / target_ref
    api = HfApi()
    markers = [
        f"{remote_dir.as_posix()}/calibration.DONE",
        *(
            f"{remote_dir.as_posix()}/sessions/{session_id}/DONE"
            for session_id in session_ids
        ),
    ]
    for marker in markers:
        if not api.file_exists(
            repo_id=workload["hf_repo"],
            filename=marker,
            repo_type=workload["hf_repo_type"],
        ):
            raise FileNotFoundError(marker)

    snapshot = Path(
        snapshot_download(
            repo_id=workload["hf_repo"],
            repo_type=workload["hf_repo_type"],
            allow_patterns=[
                f"{remote_dir.as_posix()}/calibration.json",
                f"{remote_dir.as_posix()}/hardware.json",
                markers[0],
                *(
                    f"{remote_dir.as_posix()}/sessions/{session_id}/**"
                    for session_id in session_ids
                ),
            ],
        )
    )
    source_dir = snapshot / remote_dir
    completed_local_paths = [
        local_dir / "calibration.DONE",
        *(local_dir / "sessions" / str(session_id) / "DONE" for session_id in session_ids),
    ]
    if not overwrite:
        for completed_path in completed_local_paths:
            if completed_path.exists():
                raise FileExistsError(completed_path)

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{local_dir.name}.download-",
        dir=local_dir.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        for name in ("calibration.json", "hardware.json", "calibration.DONE"):
            shutil.copyfile(source_dir / name, staging_dir / name)
        for session_id in session_ids:
            shutil.copytree(
                source_dir / "sessions" / str(session_id),
                staging_dir / "sessions" / str(session_id),
            )

        (staging_dir / "calibration.DONE").read_bytes()
        for session_id in session_ids:
            (staging_dir / "sessions" / str(session_id) / "DONE").read_bytes()

        local_dir.mkdir(parents=True, exist_ok=True)
        for name in ("calibration.json", "hardware.json", "calibration.DONE"):
            destination = local_dir / name
            if destination.exists():
                destination.unlink()
            shutil.move(staging_dir / name, destination)
        for session_id in session_ids:
            destination = local_dir / "sessions" / str(session_id)
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(staging_dir / "sessions" / str(session_id), destination)


def download_analysis_selection(
    workload: dict[str, Any],
    profile_refs: list[Path],
    real_session_ids: list[int],
    methods: list[str],
    latency_seeds: list[int],
) -> Path:
    from huggingface_hub import snapshot_download

    patterns = []
    for profile_ref in profile_refs:
        patterns.append(f"profiles/{profile_ref.as_posix()}/**")
        target = profile_ref.parent
        profile = read_json(download_profile(workload, profile_ref) / "profile.json")
        patterns.extend(
            f"raw/{target.as_posix()}/sessions/{session_id}/**"
            for session_id in real_session_ids
        )
        patterns.extend(
            f"raw/{target.as_posix()}/sessions/{session_id}/**"
            for session_id in profile["fit_session_ids"]
        )
        patterns.extend(
            f"sim/{profile_ref.as_posix()}/{method}/{latency_seed}/**"
            for method in methods
            for latency_seed in latency_seeds
        )
    return Path(
        snapshot_download(
            repo_id=workload["hf_repo"],
            repo_type=workload["hf_repo_type"],
            allow_patterns=patterns,
        )
    )


def publish_directory_leaf(
    workload: dict[str, Any],
    local_dir: Path,
    remote_dir: Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    remote_done = f"{remote_dir.as_posix()}/DONE"
    if not overwrite and api.file_exists(
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
        delete_patterns=["**"] if overwrite else None,
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
    for name in ("calibration.json", "hardware.json"):
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
