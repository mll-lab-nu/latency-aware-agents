"""Local-first Hugging Face cache for latency profile artifacts.

Profiles are addressed by the same relative path used under `PROFILES_DIR`,
for example `policy/env/L_A100/profile.json`. The cache only moves profile
artifacts, not full protocol run directories.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable


PROFILE_ARTIFACT_FILENAMES = (
    "profile.json",
    "latencies.jsonl",
    "profile_summary.json",
    "profile_summary.csv",
)


DownloadProfileFolder = Callable[..., None]


def ensure_profile_cached(
    *,
    profile_path: Path,
    profiles_dir: Path,
    hf_repo: str,
    repo_type: str = "dataset",
    revision: str | None = None,
    download_profile_folder: DownloadProfileFolder | None = None,
) -> Path:
    """Return a local profile path, downloading the profile folder if needed.

    Args:
        profile_path: Expected local `profile.json` path.
        profiles_dir: Root of the local profile cache.
        hf_repo: Hugging Face repo id containing profile folders.
        repo_type: Hugging Face repo type, normally `dataset`.
        revision: Optional Hugging Face revision.
        download_profile_folder: Injectable downloader used by tests.

    Returns:
        The expected local `profile.json` path after local hit or download.
    """
    if profile_path.exists():
        return profile_path
    downloader = download_profile_folder or download_profile_folder_from_hf
    downloader(
        repo_id=hf_repo,
        local_dir=profile_path.parent,
        remote_dir=profile_remote_dir(profile_path=profile_path, profiles_dir=profiles_dir),
        repo_type=repo_type,
        revision=revision,
    )
    return profile_path


def profile_remote_dir(*, profile_path: Path, profiles_dir: Path) -> str:
    """Map a local profile path to its Hugging Face folder path.

    Args:
        profile_path: Local `profile.json` path.
        profiles_dir: Local profile cache root.

    Returns:
        POSIX relative folder path, excluding the `profile.json` filename.
    """
    return profile_path.parent.relative_to(profiles_dir).as_posix()


def download_profile_folder_from_hf(
    *,
    repo_id: str,
    local_dir: Path,
    remote_dir: str,
    repo_type: str,
    revision: str | None,
) -> None:
    """Download the standard profile artifact files from Hugging Face.

    Args:
        repo_id: Hugging Face repo id.
        local_dir: Local folder where profile artifacts are copied.
        remote_dir: Folder path inside the repo.
        repo_type: Hugging Face repo type.
        revision: Optional revision.

    Side effects:
        Creates `local_dir` and copies profile artifacts into it.
    """
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in PROFILE_ARTIFACT_FILENAMES:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=f"{remote_dir}/{filename}",
            repo_type=repo_type,
            revision=revision,
        )
        shutil.copy2(downloaded, local_dir / filename)


def upload_profile_folder_to_hf(
    *,
    repo_id: str,
    local_dir: Path,
    remote_dir: str,
    repo_type: str,
) -> None:
    """Upload one local profile folder to Hugging Face.

    Args:
        repo_id: Hugging Face repo id.
        local_dir: Folder containing profile artifacts.
        remote_dir: Folder path inside the repo.
        repo_type: Hugging Face repo type.

    Side effects:
        Uploads the folder using `huggingface_hub.HfApi`.
    """
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI for manual profile cache upload/download.

    Args:
        argv: Optional argument vector for tests; uses process argv when None.
    """
    parser = argparse.ArgumentParser(description="Upload or download cross-device latency profile artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download one profile folder into the local profiles cache.")
    download.add_argument("--profile-path", type=Path, required=True)
    download.add_argument("--profiles-dir", type=Path, required=True)
    download.add_argument("--hf-repo", required=True)
    download.add_argument("--repo-type", default="dataset")
    download.add_argument("--revision")

    upload = subparsers.add_parser("upload", help="Upload one local profile folder to the profile cache repo.")
    upload.add_argument("--profile-path", type=Path, required=True)
    upload.add_argument("--profiles-dir", type=Path, required=True)
    upload.add_argument("--hf-repo", required=True)
    upload.add_argument("--repo-type", default="dataset")

    args = parser.parse_args(argv)
    if args.command == "download":
        path = ensure_profile_cached(
            profile_path=args.profile_path,
            profiles_dir=args.profiles_dir,
            hf_repo=args.hf_repo,
            repo_type=args.repo_type,
            revision=args.revision,
        )
        print(path)
        return

    upload_profile_folder_to_hf(
        repo_id=args.hf_repo,
        local_dir=args.profile_path.parent,
        remote_dir=profile_remote_dir(profile_path=args.profile_path, profiles_dir=args.profiles_dir),
        repo_type=args.repo_type,
    )
    print(args.profile_path)


if __name__ == "__main__":
    main()
