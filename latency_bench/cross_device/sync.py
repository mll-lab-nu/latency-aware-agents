"""Local-first Hugging Face cache for latency profile artifacts.

Profiles are addressed inside the mirrored protocol output root, for example
`profiles/model/env/device/instance/profile.json`.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Debug / bulky artifacts that workers upload (full tree) but the aggregator
# does not need. Excluded on download so `aggregate` only pulls the slim set it
# actually reads: manifests/runs.jsonl, top-level runs/**/episode_metrics.jsonl,
# and profiles/**/{profile.json,infer_latencies.jsonl,e2e_latencies.jsonl,...}.
RESULTS_DEBUG_IGNORE = [
    "**/sample_batches/**",
    "**/*.png",
    "**/warmup_*",
    "**/generated_configs/**",
    "**/runs/**/latencies.jsonl",
    "**/runs/**/infer_latencies.jsonl",
    "**/runs/**/e2e_latencies.jsonl",
    "**/runs/**/runtime_telemetry.jsonl",
    "**/runs/**/workers/**",
    "**/*.warmup.yaml",
    "**/*.calibration.yaml",
]


def hf_env() -> dict[str, str | None] | None:
    """Read the HF profile-repo configuration from the environment.

    Returns:
        `{"repo_id", "repo_type", "revision"}`, or None when `HF_PROFILE_REPO`
        is unset (all HF sync is disabled).
    """
    repo_id = os.environ.get("HF_PROFILE_REPO")
    if not repo_id:
        return None
    return {
        "repo_id": repo_id,
        "repo_type": os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"),
        "revision": os.environ.get("HF_PROFILE_REVISION"),
    }


def ensure_profile_cached(
    *,
    profile_path: Path,
    profiles_dir: Path,
    hf_repo: str,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path:
    """Return a local profile path, downloading the profile folder if needed.

    Args:
        profile_path: Expected local `profile.json` path.
        profiles_dir: Root of the local profile cache.
        hf_repo: Hugging Face repo id containing profile folders.
        repo_type: Hugging Face repo type, normally `dataset`.
        revision: Optional Hugging Face revision.

    Returns:
        The expected local `profile.json` path after local hit or download.
    """
    if profile_path.exists():
        return profile_path
    download_profile_folder_from_hf(
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
    return profile_path.parent.relative_to(profiles_dir.parent).as_posix()


def download_profile_folder_from_hf(
    *,
    repo_id: str,
    local_dir: Path,
    remote_dir: str,
    repo_type: str,
    revision: str | None,
) -> None:
    """Download one profile folder from Hugging Face.

    Args:
        repo_id: Hugging Face repo id.
        local_dir: Local folder where profile artifacts are copied.
        remote_dir: Folder path inside the repo.
        repo_type: Hugging Face repo type.
        revision: Optional revision.

    Side effects:
        Copies the remote profile folder into `local_dir`.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=f"{remote_dir}/*",
        )
    )
    shutil.copytree(snapshot_dir / remote_dir, local_dir, dirs_exist_ok=True)


def upload_folder_to_hf(
    *,
    repo_id: str,
    local_dir: Path,
    repo_type: str,
    path_in_repo: str | None = None,
) -> None:
    """Upload one local artifact folder to Hugging Face.

    Args:
        repo_id: Hugging Face repo id.
        local_dir: Folder containing protocol artifacts.
        repo_type: Hugging Face repo type.
        path_in_repo: Destination folder inside the repo. When set, the folder
            lands at `<path_in_repo>/...` so an upload round-trips with the
            `profiles/<model>/<env>/<device>/<instance>` layout that
            `ensure_profile_cached` downloads from.

    Side effects:
        Uploads the folder using `huggingface_hub.HfApi`.
    """
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
    )


def download_results_tree(
    *,
    repo_id: str,
    repo_type: str,
    local_root: Path,
    revision: str | None = None,
) -> Path:
    """Download the per-machine `raw/` results trees from Hugging Face.

    Pulls only the slim set the aggregator reads (manifests, per-run
    `episode_metrics.jsonl`, and profile artifacts), excluding the bulky debug
    files (`sample_batches/`, warmup `*.png`, etc.) that workers upload.

    Args:
        repo_id: Hugging Face repo id holding the `raw/<device>/instances/<instance>/<runtime>/...` trees.
        repo_type: Hugging Face repo type, normally `dataset`.
        local_root: Local directory to receive the downloaded `raw/` tree.
        revision: Optional Hugging Face revision.

    Returns:
        `local_root` (the raw-root to hand to `aggregate --raw-root`).

    Side effects:
        Copies the filtered remote `raw/` tree into `local_root`.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=["raw/**"],
            ignore_patterns=RESULTS_DEBUG_IGNORE,
        )
    )
    local_root.mkdir(parents=True, exist_ok=True)
    raw_src = snapshot_dir / "raw"
    if raw_src.exists():
        shutil.copytree(raw_src, local_root / "raw", dirs_exist_ok=True)
    return local_root


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

    upload = subparsers.add_parser("upload", help="Upload one local artifact folder to the profile cache repo.")
    upload.add_argument("--folder", type=Path, required=True)
    upload.add_argument("--hf-repo", required=True)
    upload.add_argument("--repo-type", default="dataset")
    upload.add_argument(
        "--path-in-repo",
        help="Destination folder inside the repo (e.g. profiles/<model>/<env>/<device>/<instance>).",
    )

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

    upload_folder_to_hf(
        repo_id=args.hf_repo,
        local_dir=args.folder,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
    )
    print(args.folder)


if __name__ == "__main__":
    main()
