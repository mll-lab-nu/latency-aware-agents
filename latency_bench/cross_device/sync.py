"""Local-first Hugging Face cache for latency profile artifacts.

Profiles are addressed inside the mirrored protocol output root, for example
`profiles/model/env/device/instance/profile.json`.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

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


def study_remote_root(study: dict[str, Any]) -> str:
    return f"cross_device/{study['study_id']}"


def download_measurements(study: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    remote_root = study_remote_root(study)
    snapshot = Path(
        snapshot_download(
            repo_id=study["hf_repo"],
            repo_type=study["hf_repo_type"],
            allow_patterns=f"{remote_root}/measure/**",
        )
    )
    return snapshot / remote_root


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


def upload_file_to_hf(
    *,
    repo_id: str,
    local_path: Path,
    repo_type: str,
    path_in_repo: str,
) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
    )


def protocol_raw_root(device_id: str, instance_id: str, protocol_id: str) -> str:
    return f"raw/{device_id}/instances/{instance_id}/cross_device/{protocol_id}"


def publish_protocol_folder(
    spec: dict,
    *,
    local_dir: Path,
    path_in_repo: str,
) -> None:
    upload_folder_to_hf(
        repo_id=spec["hf_repo"],
        repo_type=spec["hf_repo_type"],
        local_dir=local_dir,
        path_in_repo=path_in_repo,
    )


def publish_protocol_paths(
    spec: dict,
    *,
    output_root: Path,
    remote_root: str,
    relative_paths: list[str],
) -> None:
    for relative_path in dict.fromkeys(relative_paths):
        publish_protocol_folder(
            spec,
            local_dir=output_root / relative_path,
            path_in_repo=f"{remote_root}/{relative_path}",
        )


def publish_protocol_completion(
    spec: dict,
    *,
    local_path: Path,
    path_in_repo: str,
) -> None:
    upload_file_to_hf(
        repo_id=spec["hf_repo"],
        repo_type=spec["hf_repo_type"],
        local_path=local_path,
        path_in_repo=path_in_repo,
    )


def download_measure_completions(spec: dict, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=spec["hf_repo"],
            repo_type=spec["hf_repo_type"],
            allow_patterns=[
                f"raw/*/instances/*/cross_device/{spec['protocol_id']}/completion/measure/**/*.json"
            ],
        )
    )
    shutil.copytree(snapshot / "raw", destination / "raw", dirs_exist_ok=True)
    return destination


def download_profile_build(spec: dict, completion: dict, output_root: Path) -> Path:
    destination = (
        output_root
        / "profile_builds"
        / completion["model_type"]
        / completion["env_name"]
        / completion["target_device_id"]
        / completion["target_profile_instance_id"]
        / completion["profile_build_id"]
    )
    remote_dir = (
        f"profile_builds/{completion['model_type']}/{completion['env_name']}/"
        f"{completion['target_device_id']}/{completion['target_profile_instance_id']}/"
        f"{completion['profile_build_id']}"
    )
    profile_path = destination / "profile.json"
    if profile_path.exists():
        return profile_path
    download_profile_folder_from_hf(
        repo_id=spec["hf_repo"],
        repo_type=spec["hf_repo_type"],
        revision=None,
        remote_dir=remote_dir,
        local_dir=destination,
    )
    return profile_path


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
        repo_id: Hugging Face repo id holding the `raw/<device>/instances/<instance>/...` trees.
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
