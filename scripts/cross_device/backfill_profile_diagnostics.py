from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.latency.profile import load_profile_manifest, save_profile_manifest
from latency_bench.utils.io import read_json, write_json
from scripts.cross_device.profile_cache import upload_folder_to_hf


def apply_backfill(*, profile_dir: Path, run_dir: Path) -> dict[str, str]:
    fingerprint = collect_hardware_fingerprint()
    profile_path = profile_dir / "profile.json"
    hardware_path = profile_dir / "hardware.json"
    diagnostics_path = profile_dir / "diagnostics.json"
    manifest_path = run_dir / "manifest.json"

    write_json(hardware_path, fingerprint)
    write_json(diagnostics_path, fingerprint)

    profile = load_profile_manifest(profile_path)
    profile["hardware"] = fingerprint
    profile["diagnostics_path"] = diagnostics_path.name
    save_profile_manifest(profile, profile_path)

    manifest = read_json(manifest_path)
    manifest["hardware_fingerprint"] = fingerprint
    manifest["diagnostics_path"] = str(diagnostics_path)
    write_json(manifest_path, manifest)

    return {
        "diagnostics_path": str(diagnostics_path),
        "hardware_path": str(hardware_path),
        "manifest_path": str(manifest_path),
        "profile_path": str(profile_path),
    }


def upload_backfill(*, profile_dir: Path, run_dir: Path, repo_id: str, repo_type: str, profile_path_in_repo: str, run_path_in_repo: str) -> None:
    upload_folder_to_hf(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=profile_dir,
        path_in_repo=profile_path_in_repo,
        allow_patterns=["profile.json", "hardware.json", "diagnostics.json"],
    )
    upload_folder_to_hf(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=run_dir,
        path_in_repo=run_path_in_repo,
        allow_patterns=["manifest.json"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill enhanced profile diagnostics into an existing cross-device profile.")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=os.environ.get("HF_PROFILE_REPO"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"))
    parser.add_argument("--profile-path-in-repo")
    parser.add_argument("--run-path-in-repo")
    args = parser.parse_args(argv)

    written = apply_backfill(profile_dir=args.profile_dir, run_dir=args.run_dir)
    print(written)
    if args.upload:
        upload_backfill(
            profile_dir=args.profile_dir,
            run_dir=args.run_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            profile_path_in_repo=args.profile_path_in_repo,
            run_path_in_repo=args.run_path_in_repo,
        )


if __name__ == "__main__":
    main()
