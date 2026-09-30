#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi

from latency_bench.data.lerobot_export import convert_canonical_rollout_to_lerobot


def resolve_token() -> str | None:
    """Resolve the Hugging Face token used for optional uploads.

    Returns:
        Explicitly supported environment token, or ``None`` when unset.
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def convert_raw_export_to_lerobot(
    *,
    raw_export_dir: Path,
    lerobot_root: Path,
    data_mix: str,
    action_carrier: str = "native",
) -> dict:
    """Convert a raw rollout export into a named LeRobot data mix.

    Args:
        raw_export_dir: Canonical rollout export directory.
        lerobot_root: Directory that will contain the LeRobot datasets.
        data_mix: Train dataset directory name under ``lerobot_root``.
        action_carrier: ``native`` catalog width or seven-dimensional ``bridge`` width.

    Returns:
        Conversion manifest returned by the LeRobot adapter.
    """
    output_dir = lerobot_root / data_mix
    return convert_canonical_rollout_to_lerobot(
        raw_export_dir,
        output_dir,
        force=True,
        action_carrier=action_carrier,
    )


def push_lerobot_dataset(
    *,
    lerobot_root: Path,
    data_mix: str,
    repo_id: str,
    private: bool,
) -> None:
    """Upload the converted LeRobot train/val directories to Hugging Face.

    Args:
        lerobot_root: Root directory containing converted datasets.
        data_mix: Train dataset directory name under ``lerobot_root``.
        repo_id: Hugging Face dataset repository id.
        private: Whether to create or update the repository as private.
    """
    token = resolve_token()
    create_repo_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private": bool(private),
        "exist_ok": True,
    }
    upload_folder_kwargs = {
        "folder_path": str(lerobot_root),
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": [f"{data_mix}/**", f"{data_mix}__val/**"],
        "commit_message": "Upload LeRobot dataset",
    }
    if token:
        create_repo_kwargs["token"] = token
        upload_folder_kwargs["token"] = token
    api = HfApi()
    api.create_repo(**create_repo_kwargs)
    api.upload_folder(**upload_folder_kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for LeRobot preparation.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Prepare a LeRobot dataset from a raw rollout export.")
    parser.add_argument("raw_export_dir")
    parser.add_argument("lerobot_root")
    parser.add_argument("data_mix")
    parser.add_argument("--action-carrier", choices=["native", "bridge"], default="native")
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--private", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LeRobot preparation CLI.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    raw_export_dir = Path(args.raw_export_dir).expanduser().resolve()
    lerobot_root = Path(args.lerobot_root).expanduser().resolve()
    data_mix = str(args.data_mix)

    manifest = convert_raw_export_to_lerobot(
        raw_export_dir=raw_export_dir,
        lerobot_root=lerobot_root,
        data_mix=data_mix,
        action_carrier=args.action_carrier,
    )
    if args.push_to_hub:
        push_lerobot_dataset(
            lerobot_root=lerobot_root,
            data_mix=data_mix,
            repo_id=str(args.push_to_hub),
            private=bool(args.private),
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
