#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
STARVLA_ROOT = REPO_ROOT / "starVLA"
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.rl_games.data_conversion.convert_deadly_corridor_to_starvla_lerobot import (  # noqa: E402
    convert_dataset as convert_deadly_corridor_dataset,
)
from examples.rl_games.data_conversion.convert_demon_attack_to_starvla_lerobot import (  # noqa: E402
    convert_dataset as convert_demon_attack_dataset,
)
from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import (  # noqa: E402
    convert_dataset as convert_flappy_dataset,
)


Converter = Callable[..., dict]
CONVERTERS: dict[str, Converter] = {
    "deadly_corridor": convert_deadly_corridor_dataset,
    "demon_attack": convert_demon_attack_dataset,
    "flappy": convert_flappy_dataset,
}


def resolve_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def infer_env_name(raw_export_dir: Path) -> str:
    metadata = json.loads((raw_export_dir / "metadata.json").read_text(encoding="utf-8"))
    return str(metadata["env_name"])


def convert_raw_export_to_lerobot(
    *,
    raw_export_dir: Path,
    lerobot_root: Path,
    data_mix: str,
) -> dict:
    env_name = infer_env_name(raw_export_dir)
    converter = CONVERTERS[env_name]
    output_dir = lerobot_root / data_mix
    return converter(
        str(raw_export_dir),
        output_dir,
        force=True,
        require_latency_prompt_map=False,
    )


def push_lerobot_dataset(
    *,
    lerobot_root: Path,
    data_mix: str,
    repo_id: str,
    private: bool,
) -> None:
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
        "commit_message": "Upload StarVLA LeRobot dataset",
    }
    if token:
        create_repo_kwargs["token"] = token
        upload_folder_kwargs["token"] = token
    api = HfApi()
    api.create_repo(**create_repo_kwargs)
    api.upload_folder(**upload_folder_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a StarVLA LeRobot dataset from a raw rollout export.")
    parser.add_argument("raw_export_dir")
    parser.add_argument("lerobot_root")
    parser.add_argument("data_mix")
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--private", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_export_dir = Path(args.raw_export_dir).expanduser().resolve()
    lerobot_root = Path(args.lerobot_root).expanduser().resolve()
    data_mix = str(args.data_mix)

    manifest = convert_raw_export_to_lerobot(
        raw_export_dir=raw_export_dir,
        lerobot_root=lerobot_root,
        data_mix=data_mix,
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
