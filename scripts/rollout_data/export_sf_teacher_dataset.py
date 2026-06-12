#!/usr/bin/env python
"""CLI for exporting Sample Factory teacher rollouts as canonical datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from latency_bench.data.sf_teacher_export import export_sample_factory_teacher_dataset


def _parse_image_size(value: str) -> tuple[int, int] | None:
    """Parse an optional CLI image size value.

    Args:
        value: Empty string or ``WIDTH,HEIGHT``.

    Returns:
        Parsed size tuple, or ``None`` when empty.
    """
    if not value:
        return None
    width, height = [int(item.strip()) for item in str(value).split(",")]
    return width, height


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for teacher dataset export.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Export Sample Factory teacher rollouts as parquet datasets.")
    parser.add_argument("--checkpoint-root", required=True, help="Sample Factory experiment directory or checkpoint file.")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files and observation images are written.")
    parser.add_argument("--latency-config", default="", help="Optional YAML containing a training-compatible latency: section.")
    parser.add_argument("--env-name", default="", choices=["", "deadly_corridor", "demon_attack", "flappy"])
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--max-steps-per-episode", type=int, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-image-size", type=str, default="")
    parser.add_argument("--image-writer-workers", type=int, default=0)
    parser.add_argument("--row-buffer-size", type=int, default=8192)
    parser.add_argument("--filter-preset", type=str, default="")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default="")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--hf-max-shard-size", type=str, default="500MB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the teacher dataset export CLI.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata = export_sample_factory_teacher_dataset(
        checkpoint_root=args.checkpoint_root,
        output_dir=output_dir,
        latency_config_path=args.latency_config or None,
        env_name=str(args.env_name),
        max_episodes=int(args.max_episodes),
        max_steps_per_episode=int(args.max_steps_per_episode),
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        num_envs=int(args.num_envs),
        device=str(args.device),
        deterministic=bool(args.deterministic),
        export_image_size=_parse_image_size(str(args.export_image_size)),
        image_writer_workers=int(args.image_writer_workers),
        row_buffer_size=int(args.row_buffer_size),
        filter_preset=str(args.filter_preset),
        push_to_hub=bool(args.push_to_hub),
        hf_repo_id=str(args.hf_repo_id),
        hf_private=bool(args.hf_private),
        hf_token=str(args.hf_token),
        hf_max_shard_size=str(args.hf_max_shard_size),
    )
    print(f"[export] wrote train={metadata['train_rows']} rows val={metadata['val_rows']} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
