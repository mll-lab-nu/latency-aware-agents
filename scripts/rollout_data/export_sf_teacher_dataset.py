#!/usr/bin/env python
"""CLI for exporting Sample Factory teacher rollouts as canonical datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from latency_bench.data.ghost_trail import GhostTrailConfig
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
    parser.add_argument("--max-steps-per-episode", type=int, required=True, help="Max policy decision steps per episode.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replay-num-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-image-size", type=str, default="")
    parser.add_argument("--image-writer-workers", type=int, default=0)
    parser.add_argument("--row-buffer-size", type=int, default=8192)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default="")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--hf-max-shard-size", type=str, default="500MB")
    parser.add_argument("--hf-config-name", type=str, default="")
    parser.add_argument(
        "--filter-preset", type=str, default="clean_v1", choices=["none", "clean_v1", "contiguous_v1"]
    )
    parser.add_argument("--demon-episode-threshold", type=float, default=None,
                        help="Override Demon Attack episode-level clipped return threshold.")
    parser.add_argument("--rollout-seed-offset", type=int, default=0,
                        help="Offset added to the probe reset base seed so sharded probe workers "
                             "explore disjoint episodes. Default 0 reproduces single-process behavior.")
    parser.add_argument("--probe-only", action="store_true",
                        help="Run only the probe phase and persist probe_state.json (no replay/parquet). "
                             "Used by sharded probe workers; merge + single-process replay materialize the dataset.")
    parser.add_argument(
        "--image-transform",
        type=str,
        default="raw_rgb",
        choices=["raw_rgb", "flappy_ghost_trail"],
        help="Observation image transform applied during replay before row filtering.",
    )
    parser.add_argument(
        "--ghost-trail-history-frames",
        type=int,
        default=5,
        help="For flappy_ghost_trail, number of past decision-step frames to composite behind the current frame.",
    )
    parser.add_argument("--ghost-trail-gamma", type=float, default=1.3)
    parser.add_argument("--ghost-trail-min-alpha", type=int, default=35)
    parser.add_argument("--ghost-trail-ground-fraction", type=float, default=0.22)
    parser.add_argument("--ghost-trail-scroll-px-per-step", type=float, default=4.0)
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
    filter_kwargs = {}
    if args.demon_episode_threshold is not None:
        filter_kwargs["demon_episode_threshold"] = float(args.demon_episode_threshold)
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
        replay_num_envs=int(args.replay_num_envs),
        device=str(args.device),
        deterministic=bool(args.deterministic),
        export_image_size=_parse_image_size(str(args.export_image_size)),
        image_writer_workers=int(args.image_writer_workers),
        row_buffer_size=int(args.row_buffer_size),
        push_to_hub=bool(args.push_to_hub),
        hf_repo_id=str(args.hf_repo_id),
        hf_private=bool(args.hf_private),
        hf_token=str(args.hf_token),
        hf_max_shard_size=str(args.hf_max_shard_size),
        hf_config_name=str(args.hf_config_name),
        filter_preset=str(args.filter_preset),
        filter_kwargs=filter_kwargs or None,
        rollout_seed_offset=int(args.rollout_seed_offset),
        probe_only=bool(args.probe_only),
        image_transform_config=GhostTrailConfig(
            image_transform=str(args.image_transform),
            history_frames=int(args.ghost_trail_history_frames),
            gamma=float(args.ghost_trail_gamma),
            min_alpha=int(args.ghost_trail_min_alpha),
            ground_fraction=float(args.ghost_trail_ground_fraction),
            scroll_px_per_step=float(args.ghost_trail_scroll_px_per_step),
        ),
    )
    if metadata.get("probe_only"):
        print(f"[probe-only] wrote {metadata.get('accepted_episodes')} accepted specs to {output_dir}")
    else:
        print(f"[export] wrote train={metadata['train_rows']} rows val={metadata['val_rows']} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
