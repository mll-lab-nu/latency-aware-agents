#!/usr/bin/env python
from __future__ import annotations

import argparse
import functools
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from examples.rl_games.bash_scripts.gr00t.data_conversion import convert_demon_attack_to_starvla_lerobot as base
from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import source_timing_from_args


ACTION_LABELS = [
    "NOOP",
    "ATTACK",
    "TURN_LEFT",
    "TURN_LEFT + ATTACK",
    "TURN_RIGHT",
    "TURN_RIGHT + ATTACK",
]


@contextmanager
def _defend_the_line_constants() -> Iterator[None]:
    old_action_labels = base.ACTION_LABELS
    old_action_dim = base.ACTION_DIM
    try:
        base.ACTION_LABELS = list(ACTION_LABELS)
        base.ACTION_DIM = len(ACTION_LABELS)
        yield
    finally:
        base.ACTION_LABELS = old_action_labels
        base.ACTION_DIM = old_action_dim


@functools.wraps(base.convert_dataset)
def convert_dataset(*args, **kwargs):
    with _defend_the_line_constants():
        return base.convert_dataset(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--latency-filter", "--latency_filter", default=None)
    parser.add_argument("--episodes-per-latency", "--episodes_per_latency", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", "--action_carrier", choices=["native", "bridge"], default="native")
    parser.add_argument("--context-images-column", "--context_images_column", default=None)
    parser.add_argument(
        "--context-images-output-column",
        "--context_images_output_column",
        default=base.DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
    )
    parser.add_argument("--image-sequence-length", "--image_sequence_length", type=int, default=4)
    parser.add_argument("--source-metadata")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--obs-stride-raw-frames", type=int)
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"])
    parser.add_argument(
        "--target-latency-unit",
        choices=["raw_frames", "observation_steps"],
        required=True,
    )
    args = parser.parse_args()
    fps, obs_stride_raw_frames, source_rows_unit = source_timing_from_args(
        source_metadata=args.source_metadata,
        source_fps=args.source_fps,
        obs_stride_raw_frames=args.obs_stride_raw_frames,
    )
    latency_filter = None
    if args.latency_filter:
        latency_filter = [int(item.strip()) for item in str(args.latency_filter).split(",") if item.strip()]

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        dataset_config_name=args.dataset_config_name,
        dataset_source_subdir=args.dataset_source_subdir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=False,
        latency_filter=latency_filter,
        episodes_per_latency=args.episodes_per_latency,
        action_carrier=args.action_carrier,
        context_images_column=args.context_images_column,
        context_images_output_column=args.context_images_output_column,
        image_sequence_length=args.image_sequence_length,
        fps=fps,
        obs_stride_raw_frames=obs_stride_raw_frames,
        source_latency_column=args.source_latency_column,
        target_latency_unit=args.target_latency_unit,
        source_rows_unit=source_rows_unit,
    )
    print(base.json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
