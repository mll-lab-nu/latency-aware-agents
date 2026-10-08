#!/usr/bin/env python
from __future__ import annotations

import argparse
import functools
import sys
from contextlib import contextmanager
from typing import Iterator

from examples.rl_games.bash_scripts.gr00t.data_conversion import verify_demon_attack_dataset as base


EXPECTED_ACTIONS = [
    "noop",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
]
EXPECTED_PROMPT = (
    "Collect useful objects and avoid lyres. Choose exactly one action from: "
    "noop, up, right, left, down, upright, upleft, downright, downleft."
)
REQUIRED_PROMPT_PARTS = [
    "Collect useful objects",
    "noop",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
]


@contextmanager
def _asterix_constants() -> Iterator[None]:
    old_expected_actions = base.EXPECTED_ACTIONS
    old_expected_prompt = base.EXPECTED_PROMPT
    old_required_prompt_parts = base.REQUIRED_PROMPT_PARTS
    try:
        base.EXPECTED_ACTIONS = list(EXPECTED_ACTIONS)
        base.EXPECTED_PROMPT = EXPECTED_PROMPT
        base.REQUIRED_PROMPT_PARTS = list(REQUIRED_PROMPT_PARTS)
        yield
    finally:
        base.EXPECTED_ACTIONS = old_expected_actions
        base.EXPECTED_PROMPT = old_expected_prompt
        base.REQUIRED_PROMPT_PARTS = old_required_prompt_parts


@functools.wraps(base.verify_dataset)
def verify_dataset(*args, **kwargs) -> bool:
    with _asterix_constants():
        return base.verify_dataset(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-mixed-latency-prompts", "--allow_mixed_latency_prompts", action="store_true")
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"], required=True)
    parser.add_argument("--target-latency-unit", choices=["raw_frames", "observation_steps"], required=True)
    parser.add_argument("--obs-stride-raw-frames", type=int, required=True)
    args = parser.parse_args()

    try:
        ok = verify_dataset(
            args.dataset_name,
            rows=args.rows,
            cache_dir=args.cache_dir,
            dataset_config_name=args.dataset_config_name,
            dataset_source_subdir=args.dataset_source_subdir,
            strict=args.strict,
            allow_mixed_latency_prompts=args.allow_mixed_latency_prompts,
            source_latency_column=args.source_latency_column,
            target_latency_unit=args.target_latency_unit,
            obs_stride_raw_frames=args.obs_stride_raw_frames,
        )
    except Exception:
        if args.strict:
            raise
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
