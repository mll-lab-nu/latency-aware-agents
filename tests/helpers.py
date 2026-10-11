"""Shared test fixtures and helpers reused across the test suite."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord


# Repo root, anchored on this file's fixed location (tests/helpers.py). Tests live
# in mirrored subdirectories at varying depths, so they must NOT compute the repo
# root via `Path(__file__).parents[N]` (the depth changes when a test is moved).
# Import REPO_ROOT from here instead.
REPO_ROOT = Path(__file__).resolve().parent.parent


def make_action_event(action_id: int, action: Action, ready_step: int, clock: EnvClock) -> ActionEvent:
    ready_time_ms = clock.step_to_time_ms(ready_step)
    return ActionEvent(
        action_id=action_id,
        action=action,
        obs_env_step=0,
        ready_time_ms=ready_time_ms,
        ready_step_float=float(ready_step),
        latency=LatencyRecord(
            obs_id=action_id,
            latency_ms=ready_time_ms,
        ),
    )


def save_test_image(path: Path, fill: int) -> None:
    from PIL import Image

    Image.fromarray(np.full((2, 2, 3), fill, dtype=np.uint8)).save(path)


def write_rollout_fixture(output_dir, train_rows, val_rows, *, action_spec):
    """Build converter fixtures through the maintained canonical storage boundary."""
    from latency_bench.data.hf_parquet import (
        write_episode_row_shard, write_parquet_splits_from_shards, load_parquet_splits,
    )

    for rows in (train_rows, val_rows):
        for _, episode in groupby(sorted(rows, key=lambda row: row["episode_idx"]),
                                  key=lambda row: row["episode_idx"]):
            write_episode_row_shard(output_dir, list(episode))
    write_parquet_splits_from_shards(output_dir, action_spec=action_spec)
    for shard in output_dir.glob("episode_rows/*/*.jsonl"):
        shard.unlink()
    return load_parquet_splits(output_dir)
