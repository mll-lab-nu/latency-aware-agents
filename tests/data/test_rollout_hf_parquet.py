from __future__ import annotations

from itertools import groupby

import pyarrow.parquet as pq

from latency_bench.data.hf_parquet import (
    write_episode_row_shard, write_parquet_splits_from_shards, load_parquet_splits,
)

from helpers import save_test_image


def write_rollout_fixture(output_dir, train_rows, val_rows, *, action_spec):
    """Build converter fixtures through the maintained canonical storage boundary."""
    for rows in (train_rows, val_rows):
        for _, episode in groupby(sorted(rows, key=lambda row: row["episode_idx"]),
                                  key=lambda row: row["episode_idx"]):
            write_episode_row_shard(output_dir, list(episode))
    write_parquet_splits_from_shards(output_dir, action_spec=action_spec)
    for shard in output_dir.glob("episode_rows/*/*.jsonl"):
        shard.unlink()
    return load_parquet_splits(output_dir)


def test_hf_parquet_writes_canonical_columns_only(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    row = {
        "image": str(image_path),
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 11,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action_id": 1,
        "action": [0.0, 1.0],
        "action_text": "FLAP",
    }

    dataset_dict = write_rollout_fixture(
        tmp_path / "export",
        [row],
        [],
        action_spec=action_spec_for_env("flappy"),
    )

    table = pq.read_table(tmp_path / "export" / "train.parquet")
    assert len(dataset_dict["train"]) == 1
    assert "decision_step" in table.column_names
    assert "latency_raw_frames" in table.column_names
    assert "latency" not in table.column_names
    assert "done" not in table.column_names
    assert "t" not in table.column_names
    assert "context_images" not in table.column_names
    assert table.to_pylist()[0]["image"]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_hf_parquet_shards_sort_by_decision_step(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_episode_row_shard, write_parquet_splits_from_shards

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    rows = [
        {
            "image": str(image_path),
            "env_name": "flappy",
            "episode_idx": 0,
            "decision_step": decision_step,
            "split": "train",
            "seed": 11,
            "prompt": "pick action",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "raw_reward": 1.5,
            "action_id": 1,
            "action": [0.0, 1.0],
            "action_text": "FLAP",
        }
        for decision_step in (1, 0)
    ]
    output_dir = tmp_path / "export"

    write_episode_row_shard(output_dir, rows)
    write_parquet_splits_from_shards(output_dir, action_spec=action_spec_for_env("flappy"))

    table = pq.read_table(output_dir / "train.parquet")
    assert [row["decision_step"] for row in table.to_pylist()] == [0, 1]


def test_hf_parquet_embeds_context_images_when_rows_provide_them(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env

    image_path = tmp_path / "image.png"
    context_paths = [tmp_path / f"context_{idx}.png" for idx in range(3)]
    save_test_image(image_path, 9)
    for idx, context_path in enumerate(context_paths):
        save_test_image(context_path, idx + 1)
    row = {
        "image": str(image_path),
        "context_images": [str(path) for path in context_paths],
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 11,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action_id": 1,
        "action": [0.0, 1.0],
        "action_text": "FLAP",
    }

    write_rollout_fixture(
        tmp_path / "export",
        [row],
        [],
        action_spec=action_spec_for_env("flappy"),
    )

    table = pq.read_table(tmp_path / "export" / "train.parquet")
    exported = table.to_pylist()[0]
    assert "context_images" in table.column_names
    assert len(exported["context_images"]) == 3
    assert exported["context_images"][0]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")
