from __future__ import annotations

from itertools import groupby

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latency_bench.data.hf_parquet import (
    write_episode_row_shard, write_parquet_splits_from_shards, load_parquet_splits,
)
from helpers import save_test_image, write_rollout_fixture


def test_hf_parquet_embeds_images_and_writes_empty_validation_split(tmp_path):
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

    image_path.unlink()
    table = pq.read_table(tmp_path / "export" / "train.parquet")
    assert len(dataset_dict["val"]) == 0
    assert table.to_pylist()[0]["image"]["path"] is None
    assert len(dataset_dict["train"]) == 1
    assert "decision_step" in table.column_names
    assert "latency_raw_frames" in table.column_names
    assert table.to_pylist()[0]["image"]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


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


def test_write_parquet_splits_preserves_rows_across_large_exports(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env

    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((1, 1, 3), 7, dtype=np.uint8)).save(image_path)
    rows = [
        {
            "image": str(image_path),
            "env_name": "flappy",
            "episode_idx": episode_idx,
            "decision_step": 0,
            "split": "train",
            "seed": 11 + episode_idx,
            "prompt": "pick action",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "raw_reward": 1.5,
            "action": [0.0, 1.0],
            "action_id": 1,
            "action_text": "FLAP",
        }
        for episode_idx in range(260)
    ]

    output_dir = tmp_path / "export"
    dataset_dict = write_rollout_fixture(output_dir, rows, [], action_spec=action_spec_for_env("flappy"))
    table = pq.read_table(output_dir / "train.parquet")

    assert len(dataset_dict["train"]) == 260
    assert table.num_rows == 260
    assert table["episode_idx"].to_pylist() == list(range(260))


def test_hf_parquet_sorts_shards_and_replaces_existing_splits(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env

    output_dir = tmp_path / "export"
    rows_by_split = {
        "train": [
            (1, 1, 7),
            (1, 0, 5),
            (0, 0, 3),
        ],
        "val": [
            (2, 0, 11),
        ],
    }
    output_dir.mkdir()
    for split in rows_by_split:
        pq.write_table(pa.table({"episode_idx": [99]}), output_dir / f"{split}.parquet")
    for split, specs in rows_by_split.items():
        rows = []
        for episode_idx, decision_step, pixel in specs:
            image_path = (
                output_dir
                / "images"
                / split
                / f"episode_{episode_idx:06d}"
                / f"decision_step_{decision_step:06d}.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((2, 2, 3), pixel, dtype=np.uint8)).save(image_path)
            row = {
                "image": str(image_path),
                "env_name": "flappy",
                "episode_idx": episode_idx,
                "decision_step": decision_step,
                "split": split,
                "seed": 11 + episode_idx,
                "prompt": "pick action",
                "latency_raw_frames": 2,
                "latency_ms": 66.0,
                "raw_reward": 1.5,
                "action": [0.0, 1.0],
                "action_id": 1,
                "action_text": "FLAP",
            }
            rows.append(row)
        for _, episode_rows in groupby(rows, key=lambda row: row["episode_idx"]):
            write_episode_row_shard(output_dir, list(episode_rows))

    write_parquet_splits_from_shards(output_dir, action_spec=action_spec_for_env("flappy"))
    dataset_dict = load_parquet_splits(output_dir)
    train_table = pq.read_table(output_dir / "train.parquet")
    val_table = pq.read_table(output_dir / "val.parquet")

    assert len(dataset_dict["train"]) == 3
    assert len(dataset_dict["val"]) == 1
    assert train_table.num_rows == 3
    assert val_table["episode_idx"].to_pylist() == [2]
    assert [row["episode_idx"] for row in train_table.to_pylist()] == [0, 1, 1]
    assert [row["decision_step"] for row in train_table.to_pylist()] == [0, 0, 1]
    assert train_table.to_pylist()[0]["image"]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_load_written_parquet_splits_restores_image_feature(tmp_path):
    from datasets import Dataset, Features, Image, Value

    feats = Features({"image": Image(), "decision_step": Value("int64")})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    Dataset.from_dict({"image": [frame, frame], "decision_step": [0, 1]}, features=feats).to_parquet(
        str(tmp_path / "train.parquet")
    )
    Dataset.from_dict({"image": [frame], "decision_step": [0]}, features=feats).to_parquet(
        str(tmp_path / "val.parquet")
    )

    dd = load_parquet_splits(tmp_path)
    assert set(dd) == {"train", "val"}
    assert dd["train"].num_rows == 2
    assert dd["val"].num_rows == 1
    assert isinstance(dd["train"].features["image"], Image)
    assert dd["train"][0]["image"].size == (2, 2)
