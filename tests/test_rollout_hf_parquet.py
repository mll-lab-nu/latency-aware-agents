from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _save_image(path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.full((2, 2, 3), 7, dtype=np.uint8)).save(path)


def test_hf_parquet_writes_canonical_columns_only(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    _save_image(image_path)
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
        "reward": 1.5,
        "action_id": 1,
        "action": [0.0, 1.0],
        "action_text": "FLAP",
    }

    dataset_dict = write_parquet_splits(
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
    assert table.to_pylist()[0]["image"]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_hf_parquet_shards_sort_by_decision_step(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_episode_row_shard, write_parquet_splits_from_shards

    image_path = tmp_path / "image.png"
    _save_image(image_path)
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
            "reward": 1.5,
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
