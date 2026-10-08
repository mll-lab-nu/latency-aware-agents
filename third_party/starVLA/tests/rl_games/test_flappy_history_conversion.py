from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


CONVERTER_MODULE = (
    "examples.rl_games.bash_scripts.gr00t.data_conversion."
    "convert_flappy_history_to_starvla_lerobot"
)
CONVERTER = importlib.import_module(CONVERTER_MODULE)


def _png(value: int) -> bytes:
    image = Image.fromarray(np.full((2, 2, 3), fill_value=value, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source_row(
    episode_idx: int,
    decision_step: int,
    image_value: int,
    action_id: int,
) -> dict[str, Any]:
    return {
        "episode_idx": episode_idx,
        "decision_step": decision_step,
        "action_id": action_id,
        "image": {"bytes": _png(image_value), "path": None},
        "prompt": "play flappy with latency 3",
        "raw_reward": 0.1,
        "latency_raw_frames": 3,
        "latency_ms": 100.0,
        "env_name": "flappy",
        "split": "train",
    }


def _write_source(path: Path, rows: list[dict[str, Any]]) -> None:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "episode_idx": pa.array(
                [row["episode_idx"] for row in rows],
                type=pa.int64(),
            ),
            "decision_step": pa.array(
                [row["decision_step"] for row in rows],
                type=pa.int64(),
            ),
            "action_id": pa.array(
                [row["action_id"] for row in rows],
                type=pa.int64(),
            ),
            "image": pa.array([row["image"] for row in rows], type=image_type),
            "prompt": pa.array([row["prompt"] for row in rows], type=pa.string()),
            "raw_reward": pa.array(
                [row["raw_reward"] for row in rows],
                type=pa.float64(),
            ),
            "latency_raw_frames": pa.array(
                [row["latency_raw_frames"] for row in rows],
                type=pa.int64(),
            ),
            "latency_ms": pa.array(
                [row["latency_ms"] for row in rows],
                type=pa.float64(),
            ),
            "env_name": pa.array(
                [row["env_name"] for row in rows],
                type=pa.string(),
            ),
            "split": pa.array([row["split"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _pixel_value(image_entry: dict[str, Any]) -> int:
    image = Image.open(io.BytesIO(image_entry["bytes"])).convert("RGB")
    return int(np.asarray(image)[0, 0, 0])


def test_flappy_history_converter_downloads_each_source_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_paths = [
        "config/train-00000-of-00002.parquet",
        "config/train-00001-of-00002.parquet",
    ]
    downloaded_paths: list[str] = []

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        repo_type: str,
        cache_dir: str | None,
        revision: str,
    ) -> str:
        assert repo_id == "latency-sensitive-bench/memory-rollouts"
        assert repo_type == "dataset"
        assert cache_dir == "/cache"
        assert revision == "fixed-memory-data-revision"
        downloaded_paths.append(filename)
        return str(tmp_path / Path(filename).name)

    monkeypatch.setattr(CONVERTER, "hf_hub_download", fake_hf_hub_download)

    local_paths = list(
        CONVERTER._downloaded_hub_shards(
            "latency-sensitive-bench/memory-rollouts",
            repo_paths,
            "/cache",
            "fixed-memory-data-revision",
        )
    )

    assert downloaded_paths == repo_paths
    assert local_paths == [tmp_path / Path(path).name for path in repo_paths]


def test_flappy_history_converter_builds_context_with_episode_local_left_padding(
    tmp_path: Path,
) -> None:
    converter = CONVERTER
    source_path = tmp_path / "train.parquet"
    _write_source(
        source_path,
        [
            _source_row(0, 0, 10, 0),
            _source_row(0, 1, 20, 1),
            _source_row(0, 2, 30, 0),
            _source_row(1, 0, 40, 1),
            _source_row(1, 1, 50, 0),
        ],
    )
    output_dir = tmp_path / "flappy_train__bridge"

    manifest = converter._convert_split(
        [source_path],
        output_dir,
        "latency-sensitive-bench/memory-rollouts",
        "flappy_fixed_latency_3_200ep_7k2steps",
        "train",
        None,
        "bridge",
        4,
        "observation.context_images",
        2,
    )

    episode_zero = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    )
    episode_one = pq.read_table(
        output_dir / "data/chunk-000/episode_000001.parquet"
    )
    episode_zero_context = episode_zero["observation.context_images"].to_pylist()
    episode_one_context = episode_one["observation.context_images"].to_pylist()
    info = json.loads(
        (output_dir / "meta/info.json").read_text(encoding="utf-8")
    )
    prompt_map = json.loads(
        (output_dir / "latency_prompt_map.json").read_text(encoding="utf-8")
    )

    assert [
        [_pixel_value(image) for image in context]
        for context in episode_zero_context
    ] == [
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 20],
    ]
    assert [_pixel_value(image) for image in episode_one_context[0]] == [40, 40, 40]
    assert episode_zero["action"].to_pylist() == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert episode_zero["done"].to_pylist() == [False, False, True]
    assert info["features"]["observation.image"]["shape"] == [2, 2, 3]
    assert info["features"]["observation.context_images"]["shape"] == [3, 2, 2, 3]
    assert manifest["context_source"] == "previous_episode_rows"
    assert manifest["source_config"] == "flappy_fixed_latency_3_200ep_7k2steps"
    assert manifest["latency_metadata"] is True
    assert manifest["latency_unit"] == "raw_frames"
    assert manifest["latency_raw_frames"] == [3]
    assert manifest["latency_env_steps"] == [3]
    assert manifest["latency_filter"] == [3]
    assert prompt_map["3"]["latency_raw_frames"] == 3
    assert manifest["episodes"] == 2
    assert manifest["frames"] == 5


def test_flappy_history_converter_rejects_noncontiguous_episode_rows(
    tmp_path: Path,
) -> None:
    converter = CONVERTER
    source_path = tmp_path / "train.parquet"
    _write_source(
        source_path,
        [
            _source_row(0, 0, 10, 0),
            _source_row(1, 0, 20, 0),
            _source_row(0, 1, 30, 0),
        ],
    )

    with pytest.raises(ValueError, match="appears after it was already written"):
        converter._convert_split(
            [source_path],
            tmp_path / "output",
            "dataset",
            "config",
            "train",
            None,
            "bridge",
            5,
            "observation.context_images",
            2,
        )


def test_flappy_history_converter_keeps_episode_context_across_source_shards(
    tmp_path: Path,
) -> None:
    converter = CONVERTER
    first_shard = tmp_path / "train-00000.parquet"
    second_shard = tmp_path / "train-00001.parquet"
    _write_source(first_shard, [_source_row(0, 0, 10, 0)])
    _write_source(
        second_shard,
        [
            _source_row(0, 1, 20, 1),
            _source_row(1, 0, 30, 0),
        ],
    )
    output_dir = tmp_path / "output"

    converter._convert_split(
        [first_shard, second_shard],
        output_dir,
        "dataset",
        "config",
        "train",
        None,
        "bridge",
        3,
        "observation.context_images",
        1,
    )

    episode_zero = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    )
    contexts = episode_zero["observation.context_images"].to_pylist()
    assert [
        [_pixel_value(image) for image in context]
        for context in contexts
    ] == [[10, 10], [10, 10]]
    assert episode_zero["decision_step"].to_pylist() == [0, 1]


def test_flappy_history_converter_resolves_memory_rollout_shards() -> None:
    converter = CONVERTER

    paths = converter._source_shard_paths(
        [
            "README.md",
            "flappy_fixed_latency_3_200ep_7k2steps/train-00001-of-00002.parquet",
            "flappy_fixed_latency_3_200ep_7k2steps/train-00000-of-00002.parquet",
            "flappy_fixed_latency_3_200ep_7k2steps/val-00000-of-00001.parquet",
        ],
        "flappy_fixed_latency_3_200ep_7k2steps",
        "train",
    )

    assert paths == [
        "flappy_fixed_latency_3_200ep_7k2steps/train-00000-of-00002.parquet",
        "flappy_fixed_latency_3_200ep_7k2steps/train-00001-of-00002.parquet",
    ]
