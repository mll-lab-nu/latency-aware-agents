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
    "convert_deadly_corridor_history_to_starvla_lerobot"
)
CONVERTER = importlib.import_module(CONVERTER_MODULE)
ACTION_TEXT_BY_ID = {
    7: "MOVE_FORWARD + ATTACK",
    11: "MOVE_FORWARD + MOVE_RIGHT + ATTACK",
    21: "TURN_LEFT + MOVE_LEFT + ATTACK",
    33: "TURN_LEFT + MOVE_BACKWARD + MOVE_LEFT + ATTACK",
}


def _png(value: int) -> bytes:
    image = Image.fromarray(
        np.full((2, 3, 3), fill_value=value, dtype=np.uint8)
    )
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
        "action_text": ACTION_TEXT_BY_ID[action_id],
        "image": {"bytes": _png(image_value), "path": None},
        "prompt": "play deadly corridor with 6 raw-frame latency",
        "raw_reward": 0.25,
        "latency_raw_frames": 6,
        "latency_ms": 6000.0 / 35.0,
        "env_name": "deadly_corridor",
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
            "action_text": pa.array(
                [row["action_text"] for row in rows],
                type=pa.string(),
            ),
            "image": pa.array([row["image"] for row in rows], type=image_type),
            "prompt": pa.array(
                [row["prompt"] for row in rows],
                type=pa.string(),
            ),
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
            "split": pa.array(
                [row["split"] for row in rows],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(table, path)


def _pixel_value(image_entry: dict[str, Any]) -> int:
    image = Image.open(io.BytesIO(image_entry["bytes"])).convert("RGB")
    return int(np.asarray(image)[0, 0, 0])


def test_deadly_corridor_history_converter_builds_wanoft_context_dataset(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.parquet"
    _write_source(
        source_path,
        [
            _source_row(0, 0, 10, 11),
            _source_row(0, 1, 20, 7),
            _source_row(0, 2, 30, 33),
            _source_row(1, 0, 40, 21),
        ],
    )
    output_dir = tmp_path / "deadly_corridor_train__bridge"

    manifest = CONVERTER._convert_split(
        [source_path],
        output_dir,
        "latency-sensitive-bench/memory-rollouts",
        "deadly_corridor_fixed_latency_6_1000ep_7k2steps",
        "train",
        None,
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
    episode_zero_context = episode_zero[
        "observation.context_images"
    ].to_pylist()
    episode_one_context = episode_one[
        "observation.context_images"
    ].to_pylist()
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
    assert [_pixel_value(image) for image in episode_one_context[0]] == [
        40,
        40,
        40,
    ]
    assert episode_zero["action"].to_pylist() == [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ]
    assert episode_zero["timestamp"].to_pylist() == pytest.approx(
        [0.0, 1.0 / 8.75, 2.0 / 8.75]
    )
    assert episode_zero["done"].to_pylist() == [False, False, True]
    assert info["fps"] == 8.75
    assert info["features"]["observation.image"]["shape"] == [2, 3, 3]
    assert info["features"]["observation.context_images"]["shape"] == [
        3,
        2,
        3,
        3,
    ]
    assert prompt_map["6"]["latency"] == 6
    assert prompt_map["6"]["latency_raw_frames"] == 6
    assert prompt_map["6"]["latency_ms"] == pytest.approx(6000.0 / 35.0)
    assert manifest["context_source"] == "previous_episode_rows"
    assert manifest["source_action_layout"] == "deadly_corridor_joint_54"
    assert manifest["action_layout"] == "multibinary_7"
    assert manifest["latency_unit"] == "raw_frames"
    assert manifest["latency_raw_frames"] == [6]
    assert manifest["latency_env_steps"] == [1.5]
    assert manifest["episodes"] == 2
    assert manifest["frames"] == 4


def test_deadly_corridor_history_converter_rejects_action_text_mismatch(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.parquet"
    row = _source_row(0, 0, 10, 11)
    row["action_text"] = "ATTACK"
    _write_source(source_path, [row])

    with pytest.raises(ValueError, match="action_id/action_text mismatch"):
        CONVERTER._convert_split(
            [source_path],
            tmp_path / "output",
            "dataset",
            "config",
            "train",
            None,
            5,
            "observation.context_images",
            1,
        )


def test_deadly_corridor_history_converter_rejects_noncontiguous_episode_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.parquet"
    _write_source(
        source_path,
        [
            _source_row(0, 0, 10, 7),
            _source_row(1, 0, 20, 11),
            _source_row(0, 1, 30, 33),
        ],
    )

    with pytest.raises(
        ValueError,
        match="appears after it was already written",
    ):
        CONVERTER._convert_split(
            [source_path],
            tmp_path / "output",
            "dataset",
            "config",
            "train",
            None,
            5,
            "observation.context_images",
            2,
        )


def test_deadly_corridor_history_converter_keeps_context_across_source_shards(
    tmp_path: Path,
) -> None:
    first_shard = tmp_path / "train-00000.parquet"
    second_shard = tmp_path / "train-00001.parquet"
    _write_source(first_shard, [_source_row(0, 0, 10, 7)])
    _write_source(
        second_shard,
        [
            _source_row(0, 1, 20, 11),
            _source_row(1, 0, 30, 21),
        ],
    )
    output_dir = tmp_path / "output"

    CONVERTER._convert_split(
        [first_shard, second_shard],
        output_dir,
        "dataset",
        "config",
        "train",
        None,
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


def test_deadly_corridor_history_converter_resolves_memory_rollout_shards() -> None:
    paths = CONVERTER._source_shard_paths(
        [
            "README.md",
            (
                "deadly_corridor_fixed_latency_6_1000ep_7k2steps/"
                "train-00001-of-00002.parquet"
            ),
            (
                "deadly_corridor_fixed_latency_6_1000ep_7k2steps/"
                "train-00000-of-00002.parquet"
            ),
            (
                "deadly_corridor_fixed_latency_6_1000ep_7k2steps/"
                "val-00000-of-00001.parquet"
            ),
        ],
        "deadly_corridor_fixed_latency_6_1000ep_7k2steps",
        "train",
    )

    assert paths == [
        (
            "deadly_corridor_fixed_latency_6_1000ep_7k2steps/"
            "train-00000-of-00002.parquet"
        ),
        (
            "deadly_corridor_fixed_latency_6_1000ep_7k2steps/"
            "train-00001-of-00002.parquet"
        ),
    ]
