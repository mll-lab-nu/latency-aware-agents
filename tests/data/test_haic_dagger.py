from __future__ import annotations

import json
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import save_test_image


def _bootstrap_arrays(rows: int = 2) -> dict[str, np.ndarray]:
    return {
        "state": np.arange(rows * 605, dtype=np.float32).reshape(rows, 605),
        "action": np.arange(rows * 275, dtype=np.float32).reshape(rows, 275),
        "image_shape": np.asarray([2, 2, 3], dtype=np.int64),
    }


def _dagger_arrays(rows: int = 2) -> dict[str, np.ndarray]:
    return {
        **_bootstrap_arrays(rows),
        "actor_input": np.arange(rows * 5 * 605, dtype=np.float32).reshape(
            rows, 5, 605
        ),
        "teacher_action": np.arange(rows * 5 * 23, dtype=np.float32).reshape(
            rows, 5, 23
        ),
        "termination": np.asarray([[False, False, False, False, True]] * rows),
        "is_init": np.asarray([[True, False, False, False, False]] * rows),
    }


def _save_test_video(image, video):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-t",
            "0.2",
            "-r",
            "10",
            str(video),
        ],
        check=True,
    )


def test_haic_dagger_converter_keeps_exact_numeric_contract(tmp_path):
    from latency_bench.data.haic_dagger import (
        convert_haic_dagger_shards_to_lerobot,
        write_haic_dagger_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    write_haic_dagger_shard(
        raw,
        split="train",
        episode_idx=0,
        arrays=_dagger_arrays(),
        video_path=video,
    )
    output = tmp_path / "lerobot"
    manifest = convert_haic_dagger_shards_to_lerobot(
        raw, output, force=True, expected_rows=2, round_index=0
    )
    assert manifest["state_dim"] == 605
    assert manifest["action_dim"] == 275
    assert manifest["teacher_labels"] == 10
    row = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()[0]
    assert len(row["observation.state"]) == 605
    assert len(row["action"]) == 275
    assert len(row["actor_input"]) == 5
    assert len(row["actor_input"][0]) == 605
    assert len(row["teacher_action"]) == 5
    assert len(row["teacher_action"][0]) == 23
    assert row["done"] is True
    assert row["is_init"] == [True, False, False, False, False]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [275]
    assert info["features"]["actor_input"]["shape"] == [5, 605]
    assert info["features"]["teacher_action"]["shape"] == [5, 23]
    assert "privileged_target" not in row
    assert "object_target" not in row
    assert "privileged_target" not in info["features"]
    assert "object_target" not in info["features"]


def test_haic_bootstrap_converter_omits_dagger_targets(tmp_path):
    from latency_bench.data.haic_dagger import (
        convert_haic_bootstrap_shards_to_lerobot,
        write_haic_bootstrap_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _bootstrap_arrays()
    write_haic_bootstrap_shard(
        raw,
        split="train",
        episode_idx=0,
        arrays=arrays,
        video_path=video,
    )
    output = tmp_path / "lerobot"
    convert_haic_bootstrap_shards_to_lerobot(raw, output, force=True)
    row = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()[0]
    assert set(row) == {
        "observation.image",
        "observation.state",
        "action",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
        "done",
    }


def test_haic_dagger_shard_requires_all_fields(tmp_path):
    from latency_bench.data.haic_dagger import write_haic_dagger_shard

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _dagger_arrays()
    del arrays["is_init"]
    with pytest.raises(KeyError):
        write_haic_dagger_shard(
            tmp_path,
            split="train",
            episode_idx=0,
            arrays=arrays,
            video_path=video,
        )
