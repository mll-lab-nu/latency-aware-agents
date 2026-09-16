from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import save_test_image


def _dagger_arrays(rows: int = 1) -> dict[str, np.ndarray]:
    termination = np.zeros((rows, 5), dtype=bool)
    termination[:, -1] = True
    return {
        "state": np.arange(rows * 53, dtype=np.float32).reshape(rows, 53),
        "action": np.arange(rows * 34, dtype=np.float32).reshape(rows, 34),
        "termination": termination,
        "actor_observation": np.arange(rows * 5 * 753, dtype=np.float32).reshape(
            rows, 5, 753
        ),
        "image_shape": np.asarray([2, 2, 3], dtype=np.int64),
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
            "0.1",
            "-r",
            "10",
            str(video),
        ],
        check=True,
    )


def test_parkour_dagger_shard_requires_all_numeric_fields(tmp_path):
    from latency_bench.data.parkour_dagger import write_parkour_dagger_shard

    arrays = _dagger_arrays()
    del arrays["termination"]
    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    save_test_image(image, 7)
    _save_test_video(image, video)
    with pytest.raises(KeyError):
        write_parkour_dagger_shard(
            tmp_path,
            split="train",
            episode_idx=0,
            arrays=arrays,
            video_path=video,
        )


def test_parkour_dagger_converter_writes_video_and_numeric_parquet(tmp_path):
    from latency_bench.data.parkour_dagger import (
        convert_parkour_dagger_shards_to_lerobot,
        write_parkour_dagger_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    for episode_idx in range(2):
        write_parkour_dagger_shard(
            raw,
            split="train",
            episode_idx=episode_idx,
            arrays=_dagger_arrays(),
            video_path=video,
        )
    with np.load(
        raw / "rollout_shards/train/episode_000000/episode.npz"
    ) as payload:
        assert set(payload.files) == {*_dagger_arrays(), "image_shape"}
    output = tmp_path / "lerobot"
    manifest = convert_parkour_dagger_shards_to_lerobot(
        raw,
        output,
        force=True,
        expected_rows=2,
        round_index=0,
    )
    assert manifest["action_dim"] == 34
    assert manifest["state_dim"] == 53
    assert manifest["frames"] == 2
    assert not list(output.rglob("*.png"))
    assert (
        output / "videos/chunk-000/observation.image/episode_000000.mp4"
    ).is_file()
    row = pq.read_table(
        output / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    assert len(row["observation.state"]) == 53
    assert len(row["action"]) == 34
    assert len(row["actor_observation"]) == 5
    assert len(row["actor_observation"][0]) == 753
    assert "oracle_yaw" not in row
    assert row["termination"] == [False, False, False, False, True]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.image"]["dtype"] == "video"
    assert info["features"]["actor_observation"]["shape"] == [5, 753]
    assert "reward" not in info["features"]
    assert "teacher_action" not in info["features"]
    task = json.loads((output / "meta/tasks.jsonl").read_text())
    assert task["task"] == "Traverse the parkour course and follow each waypoint."


def test_parkour_chunk_export_uses_issued_command_anchor(tmp_path):
    from latency_bench.data.parkour_dagger import (
        convert_parkour_dagger_shards_to_lerobot,
        write_parkour_dagger_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _dagger_arrays()
    arrays["action"] = np.arange(40 * 34, dtype=np.float32).reshape(1, 40, 34)
    arrays.update({
        "episode_id": np.array([3]), "obs_id": np.array([2]),
        "issued_raw_frame": np.array([10]), "ready_raw_frame": np.array([16]),
        "latency_ms": np.array([120.0]), "worker_id": np.array([0]),
        "raw_reward": np.array([1.0]),
        "control_applied_command": np.concatenate((np.zeros((16, 34)), arrays["action"][0, 6:10])),
        "control_source_raw_frame": np.array([-1] * 16 + [10] * 4),
        "control_source_obs_id": np.array([-1] * 16 + [2] * 4),
        "control_chunk_index": np.array([-1] * 16 + [6, 7, 8, 9]),
        "control_reward": np.ones(20),
        "control_done": np.array([False] * 19 + [True]),
    })
    write_parkour_dagger_shard(
        raw,
        split="train",
        episode_idx=0,
        arrays=arrays,
        video_path=video,
    )

    output = tmp_path / "lerobot"
    with np.load(raw / "rollout_shards/train/episode_000000/episode.npz") as payload:
        for name in arrays:
            np.testing.assert_array_equal(payload[name], arrays[name])
    convert_parkour_dagger_shards_to_lerobot(raw, output, force=True, video_shards=True)
    row = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()[0]
    assert "issued_command_chunk" in row
    assert "action" not in row
    assert row["issued_raw_frame"] == 10
    assert row["latency"] == 120.0
    assert row["latency_raw_frames"] == 6
    info = json.loads((output / "meta/info.json").read_text())
    assert "issued_command_chunk" in info["features"]
    assert "action" not in info["features"]
    modality = json.loads((output / "meta/modality.json").read_text())
    assert modality["action"]["issued_command_chunk"]["original_key"] == "issued_command_chunk"
    del arrays["ready_raw_frame"]
    with pytest.raises(KeyError, match="ready_raw_frame"):
        write_parkour_dagger_shard(raw, split="train", episode_idx=1, arrays=arrays, video_path=video)


def test_parkour_video_bootstrap_preserves_rollout_schema(tmp_path):
    from latency_bench.data.parkour_dagger import (
        convert_parkour_dagger_shards_to_lerobot,
        write_parkour_dagger_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _dagger_arrays(rows=2)
    arrays["raw_reward"] = np.asarray([2.5, 3.5], dtype=np.float32)
    write_parkour_dagger_shard(
        raw,
        split="train",
        episode_idx=0,
        arrays=arrays,
        video_path=video,
    )
    write_parkour_dagger_shard(
        raw,
        split="val",
        episode_idx=1,
        arrays=arrays,
        video_path=video,
    )

    output = tmp_path / "lerobot"
    manifest = convert_parkour_dagger_shards_to_lerobot(
        raw,
        output,
        force=True,
        video_shards=True,
    )
    rows = pq.read_table(
        output / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()
    assert set(rows[0]) == {
        "observation.image",
        "observation.state",
        "action",
        "termination",
        "actor_observation",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
        "latency",
        "latency_raw_frames",
        "done",
        "reward",
        "density_weight",
    }
    assert [row["reward"] for row in rows] == [2.5, 3.5]
    assert [row["done"] for row in rows] == [False, True]
    assert [row["density_weight"] for row in rows] == [1.0, 1.0]
    assert [row["latency"] for row in rows] == [0, 0]
    assert [row["latency_raw_frames"] for row in rows] == [0, 0]
    info = json.loads((output / "meta/info.json").read_text())
    assert "latency" in info["features"]
    assert "latency_raw_frames" not in info["features"]
    assert "done" not in info["features"]
    assert "reward" not in info["features"]
    assert manifest["integration_name"] == "isaaclab_parkour"
    assert manifest["rows_unit"] == "decision_step"
    assert manifest["obs_fps"] == 10
    assert manifest["obs_stride_raw_frames"] == 5
    assert manifest["state_labels"] == [
        f"parkour_proprio_{index}" for index in range(53)
    ]
    validation_manifest = json.loads(
        (output.with_name(f"{output.name}__val") / "manifest.json").read_text()
    )
    assert validation_manifest["dataset_name"] == "lerobot__val"
    assert validation_manifest["split"] == "val"
    assert validation_manifest["training_dataset_name"] == "lerobot"


def test_parkour_bootstrap_rows_preserve_actor_observation_and_termination(tmp_path):
    from latency_bench.data.action_layouts import (
        GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC,
    )
    from latency_bench.data.hf_parquet import (
        write_episode_row_shard,
        write_parquet_splits_from_shards,
    )

    image = tmp_path / "frame.png"
    save_test_image(image, 7)
    row = {
        "image": str(image),
        "env_name": "extreme_parkour_go2",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 1,
        "prompt": "Traverse the parkour course and follow each waypoint.",
        "latency_raw_frames": 0,
        "latency_ms": 0.0,
        "raw_reward": 0.0,
        "action": [0.0] * 34,
        "action_text": "terrain_latent_yaw",
        "state": [0.0] * 53,
        "actor_observation": [[float(i)] * 753 for i in range(5)],
        "termination": [False, False, False, False, True],
    }
    write_episode_row_shard(tmp_path, [row])
    write_parquet_splits_from_shards(
        tmp_path,
        action_spec=GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC,
        keep_dataset=False,
    )

    row_out = pq.read_table(tmp_path / "train.parquet").to_pylist()[0]
    assert len(row_out["actor_observation"]) == 5
    assert len(row_out["actor_observation"][0]) == 753
    assert row_out["termination"] == [False, False, False, False, True]
