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
    assert manifest["official_action_horizon"] == 1
    validation_manifest = json.loads(
        (output.with_name(f"{output.name}__val") / "manifest.json").read_text()
    )
    assert validation_manifest["dataset_name"] == "lerobot__val"
    assert validation_manifest["split"] == "val"
    assert validation_manifest["training_dataset_name"] == "lerobot"
    assert validation_manifest["official_action_horizon"] == 1


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


def test_dagger_prepare_copies_bootstrap_stats(tmp_path, monkeypatch):
    from scripts.official_gr00t import prepare_parkour_dataset as prepare_module

    output = tmp_path / "lerobot"
    val_output = tmp_path / "lerobot__val"
    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "meta").mkdir(parents=True)
    bootstrap_stats = b'{"action":{"mean":[1.0]}}\n'
    (bootstrap / "meta/stats.json").write_bytes(bootstrap_stats)
    commands = []

    def fake_convert(*args, **kwargs):
        (output / "meta").mkdir(parents=True)
        (val_output / "meta").mkdir(parents=True)
        return {"dataset_name": output.name}

    monkeypatch.setattr(
        prepare_module, "convert_parkour_dagger_shards_to_lerobot", fake_convert
    )
    monkeypatch.setattr(
        prepare_module.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    manifest = prepare_module.prepare_parkour_dataset(
        tmp_path / "raw",
        output,
        force=False,
        dagger=True,
        normalization_source=bootstrap,
    )

    assert manifest == {"dataset_name": "lerobot"}
    assert commands == []
    assert (output / "meta/stats.json").read_bytes() == bootstrap_stats
    assert (val_output / "meta/stats.json").read_bytes() == bootstrap_stats


def test_video_shard_prepare_runs_stats_for_train_and_validation(tmp_path, monkeypatch):
    from scripts.official_gr00t import prepare_parkour_dataset as prepare_module

    output = tmp_path / "lerobot"
    val_output = tmp_path / "lerobot__val"
    calls = []

    def fake_convert(*args, **kwargs):
        assert kwargs["video_shards"] is True
        output.mkdir(parents=True)
        val_output.mkdir(parents=True)
        return {"dataset_name": output.name}

    monkeypatch.setattr(
        prepare_module, "convert_parkour_dagger_shards_to_lerobot", fake_convert
    )
    def fake_run(command, **kwargs):
        calls.append(command)
        dataset_dir = Path(command[command.index("--dataset-path") + 1])
        (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "meta/stats.json").write_text(
            json.dumps(
                {
                    "action": {
                        "min": [0.0] * 34,
                        "max": [0.0] * 34,
                    }
                }
            )
        )

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_run)

    manifest = prepare_module.prepare_parkour_dataset(
        tmp_path / "raw",
        output,
        force=False,
        video_shards=True,
    )

    assert manifest == {"dataset_name": "lerobot"}
    assert [command[command.index("--dataset-path") + 1] for command in calls] == [
        str(output),
        str(val_output),
    ]
    for dataset_dir in (output, val_output):
        stats = json.loads((dataset_dir / "meta/stats.json").read_text())
        assert stats["action"]["min"] == [-1.0] * 32 + [-np.pi / 1.5] * 2
        assert stats["action"]["max"] == [1.0] * 32 + [np.pi / 1.5] * 2
