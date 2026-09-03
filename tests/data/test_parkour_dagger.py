from __future__ import annotations

import json
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import save_test_image


def _dagger_arrays() -> dict[str, np.ndarray]:
    return {
        "state": np.arange(53, dtype=np.float32).reshape(1, 53),
        "action": np.arange(34, dtype=np.float32).reshape(1, 34),
        "termination": np.asarray([[False, False, False, False, True]]),
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
    assert "oracle_yaw" not in row
    assert row["termination"] == [False, False, False, False, True]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.image"]["dtype"] == "video"
    assert "actor_observation" not in info["features"]
    assert "teacher_action" not in info["features"]
    task = json.loads((output / "meta/tasks.jsonl").read_text())
    assert task["task"] == "Traverse the parkour course and follow each waypoint."


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
