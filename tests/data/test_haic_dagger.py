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
        "action": np.arange(rows * 256, dtype=np.float32).reshape(rows, 256),
        "actor_input": np.arange(rows * 605, dtype=np.float32).reshape(rows, 605),
        "teacher_action": np.arange(rows * 23, dtype=np.float32).reshape(rows, 23),
        "termination": np.zeros(rows, dtype=bool),
        "image_shape": np.asarray([2, 2, 3], dtype=np.int64),
    }


def _dagger_arrays(rows: int = 2) -> dict[str, np.ndarray]:
    arrays = _bootstrap_arrays(rows)
    arrays["termination"][-1] = True
    return arrays


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
            "0.04",
            "-r",
            "50",
            str(video),
        ],
        check=True,
    )


@pytest.mark.parametrize("issued_chunk", [False, True])
def test_haic_dagger_converter_keeps_exact_numeric_contract(tmp_path, issued_chunk):
    from latency_bench.data.haic_dagger import (
        convert_haic_dagger_shards_to_lerobot,
        write_haic_dagger_shard,
    )

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    raw = tmp_path / "raw"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _dagger_arrays()
    if issued_chunk:
        for key in ("action", "actor_input", "teacher_action"):
            arrays[key] = np.repeat(arrays[key][:, None], 40, axis=1)
        arrays["termination"] = np.zeros((2, 40), dtype=bool)
        arrays["termination"][-1, -1] = True
        arrays.update(
            episode_id=np.zeros(2, dtype=np.int64),
            obs_id=np.array([0, 5]),
            issued_raw_frame=np.array([0, 5]),
            ready_raw_frame=np.array([3, 8]),
            latency_ms=np.array([60.0, 60.0]),
            worker_id=np.zeros(2, dtype=np.int64),
            control_applied_command=np.zeros((10, 256), dtype=np.float32),
            control_source_raw_frame=np.full(10, -1),
            control_source_obs_id=np.full(10, -1),
            control_chunk_index=np.full(10, -1),
            control_reward=np.zeros(10, dtype=np.float32),
            control_done=np.arange(10) == 9,
        )
    action_key = "issued_command_chunk" if issued_chunk else "action"
    horizon = [40] if issued_chunk else []
    write_haic_dagger_shard(
        raw,
        split="train",
        episode_idx=0,
        arrays=arrays,
        video_path=video,
    )
    output = tmp_path / "lerobot"
    manifest = convert_haic_dagger_shards_to_lerobot(
        raw, output, force=True, expected_rows=2, round_index=0
    )
    assert manifest["state_dim"] == 605
    assert manifest["action_dim"] == 256
    assert manifest["teacher_latent_dim"] == 256
    assert manifest["teacher_action_dim"] == 23
    assert manifest["action_horizon"] == 40
    assert manifest["actor_input_shape"] == horizon + [605]
    assert manifest["teacher_action_shape"] == horizon + [23]
    assert manifest["termination_shape"] == horizon
    assert manifest["termination_count"] == 1
    row = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()[0]
    assert len(row["observation.state"]) == 605
    assert np.asarray(row[action_key]).shape == tuple(horizon + [256])
    assert np.asarray(row["haic_actor_input"]).shape == tuple(horizon + [605])
    assert np.asarray(row["haic_teacher_action"]).shape == tuple(horizon + [23])
    assert not np.any(row["haic_termination"])
    np.testing.assert_array_equal(row[action_key], arrays["action"][0])
    if issued_chunk:
        assert row["issued_raw_frame"] == 0
        assert row["ready_raw_frame"] == 3
        assert row["latency"] == 60.0
    assert row["done"] is False
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"][action_key]["shape"] == horizon + [256]
    assert info["fps"] == 50.0
    assert info["features"]["haic_actor_input"]["shape"] == horizon + [605]
    assert info["features"]["haic_teacher_action"]["shape"] == horizon + [23]
    assert info["features"]["haic_termination"]["shape"] == (horizon or [1])
    assert "object_target" not in info["features"]


def test_haic_bootstrap_converter_keeps_action_distillation_targets(tmp_path):
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
        "haic_actor_input",
        "haic_teacher_action",
        "haic_termination",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
        "done",
    }


def test_haic_dagger_prepare_copies_bootstrap_stats(tmp_path, monkeypatch):
    from scripts.official_gr00t import prepare_haic_dataset as prepare_module

    source = tmp_path / "bootstrap/meta/stats.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"fixed": [1.0]}\n')
    output = tmp_path / "lerobot"

    def fake_convert(*args, **kwargs):
        (output / "meta").mkdir(parents=True)
        return {"dataset_name": output.name}

    monkeypatch.setattr(
        prepare_module, "convert_haic_dagger_shards_to_lerobot", fake_convert
    )
    monkeypatch.setattr(
        prepare_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("DAgger stats must not be recomputed"),
    )

    manifest = prepare_module.prepare_haic_dataset(
        tmp_path / "raw",
        output,
        force=False,
        dagger=True,
        normalization_source=tmp_path / "bootstrap",
    )

    assert manifest == {"dataset_name": "lerobot"}
    assert (output / "meta/stats.json").read_bytes() == source.read_bytes()


def test_haic_dagger_shard_requires_all_fields(tmp_path):
    from latency_bench.data.haic_dagger import write_haic_dagger_shard

    image = tmp_path / "frame.png"
    video = tmp_path / "episode.mp4"
    save_test_image(image, 7)
    _save_test_video(image, video)
    arrays = _dagger_arrays()
    del arrays["termination"]
    with pytest.raises(KeyError):
        write_haic_dagger_shard(
            tmp_path,
            split="train",
            episode_idx=0,
            arrays=arrays,
            video_path=video,
        )


def test_failed_shard_write_leaves_no_partial_episode(tmp_path, monkeypatch):
    from latency_bench.data.haic_dagger import write_haic_dagger_shard

    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        write_haic_dagger_shard(
            tmp_path, split="train", episode_idx=0,
            arrays=_dagger_arrays(), video_path=video,
        )
    assert list((tmp_path / "rollout_shards/train").iterdir()) == []
