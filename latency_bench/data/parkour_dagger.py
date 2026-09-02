"""Video-backed LeRobot export for the Go2 Parkour DAgger contract.

The collector commits one episode as a directory containing an ``episode.npz``
numeric payload and an ``episode.mp4`` RGB video. A temporary directory is
renamed into place only after both files are complete, so readers never see a
half-written episode and the large DAgger path never creates per-frame PNGs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latency_bench.data.action_layouts import (
    GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC,
)
from latency_bench.data.official_gr00t import _write_official_gr00t_modality


PARKOUR_VLA_STATE_DIM = 53
PARKOUR_VLA_ACTION_DIM = 34
PARKOUR_ACTOR_OBSERVATION_HORIZON = 5
PARKOUR_ACTOR_OBSERVATION_DIM = 753
PARKOUR_TEACHER_ACTION_DIM = 12

PARKOUR_DAGGER_LABELS_PER_ROW = PARKOUR_ACTOR_OBSERVATION_HORIZON

_SHARD_ROOT = "rollout_shards"
_SHARD_PREFIX = "episode_"
_DATASET_FPS = 10.0
_CHUNK_SIZE = 1000
_NUMERIC_FIELDS = (
    "state",
    "action",
    "actor_observation",
    "teacher_action",
    "termination",
)


def _parkour_dagger_shard_dir(
    output_dir: Path | str,
    split: str,
    episode_idx: int,
) -> Path:
    """Return the committed directory for one collector episode shard."""
    return (
        Path(output_dir) / _SHARD_ROOT / split / f"{_SHARD_PREFIX}{episode_idx:06d}"
    )


def write_parkour_dagger_shard(
    output_dir: Path | str,
    *,
    split: str,
    episode_idx: int,
    arrays: Mapping[str, Any],
    video_path: Path | str,
) -> Path:
    """Atomically commit one numeric NPZ + MP4 Parkour DAgger shard.

    ``arrays`` must contain the five numeric fields named by
    ``_NUMERIC_FIELDS``.
    """
    payload = {name: arrays[name] for name in _NUMERIC_FIELDS}
    payload["image_shape"] = arrays["image_shape"]

    target = _parkour_dagger_shard_dir(output_dir, split, episode_idx)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    np.savez_compressed(temporary / "episode.npz", **payload)
    shutil.copyfile(video_path, temporary / "episode.mp4")
    temporary.replace(target)
    return target


def _shard_dirs(output_dir: Path | str, split: str) -> list[Path]:
    return sorted(
        path
        for path in Path(output_dir).glob(
            f"**/{_SHARD_ROOT}/{split}/{_SHARD_PREFIX}*"
        )
        if path.is_dir()
    )


def _iter_parkour_dagger_shards(
    output_dir: Path | str,
    split: str,
) -> Iterator[tuple[dict[str, Any], Path]]:
    """Yield committed shard records in deterministic episode order."""
    for shard_dir in _shard_dirs(output_dir, split):
        with np.load(shard_dir / "episode.npz", allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in _NUMERIC_FIELDS}
            arrays["image_shape"] = payload["image_shape"]
        yield arrays, shard_dir / "episode.mp4"


def _episode_table(arrays: Mapping[str, Any], episode_index: int) -> pa.Table:
    row_count = arrays["state"].shape[0]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    return pa.table(
        {
            "observation.image": pa.array(
                [{"bytes": None, "path": None} for _ in range(row_count)],
                type=image_type,
            ),
            "observation.state": pa.array(
                arrays["state"].tolist(),
                type=pa.list_(pa.float32(), PARKOUR_VLA_STATE_DIM),
            ),
            "action": pa.array(
                arrays["action"].tolist(),
                type=pa.list_(pa.float32(), PARKOUR_VLA_ACTION_DIM),
            ),
            "actor_observation": pa.array(
                arrays["actor_observation"].tolist(),
                type=pa.list_(
                    pa.list_(pa.float32(), PARKOUR_ACTOR_OBSERVATION_DIM),
                    PARKOUR_ACTOR_OBSERVATION_HORIZON,
                ),
            ),
            "teacher_action": pa.array(
                arrays["teacher_action"].tolist(),
                type=pa.list_(
                    pa.list_(pa.float32(), PARKOUR_TEACHER_ACTION_DIM),
                    PARKOUR_ACTOR_OBSERVATION_HORIZON,
                ),
            ),
            "oracle_yaw": pa.array(
                (1.5 * arrays["action"][:, -2:]).tolist(),
                type=pa.list_(pa.float32(), 2),
            ),
            "termination": pa.array(
                arrays["termination"].tolist(),
                type=pa.list_(pa.bool_(), PARKOUR_ACTOR_OBSERVATION_HORIZON),
            ),
            "timestamp": pa.array(
                np.arange(row_count, dtype=np.float64) / _DATASET_FPS,
                type=pa.float64(),
            ),
            "episode_index": pa.array(
                [episode_index for _ in range(row_count)], type=pa.int64()
            ),
            "frame_index": pa.array(
                np.arange(row_count, dtype=np.int64), type=pa.int64()
            ),
            "task_index": pa.array([0 for _ in range(row_count)], type=pa.int64()),
            "done": pa.array(arrays["termination"][:, -1], type=pa.bool_()),
        }
    )


def _write_info(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    image_shape: tuple[int, int, int],
) -> None:
    height, width, channels = image_shape
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": _DATASET_FPS,
                "video.channels": channels,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [PARKOUR_VLA_STATE_DIM],
            "names": [
                f"parkour_proprio_{index}" for index in range(PARKOUR_VLA_STATE_DIM)
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [PARKOUR_VLA_ACTION_DIM],
            "names": list(GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels),
        },
        "actor_observation": {
            "dtype": "tensor",
            "shape": [
                PARKOUR_ACTOR_OBSERVATION_HORIZON,
                PARKOUR_ACTOR_OBSERVATION_DIM,
            ],
        },
        "teacher_action": {
            "dtype": "tensor",
            "shape": [
                PARKOUR_ACTOR_OBSERVATION_HORIZON,
                PARKOUR_TEACHER_ACTION_DIM,
            ],
        },
        "oracle_yaw": {"dtype": "tensor", "shape": [2]},
        "termination": {
            "dtype": "mask",
            "shape": [PARKOUR_ACTOR_OBSERVATION_HORIZON],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "done": {"dtype": "bool", "shape": [1]},
    }
    info = {
        "codebase_version": "v2.0",
        "fps": _DATASET_FPS,
        "total_episodes": len(episode_lengths),
        "total_frames": int(sum(episode_lengths)),
        "total_tasks": 1,
        "total_videos": len(episode_lengths),
        "total_chunks": (len(episode_lengths) + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
        "chunks_size": _CHUNK_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": int(length)}) + "\n"
            for index, length in enumerate(episode_lengths)
        ),
        encoding="utf-8",
    )
    (dataset_dir / "meta/tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task": "Traverse the parkour course and follow each waypoint.",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _convert_split(
    raw_dir: Path,
    output_dir: Path,
    *,
    split: str,
    force: bool,
) -> dict[str, Any]:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    episode_lengths: list[int] = []
    image_shape: tuple[int, int, int] | None = None
    for episode_index, (arrays, source_video) in enumerate(
        _iter_parkour_dagger_shards(raw_dir, split)
    ):
        episode_dir = output_dir / f"data/chunk-{episode_index // _CHUNK_SIZE:03d}"
        parquet_path = episode_dir / f"episode_{episode_index:06d}.parquet"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(_episode_table(arrays, episode_index), parquet_path)
        video_path = output_dir / (
            f"videos/chunk-{episode_index // _CHUNK_SIZE:03d}/observation.image/"
            f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, video_path)
        image_shape = tuple(arrays["image_shape"].tolist())
        episode_lengths.append(arrays["state"].shape[0])
    if image_shape is None:
        raise ValueError(f"Parkour DAgger split {split!r} has no committed shards")
    _write_info(output_dir, episode_lengths=episode_lengths, image_shape=image_shape)
    _write_official_gr00t_modality(
        output_dir,
        PARKOUR_VLA_STATE_DIM,
        PARKOUR_VLA_ACTION_DIM,
    )
    return {
        "dataset_name": output_dir.name,
        "split": split,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
    }


def convert_parkour_dagger_shards_to_lerobot(
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    expected_rows: int | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Convert committed NPZ+MP4 DAgger shards to LeRobot datasets."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    train_manifest = _convert_split(raw_dir, output_dir, split="train", force=force)
    val_dir = output_dir.with_name(f"{output_dir.name}__val")
    val_manifest = None
    if _shard_dirs(raw_dir, "val"):
        val_manifest = _convert_split(raw_dir, val_dir, split="val", force=force)
    train_rows = train_manifest["frames"]
    if expected_rows is not None and train_rows != expected_rows:
        raise ValueError(
            f"Parkour DAgger export has {train_rows} train rows; "
            f"expected {expected_rows}"
        )
    manifest = {
        **train_manifest,
        "format": "parkour_dagger_lerobot_v2_video_numeric_parquet",
        "source": str(raw_dir),
        "action_layout": GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.layout,
        "action_labels": list(GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels),
        "action_dim": PARKOUR_VLA_ACTION_DIM,
        "state_dim": PARKOUR_VLA_STATE_DIM,
        "actor_observation_shape": [
            PARKOUR_ACTOR_OBSERVATION_HORIZON,
            PARKOUR_ACTOR_OBSERVATION_DIM,
        ],
        "teacher_action_shape": [
            PARKOUR_ACTOR_OBSERVATION_HORIZON,
            PARKOUR_TEACHER_ACTION_DIM,
        ],
        "termination_shape": [PARKOUR_ACTOR_OBSERVATION_HORIZON],
        "teacher_labels": train_rows * PARKOUR_DAGGER_LABELS_PER_ROW,
    }
    if val_manifest is not None:
        manifest.update(
            {
                "validation_dataset_name": val_manifest["dataset_name"],
                "validation_episodes": val_manifest["episodes"],
                "validation_frames": val_manifest["frames"],
            }
        )
    if round_index is not None:
        manifest["round_index"] = round_index
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_name("manifest.json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest


__all__ = [
    "convert_parkour_dagger_shards_to_lerobot",
    "write_parkour_dagger_shard",
]
