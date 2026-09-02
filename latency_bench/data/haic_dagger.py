"""Video-backed LeRobot export for the HAIC GR00T DAgger contract.

Collectors write one atomically committed directory per episode.  The numeric
payload is kept in ``episode.npz`` while the RGB stream stays in ``episode.mp4``;
the converter writes the numeric fields to LeRobot parquet and references the
video from the standard video directory.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


HAIC_VLA_STATE_DIM = 605
HAIC_PRIVILEGED_FEATURE_DIM = 256
HAIC_OBJECT_TARGET_DIM = 19
HAIC_VLA_ACTION_DIM = HAIC_PRIVILEGED_FEATURE_DIM + HAIC_OBJECT_TARGET_DIM
HAIC_ACTOR_OBSERVATION_HORIZON = 5
HAIC_ACTOR_INPUT_DIM = HAIC_VLA_STATE_DIM
HAIC_TEACHER_ACTION_DIM = 23
HAIC_DAGGER_LABELS_PER_ROW = HAIC_ACTOR_OBSERVATION_HORIZON

_SHARD_ROOT = "rollout_shards"
_SHARD_PREFIX = "episode_"
_DATASET_FPS = 10.0
_CHUNK_SIZE = 1000
_BOOTSTRAP_FIELDS = ("state", "action")
_DAGGER_FIELDS = (
    "state",
    "action",
    "actor_input",
    "teacher_action",
    "termination",
    "is_init",
)


def _haic_shard_dir(output_dir: Path | str, split: str, episode_idx: int) -> Path:
    return (
        Path(output_dir) / _SHARD_ROOT / split / f"{_SHARD_PREFIX}{episode_idx:06d}"
    )


def _write_shard(
    output_dir: Path | str,
    *,
    split: str,
    episode_idx: int,
    arrays: Mapping[str, Any],
    video_path: Path | str,
    fields: tuple[str, ...],
) -> Path:
    payload = {name: arrays[name] for name in fields}
    payload["image_shape"] = arrays["image_shape"]
    target = _haic_shard_dir(output_dir, split, episode_idx)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    np.savez_compressed(temporary / "episode.npz", **payload)
    shutil.copyfile(video_path, temporary / "episode.mp4")
    temporary.replace(target)
    return target


def write_haic_bootstrap_shard(
    output_dir: Path | str,
    *,
    split: str,
    episode_idx: int,
    arrays: Mapping[str, Any],
    video_path: Path | str,
) -> Path:
    """Atomically commit one HAIC privileged-oracle bootstrap episode."""
    return _write_shard(
        output_dir,
        split=split,
        episode_idx=episode_idx,
        arrays=arrays,
        video_path=video_path,
        fields=_BOOTSTRAP_FIELDS,
    )


def write_haic_dagger_shard(
    output_dir: Path | str,
    *,
    split: str,
    episode_idx: int,
    arrays: Mapping[str, Any],
    video_path: Path | str,
) -> Path:
    """Atomically commit one HAIC student-rollout DAgger episode."""
    return _write_shard(
        output_dir,
        split=split,
        episode_idx=episode_idx,
        arrays=arrays,
        video_path=video_path,
        fields=_DAGGER_FIELDS,
    )


def _shard_dirs(output_dir: Path | str, split: str) -> list[Path]:
    return sorted(
        path
        for path in Path(output_dir).glob(
            f"**/{_SHARD_ROOT}/{split}/{_SHARD_PREFIX}*"
        )
        if path.is_dir()
    )


def _iter_shards(
    output_dir: Path | str,
    split: str,
    *,
    fields: tuple[str, ...],
) -> Iterator[tuple[dict[str, Any], Path]]:
    for shard_dir in _shard_dirs(output_dir, split):
        with np.load(shard_dir / "episode.npz", allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in fields}
            arrays["image_shape"] = payload["image_shape"]
        yield arrays, shard_dir / "episode.mp4"


def _episode_table(
    arrays: Mapping[str, Any],
    episode_index: int,
    *,
    dagger: bool,
) -> Any:
    import pyarrow as pa

    row_count = arrays["state"].shape[0]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    columns: dict[str, Any] = {
        "observation.image": pa.array(
            [{"bytes": None, "path": None} for _ in range(row_count)],
            type=image_type,
        ),
        "observation.state": pa.array(
            arrays["state"].tolist(),
            type=pa.list_(pa.float32(), HAIC_VLA_STATE_DIM),
        ),
        "action": pa.array(
            arrays["action"].tolist(),
            type=pa.list_(pa.float32(), HAIC_VLA_ACTION_DIM),
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
    }
    if dagger:
        columns.update(
            {
                "actor_input": pa.array(
                    arrays["actor_input"].tolist(),
                    type=pa.list_(
                        pa.list_(pa.float32(), HAIC_ACTOR_INPUT_DIM),
                        HAIC_ACTOR_OBSERVATION_HORIZON,
                    ),
                ),
                "teacher_action": pa.array(
                    arrays["teacher_action"].tolist(),
                    type=pa.list_(
                        pa.list_(pa.float32(), HAIC_TEACHER_ACTION_DIM),
                        HAIC_ACTOR_OBSERVATION_HORIZON,
                    ),
                ),
                "is_init": pa.array(
                    arrays["is_init"].tolist(),
                    type=pa.list_(pa.bool_(), HAIC_ACTOR_OBSERVATION_HORIZON),
                ),
            }
        )
    columns["done"] = pa.array(
        arrays["termination"][:, -1] if dagger else np.zeros(row_count, dtype=bool),
        type=pa.bool_(),
    )
    return pa.table(columns)


def _write_info(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    image_shape: tuple[int, int, int],
    dagger: bool,
) -> None:
    height, width, channels = image_shape
    features: dict[str, Any] = {
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
            "shape": [HAIC_VLA_STATE_DIM],
            "names": [f"haic_state_{index}" for index in range(HAIC_VLA_STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": [HAIC_VLA_ACTION_DIM],
            "names": [
                *(
                    f"privileged_feature_{index}"
                    for index in range(HAIC_PRIVILEGED_FEATURE_DIM)
                ),
                *(f"object_target_{index}" for index in range(HAIC_OBJECT_TARGET_DIM)),
            ],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "done": {"dtype": "bool", "shape": [1]},
    }
    if dagger:
        features.update(
            {
                "actor_input": {
                    "dtype": "tensor",
                    "shape": [HAIC_ACTOR_OBSERVATION_HORIZON, HAIC_ACTOR_INPUT_DIM],
                },
                "teacher_action": {
                    "dtype": "tensor",
                    "shape": [
                        HAIC_ACTOR_OBSERVATION_HORIZON,
                        HAIC_TEACHER_ACTION_DIM,
                    ],
                },
                "is_init": {
                    "dtype": "mask",
                    "shape": [HAIC_ACTOR_OBSERVATION_HORIZON],
                },
            }
        )
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
                "task": "Pull the cart along the reference motion.",
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
    dagger: bool,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    fields = _DAGGER_FIELDS if dagger else _BOOTSTRAP_FIELDS
    episode_lengths: list[int] = []
    image_shape: tuple[int, int, int] | None = None
    for episode_index, (arrays, source_video) in enumerate(
        _iter_shards(raw_dir, split, fields=fields)
    ):
        episode_dir = output_dir / f"data/chunk-{episode_index // _CHUNK_SIZE:03d}"
        parquet_path = episode_dir / f"episode_{episode_index:06d}.parquet"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _episode_table(arrays, episode_index, dagger=dagger), parquet_path
        )
        video_path = output_dir / (
            f"videos/chunk-{episode_index // _CHUNK_SIZE:03d}/observation.image/"
            f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, video_path)
        image_shape = tuple(arrays["image_shape"].tolist())
        episode_lengths.append(arrays["state"].shape[0])
    if image_shape is None:
        raise ValueError(f"HAIC {split!r} split has no committed shards")
    _write_info(
        output_dir,
        episode_lengths=episode_lengths,
        image_shape=image_shape,
        dagger=dagger,
    )
    from latency_bench.data.official_gr00t import _write_official_gr00t_modality

    _write_official_gr00t_modality(
        output_dir,
        HAIC_VLA_STATE_DIM,
        HAIC_VLA_ACTION_DIM,
    )
    return {
        "dataset_name": output_dir.name,
        "split": split,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
    }


def _convert_haic_shards_to_lerobot(
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    dagger: bool,
    force: bool = False,
    expected_rows: int | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Convert HAIC bootstrap or DAgger NPZ+MP4 shards to LeRobot."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    train_manifest = _convert_split(
        raw_dir, output_dir, split="train", force=force, dagger=dagger
    )
    val_dir = output_dir.with_name(f"{output_dir.name}__val")
    val_manifest = None
    if _shard_dirs(raw_dir, "val"):
        val_manifest = _convert_split(
            raw_dir, val_dir, split="val", force=force, dagger=dagger
        )
    train_rows = train_manifest["frames"]
    if expected_rows is not None and train_rows != expected_rows:
        raise ValueError(
            f"HAIC export has {train_rows} train rows; expected {expected_rows}"
        )
    manifest: dict[str, Any] = {
        **train_manifest,
        "format": (
            f"haic_{'dagger' if dagger else 'bootstrap'}_lerobot_v1_"
            "video_numeric_parquet"
        ),
        "source": str(raw_dir),
        "action_dim": HAIC_VLA_ACTION_DIM,
        "state_dim": HAIC_VLA_STATE_DIM,
        "privileged_feature_dim": HAIC_PRIVILEGED_FEATURE_DIM,
        "object_target_dim": HAIC_OBJECT_TARGET_DIM,
        "dagger": dagger,
    }
    if dagger:
        manifest.update(
            {
                "actor_input_shape": [
                    HAIC_ACTOR_OBSERVATION_HORIZON,
                    HAIC_ACTOR_INPUT_DIM,
                ],
                "teacher_action_shape": [
                    HAIC_ACTOR_OBSERVATION_HORIZON,
                    HAIC_TEACHER_ACTION_DIM,
                ],
                "is_init_shape": [HAIC_ACTOR_OBSERVATION_HORIZON],
                "teacher_labels": train_rows * HAIC_DAGGER_LABELS_PER_ROW,
            }
        )
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


def convert_haic_dagger_shards_to_lerobot(
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    expected_rows: int | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Convert HAIC DAgger shards to a LeRobot dataset."""
    return _convert_haic_shards_to_lerobot(
        raw_dir,
        output_dir,
        dagger=True,
        force=force,
        expected_rows=expected_rows,
        round_index=round_index,
    )


def convert_haic_bootstrap_shards_to_lerobot(
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    expected_rows: int | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Convert HAIC privileged-oracle bootstrap shards to LeRobot."""
    return _convert_haic_shards_to_lerobot(
        raw_dir,
        output_dir,
        dagger=False,
        force=force,
        expected_rows=expected_rows,
        round_index=round_index,
    )


__all__ = [
    "HAIC_ACTOR_INPUT_DIM",
    "HAIC_ACTOR_OBSERVATION_HORIZON",
    "HAIC_DAGGER_LABELS_PER_ROW",
    "HAIC_OBJECT_TARGET_DIM",
    "HAIC_PRIVILEGED_FEATURE_DIM",
    "HAIC_TEACHER_ACTION_DIM",
    "HAIC_VLA_ACTION_DIM",
    "HAIC_VLA_STATE_DIM",
    "convert_haic_bootstrap_shards_to_lerobot",
    "convert_haic_dagger_shards_to_lerobot",
    "write_haic_bootstrap_shard",
    "write_haic_dagger_shard",
]
