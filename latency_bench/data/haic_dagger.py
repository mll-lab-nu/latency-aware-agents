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

from latency_bench.data.episode_io import commit_episode


HAIC_VLA_STATE_DIM = 605
HAIC_TEACHER_LATENT_DIM = 256
HAIC_VLA_ACTION_DIM = HAIC_TEACHER_LATENT_DIM
HAIC_ACTION_HORIZON = 40
HAIC_TEACHER_ACTION_DIM = 23
HAIC_ACTOR_INPUT_DIM = HAIC_VLA_STATE_DIM

_SHARD_ROOT = "rollout_shards"
_SHARD_PREFIX = "episode_"
_DATASET_FPS = 50.0
_CHUNK_SIZE = 1000
_NUMERIC_FIELDS = (
    "state",
    "action",
    "actor_input",
    "teacher_action",
    "termination",
)
_BOOTSTRAP_FIELDS = _NUMERIC_FIELDS
_DAGGER_FIELDS = _NUMERIC_FIELDS
_ISSUED_FIELDS = (
    "episode_id",
    "obs_id",
    "issued_raw_frame",
    "ready_raw_frame",
    "latency_ms",
    "worker_id",
)
_CONTROL_FIELDS = (
    "control_applied_command",
    "control_source_raw_frame",
    "control_source_obs_id",
    "control_chunk_index",
    "control_reward",
    "control_done",
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
    if np.asarray(arrays["action"]).ndim == 3:
        payload.update({name: arrays[name] for name in _ISSUED_FIELDS})
        payload.update({name: arrays[name] for name in _CONTROL_FIELDS})
    payload["image_shape"] = arrays["image_shape"]
    target = _haic_shard_dir(output_dir, split, episode_idx)
    return commit_episode(target, payload, video_path)


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
            if np.asarray(arrays["action"]).ndim == 3:
                arrays.update({name: payload[name] for name in _ISSUED_FIELDS})
                arrays.update({name: payload[name] for name in _CONTROL_FIELDS})
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
    action = np.asarray(arrays["action"])
    actor_input = np.asarray(arrays["actor_input"])
    teacher_action = np.asarray(arrays["teacher_action"])
    termination = np.asarray(arrays["termination"])
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    issued_chunk = action.ndim == 3
    action_key = "issued_command_chunk" if issued_chunk else "action"
    columns: dict[str, Any] = {
        "observation.image": pa.array(
            [{"bytes": None, "path": None} for _ in range(row_count)],
            type=image_type,
        ),
        "observation.state": pa.array(
            arrays["state"].tolist(),
            type=pa.list_(pa.float32(), HAIC_VLA_STATE_DIM),
        ),
        action_key: pa.array(
            action.tolist(),
            type=(
                pa.list_(
                    pa.list_(pa.float32(), HAIC_VLA_ACTION_DIM),
                    action.shape[1],
                )
                if issued_chunk
                else pa.list_(pa.float32(), HAIC_VLA_ACTION_DIM)
            ),
        ),
        "haic_actor_input": pa.array(
            actor_input.tolist(),
            type=(
                pa.list_(
                    pa.list_(pa.float32(), HAIC_ACTOR_INPUT_DIM),
                    actor_input.shape[1],
                )
                if actor_input.ndim == 3
                else pa.list_(pa.float32(), HAIC_ACTOR_INPUT_DIM)
            ),
        ),
        "haic_teacher_action": pa.array(
            teacher_action.tolist(),
            type=(
                pa.list_(
                    pa.list_(pa.float32(), HAIC_TEACHER_ACTION_DIM),
                    teacher_action.shape[1],
                )
                if teacher_action.ndim == 3
                else pa.list_(pa.float32(), HAIC_TEACHER_ACTION_DIM)
            ),
        ),
        "haic_termination": pa.array(
            termination.tolist(),
            type=(
                pa.list_(pa.bool_(), termination.shape[1])
                if termination.ndim == 2
                else pa.bool_()
            ),
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
    columns["done"] = pa.array(
        np.any(termination, axis=1) if termination.ndim == 2 else termination,
        type=pa.bool_(),
    )
    if issued_chunk:
        columns.update({name: pa.array(arrays[name]) for name in _ISSUED_FIELDS})
        columns["latency"] = pa.array(arrays["latency_ms"], type=pa.float64())
        columns["latency_raw_frames"] = pa.array(
            arrays["ready_raw_frame"] - arrays["issued_raw_frame"],
            type=pa.int64(),
        )
    return pa.table(columns)


def _write_info(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    image_shape: tuple[int, int, int],
    action_shape: tuple[int, ...],
    actor_input_shape: tuple[int, ...],
    teacher_action_shape: tuple[int, ...],
    termination_shape: tuple[int, ...],
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
        ("issued_command_chunk" if len(action_shape) == 2 else "action"): {
            "dtype": "float32",
            "shape": list(action_shape),
            "names": [
                *(f"teacher_latent_{index}" for index in range(HAIC_TEACHER_LATENT_DIM)),
            ],
        },
        "haic_actor_input": {
            "dtype": "float32",
            "shape": list(actor_input_shape),
            "names": [
                f"actor_input_{index}" for index in range(HAIC_ACTOR_INPUT_DIM)
            ],
        },
        "haic_teacher_action": {
            "dtype": "float32",
            "shape": list(teacher_action_shape),
            "names": [
                f"teacher_action_{index}"
                for index in range(HAIC_TEACHER_ACTION_DIM)
            ],
        },
        "haic_termination": {
            "dtype": "mask",
            "shape": list(termination_shape) if termination_shape else [1],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "done": {"dtype": "bool", "shape": [1]},
    }
    if len(action_shape) == 2:
        features["latency"] = {"dtype": "float64", "shape": [1]}
        features.update(
            {
                name: {
                    "dtype": "float64" if name == "latency_ms" else "int64",
                    "shape": [1],
                }
                for name in _ISSUED_FIELDS
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
    termination_counts = None
    action_shape: tuple[int, ...] | None = None
    actor_input_shape: tuple[int, ...] | None = None
    teacher_action_shape: tuple[int, ...] | None = None
    termination_shape: tuple[int, ...] | None = None
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
        action_shape = tuple(np.asarray(arrays["action"]).shape[1:])
        actor_input_shape = tuple(np.asarray(arrays["actor_input"]).shape[1:])
        teacher_action_shape = tuple(
            np.asarray(arrays["teacher_action"]).shape[1:]
        )
        termination_shape = tuple(np.asarray(arrays["termination"]).shape[1:])
        episode_lengths.append(arrays["state"].shape[0])
        counts = arrays["termination"].sum()
        termination_counts = (
            counts
            if termination_counts is None
            else termination_counts + counts
        )
    if image_shape is None:
        raise ValueError(f"HAIC {split!r} split has no committed shards")
    _write_info(
        output_dir,
        episode_lengths=episode_lengths,
        image_shape=image_shape,
        action_shape=action_shape,
        actor_input_shape=actor_input_shape,
        teacher_action_shape=teacher_action_shape,
        termination_shape=termination_shape,
    )
    from latency_bench.data.exporters.starvla_lerobot import write_starvla_modality

    write_starvla_modality(
        output_dir,
        HAIC_VLA_STATE_DIM,
        HAIC_VLA_ACTION_DIM,
        action_key=(
            "issued_command_chunk" if len(action_shape) == 2 else "action"
        ),
    )
    manifest = {
        "dataset_name": output_dir.name,
        "split": split,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "action_shape": list(action_shape),
        "actor_input_shape": list(actor_input_shape),
        "teacher_action_shape": list(teacher_action_shape),
        "termination_shape": list(termination_shape),
    }
    if termination_counts is not None:
        manifest["termination_count"] = int(termination_counts)
    return manifest


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
            f"haic_{'dagger' if dagger else 'bootstrap'}_lerobot_v3_"
            "video_numeric_parquet"
        ),
        "source": str(raw_dir),
        "action_dim": HAIC_VLA_ACTION_DIM,
        "state_dim": HAIC_VLA_STATE_DIM,
        "teacher_latent_dim": HAIC_TEACHER_LATENT_DIM,
        "teacher_action_dim": HAIC_TEACHER_ACTION_DIM,
        "action_horizon": HAIC_ACTION_HORIZON,
        "actor_input_shape": train_manifest["actor_input_shape"],
        "teacher_action_shape": train_manifest["teacher_action_shape"],
        "termination_shape": train_manifest["termination_shape"],
        "action_shape": train_manifest["action_shape"],
        "dagger": dagger,
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
    "HAIC_TEACHER_LATENT_DIM",
    "HAIC_TEACHER_ACTION_DIM",
    "HAIC_ACTOR_INPUT_DIM",
    "HAIC_ACTION_HORIZON",
    "HAIC_VLA_ACTION_DIM",
    "HAIC_VLA_STATE_DIM",
    "convert_haic_bootstrap_shards_to_lerobot",
    "convert_haic_dagger_shards_to_lerobot",
    "write_haic_bootstrap_shard",
    "write_haic_dagger_shard",
]
