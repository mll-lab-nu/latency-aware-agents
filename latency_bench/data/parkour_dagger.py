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

from latency_bench.data.action_layouts import (
    GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC,
)
PARKOUR_VLA_STATE_DIM = 53
PARKOUR_VLA_ACTION_DIM = 34
PARKOUR_CONTROL_REPEAT = 5
PARKOUR_ACTION_HORIZON = 40
PARKOUR_ACTOR_OBSERVATION_DIM = 753

_SHARD_ROOT = "rollout_shards"
_SHARD_PREFIX = "episode_"
_DATASET_FPS = 10.0
_CHUNK_SIZE = 1000
_NUMERIC_FIELDS = (
    "state",
    "action",
    "termination",
    "actor_observation",
)
_ISSUED_FIELDS = (
    "episode_id", "obs_id", "issued_raw_frame", "ready_raw_frame", "latency_ms", "worker_id",
)
_CONTROL_FIELDS = (
    "control_applied_command", "control_source_raw_frame", "control_source_obs_id",
    "control_chunk_index", "control_reward", "control_done",
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

    Issued H40 episodes retain admission metadata and the complete raw control
    trace. Control arrays have one row per raw frame, not per admitted anchor;
    source/index -1 denotes a noop. Only admitted anchors enter LeRobot.
    """
    payload = {name: arrays[name] for name in _NUMERIC_FIELDS}
    if arrays["action"].ndim == 3:
        payload.update({name: arrays[name] for name in _ISSUED_FIELDS})
        payload.update({name: arrays[name] for name in _CONTROL_FIELDS})
    if "raw_reward" in arrays:
        payload["raw_reward"] = arrays["raw_reward"]
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
        for path in (Path(output_dir) / _SHARD_ROOT / split).glob(
            f"{_SHARD_PREFIX}*"
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
            if arrays["action"].ndim == 3:
                arrays.update({name: payload[name] for name in _ISSUED_FIELDS})
            if "raw_reward" in payload.files:
                arrays["raw_reward"] = payload["raw_reward"]
            arrays["image_shape"] = payload["image_shape"]
        yield arrays, shard_dir / "episode.mp4"


def _episode_table(
    arrays: Mapping[str, Any], episode_index: int, *, include_bootstrap_fields: bool
):
    # Isaac Lab's collector environment does not ship pyarrow; keep this
    # import at the conversion boundary so collection can run without it.
    import pyarrow as pa

    row_count = arrays["state"].shape[0]
    action = np.asarray(arrays["action"])
    actor_observation = np.asarray(arrays["actor_observation"])
    termination = np.asarray(arrays["termination"])
    issued_chunk = action.ndim == 3
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    action_key = "issued_command_chunk" if issued_chunk else "action"
    columns = {
        "observation.image": pa.array(
            [{"bytes": None, "path": None} for _ in range(row_count)],
            type=image_type,
        ),
        "observation.state": pa.array(
            arrays["state"].tolist(),
            type=pa.list_(pa.float32(), PARKOUR_VLA_STATE_DIM),
        ),
        action_key: pa.array(
            action.tolist(),
            type=(
                pa.list_(
                    pa.list_(pa.float32(), PARKOUR_VLA_ACTION_DIM),
                    action.shape[1],
                )
                if issued_chunk
                else pa.list_(pa.float32(), PARKOUR_VLA_ACTION_DIM)
            ),
        ),
        "termination": pa.array(
            termination.tolist(),
            type=pa.list_(pa.bool_(), termination.shape[1]),
        ),
        "actor_observation": pa.array(
            actor_observation.tolist(),
            type=pa.list_(
                pa.list_(pa.float32(), PARKOUR_ACTOR_OBSERVATION_DIM),
                actor_observation.shape[1],
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
    if issued_chunk:
        columns.update({name: pa.array(arrays[name]) for name in _ISSUED_FIELDS})
    if include_bootstrap_fields:
        done = np.zeros(row_count, dtype=bool)
        done[-1] = True
        columns.update(
            {
                "latency": pa.array(
                    arrays["latency_ms"] if issued_chunk else np.zeros(row_count, dtype=np.int64)
                ),
                "latency_raw_frames": pa.array(
                    arrays["ready_raw_frame"] - arrays["issued_raw_frame"]
                    if issued_chunk else np.zeros(row_count, dtype=np.int64),
                    type=pa.int64(),
                ),
                "done": pa.array(done, type=pa.bool_()),
                "reward": pa.array(arrays["raw_reward"], type=pa.float32()),
                "density_weight": pa.array(
                    np.ones(row_count, dtype=np.float32), type=pa.float32()
                ),
            }
        )
    return pa.table(columns)


def _write_info(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    image_shape: tuple[int, int, int],
    include_bootstrap_fields: bool,
    action_shape: tuple[int, ...],
    actor_observation_shape: tuple[int, ...],
    termination_shape: tuple[int, ...],
) -> None:
    height, width, channels = image_shape
    action_key = "issued_command_chunk" if len(action_shape) == 2 else "action"
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
        action_key: {
            "dtype": "float32",
            "shape": list(action_shape),
            "names": list(GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels),
        },
        "termination": {
            "dtype": "mask",
            "shape": list(termination_shape),
        },
        "actor_observation": {
            "dtype": "float32",
            "shape": list(actor_observation_shape),
            "names": [
                [
                    f"actor_observation_{index}"
                    for index in range(PARKOUR_ACTOR_OBSERVATION_DIM)
                ]
                for _ in range(actor_observation_shape[0])
            ],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    if include_bootstrap_fields:
        features["latency"] = {
            "dtype": "float64" if len(action_shape) == 2 else "int64", "shape": [1]
        }
    if len(action_shape) == 2:
        features.update({name: {
            "dtype": "float64" if name == "latency_ms" else "int64", "shape": [1],
        } for name in _ISSUED_FIELDS})
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
    include_bootstrap_fields: bool,
) -> dict[str, Any]:
    # Isaac Lab's collector environment does not ship pyarrow; keep this
    # import at the conversion boundary so collection can run without it.
    import pyarrow.parquet as pq
    from latency_bench.data.official_gr00t import _write_official_gr00t_modality

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    episode_lengths: list[int] = []
    image_shape: tuple[int, int, int] | None = None
    action_shape: tuple[int, ...] | None = None
    actor_observation_shape: tuple[int, ...] | None = None
    termination_shape: tuple[int, ...] | None = None
    for episode_index, (arrays, source_video) in enumerate(
        _iter_parkour_dagger_shards(raw_dir, split)
    ):
        episode_dir = output_dir / f"data/chunk-{episode_index // _CHUNK_SIZE:03d}"
        parquet_path = episode_dir / f"episode_{episode_index:06d}.parquet"
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _episode_table(
                arrays,
                episode_index,
                include_bootstrap_fields=include_bootstrap_fields,
            ),
            parquet_path,
        )
        video_path = output_dir / (
            f"videos/chunk-{episode_index // _CHUNK_SIZE:03d}/observation.image/"
            f"episode_{episode_index:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_video, video_path)
        image_shape = tuple(arrays["image_shape"].tolist())
        action_shape = tuple(np.asarray(arrays["action"]).shape[1:])
        actor_observation_shape = tuple(
            np.asarray(arrays["actor_observation"]).shape[1:]
        )
        termination_shape = tuple(np.asarray(arrays["termination"]).shape[1:])
        episode_lengths.append(arrays["state"].shape[0])
    if image_shape is None:
        raise ValueError(f"Parkour DAgger split {split!r} has no committed shards")
    _write_info(
        output_dir,
        episode_lengths=episode_lengths,
        image_shape=image_shape,
        include_bootstrap_fields=include_bootstrap_fields,
        action_shape=action_shape,
        actor_observation_shape=actor_observation_shape,
        termination_shape=termination_shape,
    )
    _write_official_gr00t_modality(
        output_dir,
        PARKOUR_VLA_STATE_DIM,
        PARKOUR_VLA_ACTION_DIM,
        action_key=("issued_command_chunk" if len(action_shape) == 2 else "action"),
    )
    return {
        "dataset_name": output_dir.name,
        "split": split,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "action_shape": list(action_shape),
        "termination_shape": list(termination_shape),
        "actor_observation_shape": list(actor_observation_shape),
    }


def convert_parkour_dagger_shards_to_lerobot(
    raw_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    expected_rows: int | None = None,
    round_index: int | None = None,
    video_shards: bool = False,
) -> dict[str, Any]:
    """Convert committed NPZ+MP4 DAgger shards to LeRobot datasets."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    train_manifest = _convert_split(
        raw_dir,
        output_dir,
        split="train",
        force=force,
        include_bootstrap_fields=video_shards,
    )
    val_dir = output_dir.with_name(f"{output_dir.name}__val")
    val_manifest = None
    if _shard_dirs(raw_dir, "val"):
        val_manifest = _convert_split(
            raw_dir,
            val_dir,
            split="val",
            force=force,
            include_bootstrap_fields=video_shards,
        )
    train_rows = train_manifest["frames"]
    if expected_rows is not None and train_rows != expected_rows:
        raise ValueError(
            f"Parkour DAgger export has {train_rows} train rows; "
            f"expected {expected_rows}"
        )
    manifest = {
        **train_manifest,
        "format": (
            "parkour_video_bootstrap_lerobot_v1"
            if video_shards
            else "parkour_dagger_lerobot_v4_video_actor_observation_parquet"
        ),
        "source": str(raw_dir),
        "action_layout": GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.layout,
        "action_labels": list(GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels),
        "action_dim": PARKOUR_VLA_ACTION_DIM,
        "state_dim": PARKOUR_VLA_STATE_DIM,
        "official_action_horizon": PARKOUR_ACTION_HORIZON,
        "termination_shape": train_manifest["termination_shape"],
        "actor_observation_shape": train_manifest["actor_observation_shape"],
        "action_shape": train_manifest["action_shape"],
    }
    bootstrap_contract = None
    if video_shards:
        state_labels = [
            f"parkour_proprio_{index}" for index in range(PARKOUR_VLA_STATE_DIM)
        ]
        bootstrap_contract = {
            "env_name": "extreme_parkour_go2",
            "task_prompts": [
                "Traverse the parkour course and follow each waypoint."
            ],
            "integration_name": "isaaclab_parkour",
            "task_name": "extreme_parkour_go2",
            "carrier_action_labels": list(
                GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels
            ),
            "active_action_dim": PARKOUR_VLA_ACTION_DIM,
            "action_carrier": "native",
            "rows_unit": "decision_step",
            "obs_fps": int(_DATASET_FPS),
            "obs_stride_raw_frames": PARKOUR_CONTROL_REPEAT,
            "uses_state": True,
            "state_labels": state_labels,
            "state_normalization": None,
            "official_embodiment_tag": "NEW_EMBODIMENT",
            "official_action_horizon": PARKOUR_ACTION_HORIZON,
            "official_modality_config": (
                "latency_bench/policy/official_gr00t_parkour_modality.py"
            ),
        }
        manifest.update(bootstrap_contract)
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
    if val_manifest is not None and bootstrap_contract is not None:
        validation_manifest = {
            **val_manifest,
            "format": "parkour_video_bootstrap_lerobot_v1",
            "source": str(raw_dir),
            "action_layout": GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.layout,
            "action_labels": list(
                GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC.labels
            ),
            "action_dim": PARKOUR_VLA_ACTION_DIM,
            "state_dim": PARKOUR_VLA_STATE_DIM,
            "termination_shape": val_manifest["termination_shape"],
            "actor_observation_shape": val_manifest["actor_observation_shape"],
            "action_shape": val_manifest["action_shape"],
            **bootstrap_contract,
            "training_dataset_name": train_manifest["dataset_name"],
        }
        validation_manifest_path = val_dir / "manifest.json"
        temporary = validation_manifest_path.with_name("manifest.json.tmp")
        temporary.write_text(
            json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(validation_manifest_path)
    return manifest


__all__ = [
    "convert_parkour_dagger_shards_to_lerobot",
    "write_parkour_dagger_shard",
    "PARKOUR_ACTION_HORIZON",
    "PARKOUR_ACTOR_OBSERVATION_DIM",
]
