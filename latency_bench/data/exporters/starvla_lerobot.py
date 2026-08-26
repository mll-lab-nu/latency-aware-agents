"""Convert canonical rollout exports to the StarVLA LeRobot image format."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from latency_bench.utils.io import write_jsonl

from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ACTION_LAYOUT_GYMNASIUM_DISCRETE_V1,
    action_spec_for_env,
)
from latency_bench.data.filters import density_weights_clean_v1

STATE_DIM = 1
BRIDGE_ACTION_DIM = 7
LE_ROBOT_CHUNK_SIZE = 1000
_ROBOT_TYPES_BY_ENV = {
    "flappy": "rl_games_flappy",
    "demon_attack": "rl_games_demon_attack",
    "deadly_corridor": "rl_games_deadly_corridor",
}


def _action_carrier_spec(
    action_labels: list[str],
    action_carrier: str,
) -> tuple[list[str], int, int, list[str]]:
    if action_carrier == "native":
        return action_labels, len(action_labels), STATE_DIM, ["state"]
    if action_carrier != "bridge":
        raise ValueError(f"Unsupported action_carrier={action_carrier!r}")
    if len(action_labels) > BRIDGE_ACTION_DIM:
        raise ValueError(
            f"bridge action carrier supports at most {BRIDGE_ACTION_DIM} actions, "
            f"got {len(action_labels)}"
        )
    carrier_labels = [
        *action_labels,
        *(
            f"BRIDGE_PAD_{index}"
            for index in range(len(action_labels), BRIDGE_ACTION_DIM)
        ),
    ]
    state_labels = [f"BRIDGE_STATE_{index}" for index in range(BRIDGE_ACTION_DIM)]
    return carrier_labels, len(action_labels), BRIDGE_ACTION_DIM, state_labels


def _robot_type_for_env(env_name: str, integration_name: str) -> str:
    if integration_name == "gymnasium":
        return "rl_games_gymnasium"
    return _ROBOT_TYPES_BY_ENV[env_name]


def _write_starvla_handoff_files(
    output_dir: Path,
    val_output_dir: Path,
    *,
    env_name: str,
    integration_name: str,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    prompt_map = {
        str(int(row["latency_raw_frames"])): {
            "prompt": row["prompt"],
            "latency_raw_frames": int(row["latency_raw_frames"]),
            "latency_ms": float(row["latency_ms"]),
        }
        for row in rows
    }
    prompt_map_path = output_dir / "latency_prompt_map.json"
    prompt_map_path.write_text(json.dumps(prompt_map, indent=2), encoding="utf-8")

    mixture_path = output_dir.parent / "_generated_mixtures" / f"{output_dir.name}.json"
    mixture_path.parent.mkdir(parents=True, exist_ok=True)
    robot_type = _robot_type_for_env(env_name, integration_name)
    mixture_path.write_text(
        json.dumps(
            {
                output_dir.name: [[output_dir.name, 1.0, robot_type]],
                val_output_dir.name: [[val_output_dir.name, 1.0, robot_type]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return prompt_map_path, mixture_path


def _png_bytes(image: Any) -> bytes:
    """Serialize a rollout image value to PNG bytes.

    Args:
        image: Hugging Face image struct, PIL image, or array-like image.

    Returns:
        RGB PNG bytes.
    """
    if isinstance(image, Mapping):
        if image["bytes"] is not None:
            return bytes(image["bytes"])
        with Path(image["path"]).open("rb") as handle:
            return handle.read()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_rows(raw_export_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load canonical rollout rows from one parquet split.

    Args:
        raw_export_dir: Directory containing ``train.parquet`` and ``val.parquet``.
        split: Split name to read.

    Returns:
        Rows as Python dictionaries.
    """
    path = raw_export_dir / f"{split}.parquet"
    table = pq.read_table(path)
    return table.to_pylist()


def _episode_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group canonical rows by episode and order by decision step.

    Args:
        rows: Canonical rollout rows from one split.

    Returns:
        Episode row groups sorted by episode index.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_idx"]), []).append(row)
    return [
        sorted(episode_rows, key=lambda row: int(row["decision_step"]))
        for _, episode_rows in sorted(grouped.items())
    ]


def _legacy_row_action_vector(row: dict[str, Any], *, action_dim: int) -> list[float]:
    if "action" in row and row["action"] is not None:
        return [float(value) for value in row["action"]]
    if "action_id" not in row:
        raise KeyError(f"row is missing action fields; available columns: {sorted(row)}")
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= action_dim:
        raise ValueError(
            f"action_id={action_id} is outside action range [0, {action_dim - 1}]"
        )
    return [1.0 if index == action_id else 0.0 for index in range(action_dim)]


def _write_legacy_episode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    action_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.image": pa.array(
                [{"bytes": row["image_bytes"], "path": None} for row in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "observation.state": pa.array(
                [[0.0] * STATE_DIM for _ in rows],
                type=pa.list_(pa.float32(), STATE_DIM),
            ),
            "action": pa.array(
                [row["action"] for row in rows],
                type=pa.list_(pa.float32(), action_dim),
            ),
            "timestamp": pa.array(
                [row["timestamp"] for row in rows], type=pa.float64()
            ),
            "episode_index": pa.array(
                [row["episode_index"] for row in rows], type=pa.int64()
            ),
            "frame_index": pa.array(
                [row["frame_index"] for row in rows], type=pa.int64()
            ),
            "task_index": pa.array(
                [row["task_index"] for row in rows], type=pa.int64()
            ),
            "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
            "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
            "density_weight": pa.array(
                [row.get("density_weight", 1.0) for row in rows],
                type=pa.float32(),
            ),
        }
    )
    pq.write_table(table, path)


def _write_legacy_metadata(
    dataset_dir: Path,
    *,
    fps: float,
    action_labels: list[str],
    episode_lengths: list[int],
    task_prompts: list[str],
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    action_dim = len(action_labels)
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "fps": fps,
                "features": {
                    "observation.image": {"dtype": "image"},
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [STATE_DIM],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [action_dim],
                        "names": action_labels,
                    },
                    "timestamp": {"dtype": "float64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": index, "length": int(length)}
            for index, length in enumerate(episode_lengths)
        ],
    )
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": index, "task": prompt}
            for index, prompt in enumerate(task_prompts)
        ],
    )


def _convert_legacy_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    env_name: str,
    action_labels: list[str],
    fps: float,
    reward_field: str,
    density_weighting: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_to_task_index: dict[str, int] = {}
    task_prompts: list[str] = []
    episode_lengths: list[int] = []
    for new_episode_idx, episode_rows in enumerate(_episode_groups(rows)):
        density_weights = (
            density_weights_clean_v1(env_name, episode_rows)
            if density_weighting
            else [1.0] * len(episode_rows)
        )
        out_rows = []
        for frame_idx, row in enumerate(episode_rows):
            row["latency_raw_frames"]
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)
            out_rows.append(
                {
                    "image_bytes": _png_bytes(row["image"]),
                    "action": _legacy_row_action_vector(
                        row,
                        action_dim=len(action_labels),
                    ),
                    "timestamp": frame_idx / fps,
                    "episode_index": new_episode_idx,
                    "frame_index": frame_idx,
                    "task_index": prompt_to_task_index[prompt],
                    "done": frame_idx == len(episode_rows) - 1,
                    "reward": float(row[reward_field]),
                    "density_weight": float(density_weights[frame_idx]),
                }
            )
        episode_lengths.append(len(out_rows))
        episode_chunk = new_episode_idx // 1000
        _write_legacy_episode(
            output_dir
            / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
            out_rows,
            action_dim=len(action_labels),
        )
    _write_legacy_metadata(
        output_dir,
        fps=fps,
        action_labels=action_labels,
        episode_lengths=episode_lengths,
        task_prompts=task_prompts,
    )
    return {
        "dataset_name": output_dir.name,
        "env_name": env_name,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "task_prompts": task_prompts,
    }


def _convert_legacy_export(
    raw_export_dir: Path,
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    env_name = str(metadata["env_name"])
    action_spec = action_spec_for_env(env_name)
    action_labels = list(metadata["action_labels"])
    reward_field = str(metadata["reward_field"])
    rows_unit = metadata["rows_unit"]
    if rows_unit != "decision_step":
        raise ValueError(f"rows_unit must be 'decision_step', got {rows_unit!r}")
    fps = metadata["obs_fps"]
    obs_stride = metadata["obs_stride_raw_frames"]
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)
    train_manifest = _convert_legacy_split(
        _load_rows(raw_export_dir, "train"),
        output_dir,
        env_name=env_name,
        action_labels=action_labels,
        fps=fps,
        reward_field=reward_field,
        density_weighting=True,
    )
    val_manifest = _convert_legacy_split(
        _load_rows(raw_export_dir, "val"),
        val_output_dir,
        env_name=env_name,
        action_labels=action_labels,
        fps=fps,
        reward_field=reward_field,
        density_weighting=False,
    )
    manifest = {
        **train_manifest,
        "format": "starvla_lerobot_v2_image_parquet",
        "source": str(raw_export_dir),
        "action_layout": str(metadata["action_layout"]),
        "action_labels": action_labels,
        "action_dim": len(action_labels),
        "active_action_dim": len(action_spec.labels),
        "rows_unit": rows_unit,
        "obs_fps": fps,
        "obs_stride_raw_frames": obs_stride,
        "validation_dataset_name": val_output_dir.name,
        "validation_episodes": val_manifest["episodes"],
        "validation_frames": val_manifest["frames"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _write_episode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    action_dim: int,
    state_dim: int,
) -> None:
    """Write one LeRobot episode parquet file.

    Args:
        path: Destination episode parquet path.
        rows: Converted LeRobot frame rows.
        action_dim: Expected action vector dimension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.image": pa.array(
            [{"bytes": row["image_bytes"], "path": None} for row in rows],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "observation.state": pa.array(
            [row.get("state", [0.0] * state_dim) for row in rows],
            type=pa.list_(pa.float32(), state_dim),
        ),
        "action": pa.array(
            [row["action"] for row in rows],
            type=pa.list_(pa.float32(), action_dim),
        ),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
        "episode_index": pa.array(
            [row["episode_index"] for row in rows], type=pa.int64()
        ),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        "latency": pa.array(
            [row["latency_raw_frames"] for row in rows], type=pa.int64()
        ),
        "latency_raw_frames": pa.array(
            [row["latency_raw_frames"] for row in rows], type=pa.int64()
        ),
        "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
        "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
        # Per-row loss weight for KV-memory scheme-B supervision (clean_v1 event
        # balancing without deleting rows). Uniform 1.0 when density is disabled.
        "density_weight": pa.array(
            [row["density_weight"] for row in rows], type=pa.float32()
        ),
    }
    if rows and "action_id" in rows[0]:
        columns["action_id"] = pa.array(
            [row["action_id"] for row in rows], type=pa.int64()
        )
    if rows and "context_image_bytes" in rows[0]:
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        columns["observation.context_images"] = pa.array(
            [
                [{"bytes": value, "path": None} for value in row["context_image_bytes"]]
                for row in rows
            ],
            type=pa.list_(image_type),
        )
    table = pa.table(columns)
    pq.write_table(table, path)


def _row_action_vector(
    row: dict[str, Any],
    *,
    active_action_dim: int,
    action_dim: int,
) -> list[float]:
    if "action" in row and row["action"] is not None:
        values = [float(value) for value in row["action"]]
        return [*values, *([0.0] * (action_dim - active_action_dim))]
    if "action_id" not in row:
        raise KeyError(
            f"row is missing action fields; available columns: {sorted(row)}"
        )
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= active_action_dim:
        raise ValueError(
            f"action_id={action_id} is outside action range [0, {active_action_dim - 1}]"
        )
    return [1.0 if idx == action_id else 0.0 for idx in range(action_dim)]


def _write_metadata(
    dataset_dir: Path,
    *,
    fps: float,
    action_labels: list[str],
    episode_lengths: list[int],
    task_prompts: list[str],
    image_shape: tuple[int, int, int],
    state_dim: int,
    state_labels: list[str],
    context_image_count: int | None,
) -> None:
    """Write minimal LeRobot metadata files for a split dataset.

    Args:
        dataset_dir: LeRobot dataset split directory.
        fps: Frame rate stored in metadata.
        action_labels: Canonical action labels in vector order.
        episode_lengths: Number of frames per converted episode.
        task_prompts: Unique prompts used as LeRobot tasks.
    """
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    action_dim = len(action_labels)
    height, width, channels = image_shape
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": fps},
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_labels,
        },
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_labels},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "latency": {"dtype": "int64", "shape": [1]},
    }
    if context_image_count is not None:
        features["observation.context_images"] = {
            "dtype": "image_sequence",
            "shape": [context_image_count, height, width, channels],
            "names": ["time", "height", "width", "channel"],
            "video_info": {"video.fps": fps},
        }
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "fps": fps,
                "total_episodes": len(episode_lengths),
                "total_frames": int(sum(episode_lengths)),
                "total_tasks": len(task_prompts),
                "total_videos": 0,
                "total_chunks": (len(episode_lengths) + LE_ROBOT_CHUNK_SIZE - 1)
                // LE_ROBOT_CHUNK_SIZE,
                "chunks_size": LE_ROBOT_CHUNK_SIZE,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(
            {
                "state": {
                    "game_state": {
                        "start": 0,
                        "end": state_dim,
                        "dtype": "float32",
                        "absolute": True,
                        "original_key": "observation.state",
                    }
                },
                "action": {
                    "button": {
                        "start": 0,
                        "end": action_dim,
                        "dtype": "float32",
                        "absolute": True,
                        "original_key": "action",
                    }
                },
                "video": {"image": {"original_key": "observation.image"}},
                "annotation": {
                    "human.action.task_description": {"original_key": "task_index"}
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": idx, "length": int(length)}
            for idx, length in enumerate(episode_lengths)
        ],
    )
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": idx, "task": prompt}
            for idx, prompt in enumerate(task_prompts)
        ],
    )


def _convert_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    env_name: str,
    integration_name: str,
    action_labels: list[str],
    active_action_dim: int,
    state_dim: int,
    state_labels: list[str],
    context_image_count: int | None,
    image_shape: tuple[int, int, int],
    fps: float,
    reward_field: str,
    state_normalization: dict[str, Any] | None,
    density_weighting: bool = False,
) -> dict[str, Any]:
    """Convert one canonical rollout split into a LeRobot dataset directory.

    Args:
        rows: Canonical rollout rows from one split.
        output_dir: Destination LeRobot split directory.
        env_name: Canonical benchmark environment name.
        action_labels: Canonical action labels in vector order.
        fps: Frame rate used for timestamps.
        reward_field: Source canonical reward column.
        density_weighting: When True, write per-row ``density_weight`` from clean_v1's
            event balancing (train split). When False, weights are uniform 1.0 (val
            split keeps an unweighted held-out loss).

    Returns:
        Split manifest summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_to_task_index: dict[str, int] = {}
    task_prompts: list[str] = []
    episode_lengths: list[int] = []
    for new_episode_idx, episode_rows in enumerate(_episode_groups(rows)):
        density_weights = (
            density_weights_clean_v1(env_name, episode_rows)
            if density_weighting and integration_name != "gymnasium"
            else [1.0] * len(episode_rows)
        )
        out_rows = []
        for frame_idx, row in enumerate(episode_rows):
            # Enforce the canonical rollout contract; downstream must not read the legacy ``latency`` alias.
            row["latency_raw_frames"]
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)
            out_row = {
                "image_bytes": _png_bytes(row["image"]),
                "action": _row_action_vector(
                    row,
                    active_action_dim=active_action_dim,
                    action_dim=len(action_labels),
                ),
                "timestamp": frame_idx / fps,
                "episode_index": new_episode_idx,
                "frame_index": frame_idx,
                "task_index": prompt_to_task_index[prompt],
                "latency_raw_frames": int(row["latency_raw_frames"]),
                # ``done`` is generated only for the LeRobot target format, not for canonical rollout rows.
                "done": frame_idx == len(episode_rows) - 1,
                "reward": float(row[reward_field]),
                "density_weight": float(density_weights[frame_idx]),
            }
            if state_normalization is not None:
                state = np.asarray(row["state"], dtype=np.float32)
                state_min = np.asarray(state_normalization["min"], dtype=np.float32)
                state_max = np.asarray(state_normalization["max"], dtype=np.float32)
                out_row["state"] = np.clip(
                    2.0 * (state - state_min) / (state_max - state_min) - 1.0,
                    -1.0,
                    1.0,
                ).tolist()
            if "action_id" in row:
                out_row["action_id"] = int(row["action_id"])
            if "context_images" in row and row["context_images"] is not None:
                out_row["context_image_bytes"] = [
                    _png_bytes(image) for image in row["context_images"]
                ]
            out_rows.append(out_row)
        episode_lengths.append(len(out_rows))
        episode_chunk = new_episode_idx // LE_ROBOT_CHUNK_SIZE
        _write_episode(
            output_dir
            / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
            out_rows,
            action_dim=len(action_labels),
            state_dim=state_dim,
        )
    _write_metadata(
        output_dir,
        fps=fps,
        action_labels=action_labels,
        episode_lengths=episode_lengths,
        task_prompts=task_prompts,
        image_shape=image_shape,
        state_dim=state_dim,
        state_labels=state_labels,
        context_image_count=context_image_count,
    )
    return {
        "dataset_name": output_dir.name,
        "env_name": env_name,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "task_prompts": task_prompts,
    }


def convert_canonical_rollout_to_lerobot(
    raw_export_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    action_carrier: str = "native",
) -> dict[str, Any]:
    """Convert a canonical rollout export into train/val LeRobot datasets.

    Args:
        raw_export_dir: Directory with canonical rollout parquet files and metadata.
        output_dir: Destination LeRobot train dataset directory.
        force: Remove existing train/val output directories before writing.

    Returns:
        Manifest describing the converted train dataset and validation companion.
    """
    raw_export_dir = Path(raw_export_dir)
    output_dir = Path(output_dir)
    metadata = json.loads(
        (raw_export_dir / "metadata.json").read_text(encoding="utf-8")
    )
    env_name = str(metadata["env_name"])
    integration_name = (
        metadata["integration_name"] if "integration_name" in metadata else env_name
    )
    if integration_name != "gymnasium":
        return _convert_legacy_export(
            raw_export_dir,
            output_dir,
            metadata=metadata,
            force=force,
        )
    action_layout = metadata["action_layout"]
    if action_layout not in (
        ACTION_LAYOUT_GYMNASIUM_DISCRETE_V1,
        ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ):
        raise ValueError(f"Unsupported Gymnasium action_layout={action_layout!r}")
    active_action_labels = list(metadata["action_labels"])
    gymnasium_task = metadata["gymnasium_task"]
    if env_name != gymnasium_task["task_name"]:
        raise ValueError(
            "Canonical Gymnasium env_name does not match gymnasium_task.task_name"
        )
    continuous = action_layout == ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1
    task_action_labels = (
        gymnasium_task["action_space"]["labels"]
        if continuous
        else gymnasium_task["action_labels"]
    )
    if active_action_labels != task_action_labels:
        raise ValueError(
            "Canonical Gymnasium action_labels do not match gymnasium_task action-space labels"
        )
    action_labels, active_action_dim, state_dim, state_labels = _action_carrier_spec(
        active_action_labels,
        action_carrier,
    )
    uses_state = "state_space" in gymnasium_task
    if uses_state:
        state_labels = list(gymnasium_task["state_space"]["labels"])
        state_dim = len(state_labels)
    reward_field = str(metadata["reward_field"])
    rows_unit = metadata["rows_unit"]
    if rows_unit != "decision_step":
        raise ValueError(f"rows_unit must be 'decision_step', got {rows_unit!r}")
    fps = metadata["obs_fps"]
    obs_stride = metadata["obs_stride_raw_frames"]
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)
    train_rows = _load_rows(raw_export_dir, "train")
    val_rows = _load_rows(raw_export_dir, "val")
    state_normalization = None
    if uses_state:
        train_states = np.asarray([row["state"] for row in train_rows], dtype=np.float32)
        state_normalization = {
            "type": "min_max",
            "min": train_states.min(axis=0).tolist(),
            "max": train_states.max(axis=0).tolist(),
        }
    first_row = (train_rows or val_rows)[0]
    image_source = first_row["image"]
    with Image.open(io.BytesIO(_png_bytes(image_source))) as image:
        width, height = image.size
        image_shape = (height, width, 3)
    context_image_count = (
        len(first_row["context_images"])
        if "context_images" in first_row and first_row["context_images"] is not None
        else None
    )
    train_manifest = _convert_split(
        train_rows,
        output_dir,
        env_name=env_name,
        integration_name=integration_name,
        action_labels=action_labels,
        active_action_dim=active_action_dim,
        state_dim=state_dim,
        state_labels=state_labels,
        context_image_count=context_image_count,
        image_shape=image_shape,
        fps=fps,
        reward_field=reward_field,
        state_normalization=state_normalization,
        density_weighting=True,
    )
    val_manifest = _convert_split(
        val_rows,
        val_output_dir,
        env_name=env_name,
        integration_name=integration_name,
        action_labels=action_labels,
        active_action_dim=active_action_dim,
        state_dim=state_dim,
        state_labels=state_labels,
        context_image_count=context_image_count,
        image_shape=image_shape,
        fps=fps,
        reward_field=reward_field,
        state_normalization=state_normalization,
        density_weighting=False,
    )
    handoff_contract = {
        "format": (
            "lerobot_v2_image_parquet"
            if continuous
            else "starvla_lerobot_v2_image_parquet"
        ),
        "source": str(raw_export_dir),
        "integration_name": integration_name,
        "task_name": env_name,
        "action_layout": str(metadata["action_layout"]),
        "action_labels": active_action_labels,
        "carrier_action_labels": action_labels,
        "action_dim": len(action_labels),
        "active_action_dim": active_action_dim,
        "action_carrier": action_carrier,
        "rows_unit": rows_unit,
        "obs_fps": fps,
        "obs_stride_raw_frames": obs_stride,
        "uses_state": uses_state,
        "state_dim": state_dim,
        "state_labels": state_labels,
        "state_normalization": state_normalization,
    }
    handoff_contract["robot_type"] = _robot_type_for_env(env_name, integration_name)
    prompt_map_path, mixture_path = _write_starvla_handoff_files(
        output_dir,
        val_output_dir,
        env_name=env_name,
        integration_name=integration_name,
        rows=[*train_rows, *val_rows],
    )
    handoff_contract.update(
        {
            "latency_prompt_map_path": str(prompt_map_path),
            "custom_mixtures_path": str(mixture_path),
        }
    )
    handoff_contract["gymnasium_task"] = gymnasium_task
    manifest = {
        **train_manifest,
        **handoff_contract,
        "validation_dataset_name": val_output_dir.name,
        "validation_episodes": val_manifest["episodes"],
        "validation_frames": val_manifest["frames"],
    }
    validation_manifest = {
        **val_manifest,
        **handoff_contract,
        "training_dataset_name": output_dir.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (val_output_dir / "manifest.json").write_text(
        json.dumps(validation_manifest, indent=2), encoding="utf-8"
    )
    return manifest
