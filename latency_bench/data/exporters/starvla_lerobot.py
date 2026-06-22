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

from latency_bench.data.action_layouts import action_spec_for_env

FPS_BY_ENV = {
    "flappy": 30,
    "demon_attack": 60,
    "deadly_corridor": 35,
}
STATE_DIM = 1


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


def _write_episode(path: Path, rows: list[dict[str, Any]], *, action_dim: int) -> None:
    """Write one LeRobot episode parquet file.

    Args:
        path: Destination episode parquet path.
        rows: Converted LeRobot frame rows.
        action_dim: Expected action vector dimension.
    """
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
            "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
            "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
            "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
        }
    )
    pq.write_table(table, path)


def _row_action_vector(row: dict[str, Any], *, action_dim: int) -> list[float]:
    if "action" in row and row["action"] is not None:
        return [float(value) for value in row["action"]]
    if "action_id" not in row:
        raise KeyError(f"row is missing action fields; available columns: {sorted(row)}")
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= action_dim:
        raise ValueError(f"action_id={action_id} is outside action range [0, {action_dim - 1}]")
    return [1.0 if idx == action_id else 0.0 for idx in range(action_dim)]


def _write_metadata(
    dataset_dir: Path,
    *,
    fps: int,
    action_labels: list[str],
    episode_lengths: list[int],
    task_prompts: list[str],
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
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "fps": int(fps),
                "features": {
                    "observation.image": {"dtype": "image"},
                    "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                    "action": {"dtype": "float32", "shape": [action_dim], "names": action_labels},
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
        [{"episode_index": idx, "length": int(length)} for idx, length in enumerate(episode_lengths)],
    )
    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": idx, "task": prompt} for idx, prompt in enumerate(task_prompts)])


def _convert_split(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    env_name: str,
    action_labels: list[str],
    fps: int,
    reward_field: str,
    obs_stride: int = 1,
) -> dict[str, Any]:
    """Convert one canonical rollout split into a LeRobot dataset directory.

    Args:
        rows: Canonical rollout rows from one split.
        output_dir: Destination LeRobot split directory.
        env_name: Canonical benchmark environment name.
        action_labels: Canonical action labels in vector order.
        fps: Frame rate used for timestamps.
        reward_field: Source canonical reward column.
        obs_stride: Raw frames per observation decision (``round(env_fps/obs_fps)``).
            Canonical rows are recorded once per raw frame; keeping only the obs
            decisions (``decision_step % obs_stride == 0``) makes adjacent LeRobot
            rows adjacent observations, so KV-memory rollouts unroll at the eval
            observation cadence.

    Returns:
        Split manifest summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_to_task_index: dict[str, int] = {}
    task_prompts: list[str] = []
    episode_lengths: list[int] = []
    for new_episode_idx, episode_rows in enumerate(_episode_groups(rows)):
        episode_rows = [row for row in episode_rows if int(row["decision_step"]) % obs_stride == 0]
        out_rows = []
        for frame_idx, row in enumerate(episode_rows):
            # Enforce the canonical rollout contract; downstream must not read the legacy ``latency`` alias.
            row["latency_raw_frames"]
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)
            out_rows.append(
                {
                    "image_bytes": _png_bytes(row["image"]),
                    "action": _row_action_vector(row, action_dim=len(action_labels)),
                    "timestamp": float(frame_idx) / int(fps),
                    "episode_index": new_episode_idx,
                    "frame_index": frame_idx,
                    "task_index": prompt_to_task_index[prompt],
                    # ``done`` is generated only for the LeRobot target format, not for canonical rollout rows.
                    "done": frame_idx == len(episode_rows) - 1,
                    "reward": float(row[reward_field]),
                }
            )
        episode_lengths.append(len(out_rows))
        episode_chunk = new_episode_idx // 1000
        _write_episode(
            output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
            out_rows,
            action_dim=len(action_labels),
        )
    _write_metadata(
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


def convert_canonical_rollout_to_lerobot(
    raw_export_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
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
    metadata = json.loads((raw_export_dir / "metadata.json").read_text(encoding="utf-8"))
    env_name = str(metadata["env_name"])
    action_spec = action_spec_for_env(env_name)
    action_labels = list(metadata["action_labels"])
    reward_field = str(metadata["reward_field"])
    obs_stride = int(metadata.get("obs_stride_raw_frames", 1))
    # Rows survive at the observation cadence, so timestamps advance at obs_fps.
    fps = max(1, int(FPS_BY_ENV[env_name]) // obs_stride)
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)
    train_manifest = _convert_split(
        _load_rows(raw_export_dir, "train"),
        output_dir,
        env_name=env_name,
        action_labels=action_labels,
        fps=fps,
        reward_field=reward_field,
        obs_stride=obs_stride,
    )
    val_manifest = _convert_split(
        _load_rows(raw_export_dir, "val"),
        val_output_dir,
        env_name=env_name,
        action_labels=action_labels,
        fps=fps,
        reward_field=reward_field,
        obs_stride=obs_stride,
    )
    manifest = {
        **train_manifest,
        "format": "starvla_lerobot_v2_image_parquet",
        "source": str(raw_export_dir),
        "action_layout": str(metadata["action_layout"]),
        "action_labels": action_labels,
        "action_dim": len(action_labels),
        "active_action_dim": len(action_spec.labels),
        "validation_dataset_name": val_output_dir.name,
        "validation_episodes": val_manifest["episodes"],
        "validation_frames": val_manifest["frames"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
