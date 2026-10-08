#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import datasets
from datasets import load_dataset
try:
    from datasets.exceptions import DatasetGenerationError
except Exception:
    class DatasetGenerationError(Exception):
        pass
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
    build_latency_prompt_map,
    concatenate_latency_parts,
    latency_id_from_row,
    resolve_latency_subdirs,
    source_timing_from_args,
)


ACTION_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
ACTION_DIM = len(ACTION_LABELS)
BRIDGE_ACTION_DIM = 7
STATE_DIM = 1
BRIDGE_STATE_DIM = 7
DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN = "observation.context_images"
EpisodeKey = int | tuple[int, int]
READ_SCHEMA_EXCEPTIONS = (ValueError, KeyError, FileNotFoundError, pa.ArrowInvalid, DatasetGenerationError)


class DemonAttackColumns(NamedTuple):
    frame: str
    reward: str
    done: str | None
    latency: str | None
    latency_ms: str | None


def _local_parquet_files(dataset_name: str, split: str, dataset_source_subdir: str | None = None) -> list[str] | None:
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.exists():
        return None
    if dataset_source_subdir not in (None, ""):
        dataset_path = dataset_path / str(dataset_source_subdir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset_source_subdir={dataset_source_subdir!r} does not exist under {dataset_name!r}"
            )
    dataset_path = dataset_path.resolve(strict=True)
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"dataset_name={dataset_name!r} exists but is not a parquet file")
        return [str(dataset_path)]

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"dataset_name={dataset_name!r} exists but contains no parquet files")

    split_markers = {"train"} if split == "train" else {"validation", "val", "test"}
    split_files = [
        parquet_file
        for parquet_file in parquet_files
        if any(marker in part.lower() for marker in split_markers for part in parquet_file.relative_to(dataset_path).parts)
    ]
    return [str(parquet_file) for parquet_file in (split_files or parquet_files)]


def _local_parquet_columns(dataset_name: str, split: str, dataset_source_subdir: str | None = None) -> set[str] | None:
    if dataset_source_subdir not in (None, "") and "," in str(dataset_source_subdir):
        columns: set[str] = set()
        for subdir in str(dataset_source_subdir).split(","):
            subdir = subdir.strip()
            if not subdir:
                continue
            subdir_columns = _local_parquet_columns(dataset_name, split, subdir)
            if subdir_columns is not None:
                columns.update(subdir_columns)
        return columns or None
    local_files = _local_parquet_files(dataset_name, split, dataset_source_subdir)
    if local_files is None:
        return None
    columns: set[str] = set()
    for parquet_file in local_files:
        columns.update(pq.read_schema(parquet_file).names)
    return columns


def _resolve_required_column(available: set[str] | None, names: tuple[str, ...], label: str) -> str:
    if available is None:
        return names[0]
    for name in names:
        if name in available:
            return name
    raise ValueError(
        f"Demon Attack dataset is missing required {label} column. "
        f"Expected one of {names}; available={sorted(available)}"
    )


def _resolve_optional_column(available: set[str] | None, names: tuple[str, ...]) -> str | None:
    if available is None:
        return names[0]
    for name in names:
        if name in available:
            return name
    return None


def _resolve_demon_attack_columns(
    dataset_name: str,
    split: str,
    want_latency: bool,
    dataset_source_subdir: str | None = None,
) -> DemonAttackColumns:
    available = _local_parquet_columns(dataset_name, split, dataset_source_subdir)
    return DemonAttackColumns(
        frame=_resolve_required_column(available, ("t", "decision_step"), "frame index"),
        reward=_resolve_required_column(available, ("reward", "raw_reward"), "reward"),
        done=_resolve_optional_column(available, ("done",)),
        latency=_resolve_optional_column(available, ("latency", "latency_raw_frames")) if want_latency else None,
        latency_ms=_resolve_optional_column(available, ("latency_ms",)) if want_latency else None,
    )


def _demon_attack_column_candidates(
    dataset_name: str,
    split: str,
    want_latency: bool,
    dataset_source_subdir: str | None = None,
) -> list[DemonAttackColumns]:
    available = _local_parquet_columns(dataset_name, split, dataset_source_subdir)
    if available is not None:
        return [_resolve_demon_attack_columns(dataset_name, split, want_latency, dataset_source_subdir)]

    base_candidates = (
        DemonAttackColumns(frame="decision_step", reward="raw_reward", done=None, latency="latency_raw_frames", latency_ms="latency_ms"),
        DemonAttackColumns(frame="decision_step", reward="raw_reward", done=None, latency="latency", latency_ms="latency_ms"),
        DemonAttackColumns(frame="t", reward="reward", done="done", latency="latency", latency_ms="latency_ms"),
        DemonAttackColumns(frame="t", reward="reward", done="done", latency="latency_raw_frames", latency_ms="latency_ms"),
    )
    if want_latency:
        return list(base_candidates)
    return [
        DemonAttackColumns(frame=columns.frame, reward=columns.reward, done=columns.done, latency=None, latency_ms=None)
        for columns in base_candidates
    ]


def _load_hf_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_source_subdir: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    columns: list[str] | None = None,
):
    load_kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "columns": columns,
        # Some hosted RL-game datasets record validation as `val` in
        # dataset_info but generate it as `validation`. Do not let that
        # metadata-only mismatch block train/validation loading.
        "verification_mode": "no_checks",
    }
    if dataset_config_name not in (None, ""):
        return load_dataset(dataset_name, dataset_config_name, **load_kwargs)
    if dataset_source_subdir not in (None, ""):
        source_subdir = str(dataset_source_subdir)
        try:
            return load_dataset(dataset_name, source_subdir, **load_kwargs)
        except (ValueError, FileNotFoundError):
            load_kwargs["data_dir"] = source_subdir
            return load_dataset(dataset_name, **load_kwargs)
    return load_dataset(dataset_name, **load_kwargs)


def _load_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    columns: list[str] | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    latencies: list[int] | None = None,
):
    split_values = {"train"} if split == "train" else {"validation", "val", "test"}

    def _filter_internal_split(ds):
        if "split" in ds.column_names:
            return ds.filter(lambda row: str(row["split"]).lower() in split_values)
        return ds

    def _load_remote_parquet(subdir: str | None):
        if subdir in (None, "") or Path(dataset_name).expanduser().exists():
            return None
        split_files = ["train.parquet"] if split == "train" else ["validation.parquet", "val.parquet", "test.parquet"]
        load_columns = list(columns) if columns is not None else None
        if load_columns is not None and "split" not in load_columns:
            load_columns.append("split")
        last_exc: Exception | None = None
        for split_file in split_files:
            remote_file = f"hf://datasets/{dataset_name}/{str(subdir).strip('/')}/{split_file}"
            try:
                ds = load_dataset("parquet", data_files=[remote_file], split="train", cache_dir=cache_dir, columns=load_columns)
            except READ_SCHEMA_EXCEPTIONS as exc:
                last_exc = exc
                if columns is None:
                    continue
                try:
                    ds = load_dataset("parquet", data_files=[remote_file], split="train", cache_dir=cache_dir, columns=columns)
                except READ_SCHEMA_EXCEPTIONS as retry_exc:
                    last_exc = retry_exc
                    continue
            return _filter_internal_split(ds)
        return None

    def _load_one(subdir: str | None):
        local_files = _local_parquet_files(dataset_name, split, subdir)
        if local_files is not None:
            load_columns = list(columns) if columns is not None else None
            if load_columns is not None and "split" not in load_columns:
                load_columns.append("split")
            try:
                ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=load_columns)
            except READ_SCHEMA_EXCEPTIONS:
                if columns is None:
                    raise
                ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=columns)
            return _filter_internal_split(ds)

        remote_ds = _load_remote_parquet(subdir)
        if remote_ds is not None:
            return remote_ds

        if split == "train":
            try:
                ds = _load_hf_dataset(
                    dataset_name, dataset_config_name, subdir,
                    split="train", cache_dir=cache_dir, columns=columns,
                )
                return _filter_internal_split(ds)
            except READ_SCHEMA_EXCEPTIONS:
                pass
        else:
            for candidate in ("validation", "val", "test"):
                try:
                    ds = _load_hf_dataset(
                        dataset_name, dataset_config_name, subdir,
                        split=candidate, cache_dir=cache_dir, columns=columns,
                    )
                    if len(ds) > 0:
                        return _filter_internal_split(ds)
                except READ_SCHEMA_EXCEPTIONS:
                    continue

        load_columns = list(columns or [])
        if "split" not in load_columns:
            load_columns.append("split")
        try:
            ds_all = _load_hf_dataset(
                dataset_name, dataset_config_name, subdir,
                split="train", cache_dir=cache_dir, columns=load_columns or None,
            )
        except READ_SCHEMA_EXCEPTIONS:
            if columns is None:
                raise
            ds_all = _load_hf_dataset(
                dataset_name, dataset_config_name, subdir,
                split="train", cache_dir=cache_dir,
            )
        return ds_all.filter(lambda row: str(row["split"]).lower() in split_values)

    subdirs = resolve_latency_subdirs(dataset_source_subdir, latencies)
    parts = [_load_one(subdir) for subdir in subdirs]
    return concatenate_latency_parts(parts)


def _row_get(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row.get(name) is not None:
            return row[name]
    return default


def _png_bytes(image: Any) -> bytes:
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        image_path = image.get("path")
        if image_bytes is not None:
            return bytes(image_bytes)
        if image_path is not None:
            return Path(str(image_path)).read_bytes()
        raise ValueError("Image dict must contain `bytes` or `path`")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_struct(image: Any) -> dict[str, bytes | str | None]:
    return {"bytes": _png_bytes(image), "path": None}


def _image_shape(image_bytes: bytes) -> list[int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    return [height, width, 3]


def _context_images_from_context(
    row: dict[str, Any],
    *,
    context_images_column: str,
    image_sequence_length: int,
) -> list[dict[str, bytes | str | None]]:
    if image_sequence_length < 2:
        raise ValueError(f"image_sequence_length must be at least 2 for context image conversion, got {image_sequence_length}")
    if context_images_column not in row:
        raise ValueError(f"Row is missing context_images_column={context_images_column!r}")
    context_images = row[context_images_column]
    if context_images is None:
        raise ValueError(f"Row has null context_images_column={context_images_column!r}")
    context_image_list = list(context_images)
    expected_context_images = image_sequence_length - 1
    if len(context_image_list) != expected_context_images:
        raise ValueError(
            f"Expected {expected_context_images} context image(s), got {len(context_image_list)} "
            f"from column {context_images_column!r}"
        )
    return [_image_struct(image) for image in context_image_list]


def _normalize_action_carrier(action_carrier: str) -> str:
    carrier = str(action_carrier or "native").lower()
    if carrier in {"native", "bridge"}:
        return carrier
    raise ValueError(f"Unsupported action_carrier={action_carrier!r}; expected native or bridge")


def _action_dim(action_carrier: str) -> int:
    return BRIDGE_ACTION_DIM if _normalize_action_carrier(action_carrier) == "bridge" else ACTION_DIM


def _action_labels(action_carrier: str) -> list[str]:
    if _normalize_action_carrier(action_carrier) == "native":
        return list(ACTION_LABELS)
    return [*ACTION_LABELS, *[f"BRIDGE_PAD_{idx}" for idx in range(ACTION_DIM, BRIDGE_ACTION_DIM)]]


def _state_dim(action_carrier: str) -> int:
    return BRIDGE_STATE_DIM if _normalize_action_carrier(action_carrier) == "bridge" else STATE_DIM


def _state_labels(action_carrier: str) -> list[str]:
    if _normalize_action_carrier(action_carrier) == "native":
        return ["state"]
    return [f"BRIDGE_STATE_{idx}" for idx in range(BRIDGE_STATE_DIM)]


def _one_hot(action_id: int, *, action_dim: int = ACTION_DIM) -> list[float]:
    if action_id < 0 or action_id >= ACTION_DIM:
        raise ValueError(f"action_id={action_id} is outside Demon Attack action range [0, {ACTION_DIM - 1}]")
    values = [0.0] * action_dim
    values[action_id] = 1.0
    return values


def _select_episode_ids(
    episode_ids: list[EpisodeKey],
    episode_latencies: dict[EpisodeKey, int],
    *,
    max_episodes: int | None,
    require_latency_prompt_map: bool,
    episodes_per_latency: int | None = None,
    episodes_per_latency_by_latency: dict[int, int] | None = None,
) -> list[EpisodeKey]:
    if episodes_per_latency_by_latency is not None:
        if not episode_latencies:
            raise ValueError("episodes_per_latency_by_latency was requested, but no episode latency metadata is available")
        quotas = {int(latency): int(count) for latency, count in episodes_per_latency_by_latency.items()}
        selected: list[EpisodeKey] = []
        for latency in sorted(quotas):
            latency_episode_ids = [episode_id for episode_id in episode_ids if episode_latencies.get(episode_id) == latency]
            selected.extend(latency_episode_ids[: quotas[latency]])
        return selected

    if episodes_per_latency is not None:
        if not episode_latencies:
            raise ValueError("episodes_per_latency was requested, but no episode latency metadata is available")
        selected: list[EpisodeKey] = []
        for latency in sorted(set(episode_latencies.values())):
            latency_episode_ids = [episode_id for episode_id in episode_ids if episode_latencies.get(episode_id) == latency]
            selected.extend(latency_episode_ids[: int(episodes_per_latency)])
        return selected

    if max_episodes is None:
        return episode_ids
    if not require_latency_prompt_map:
        return episode_ids[:max_episodes]

    selected: list[EpisodeKey] = []
    selected_set: set[EpisodeKey] = set()
    for latency in sorted(set(episode_latencies.values())):
        for episode_id in episode_ids:
            if episode_id in selected_set:
                continue
            if episode_latencies.get(episode_id) == latency:
                selected.append(episode_id)
                selected_set.add(episode_id)
                break

    target_count = max(max_episodes, len(selected))
    for episode_id in episode_ids:
        if len(selected) >= target_count:
            break
        if episode_id not in selected_set:
            selected.append(episode_id)
            selected_set.add(episode_id)
    return selected


def _normalize_prompt_map(prompt_map: dict[str, Any] | dict[int, Any] | None) -> dict[int, dict[str, Any]]:
    if not prompt_map:
        return {}
    normalized: dict[int, dict[str, Any]] = {}
    for key, value in prompt_map.items():
        if not isinstance(value, dict):
            continue
        latency = int(value.get("latency", key))
        normalized[latency] = {
            "latency": latency,
            "latency_ms": value.get("latency_ms"),
            "prompt": str(value["prompt"]),
        }
    return normalized


def _row_latency(
    row: dict[str, Any],
    *,
    latency_column: str | None,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    default_latency: int | None,
) -> int | None:
    return latency_id_from_row(
        row,
        latency_column=latency_column,
        target_latency_unit=target_latency_unit,
        obs_stride_raw_frames=obs_stride_raw_frames,
        default_latency=default_latency,
    )


def _filter_latency(
    ds,
    latency_filter: list[int] | None,
    *,
    latency_column: str | None,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    default_latency: int | None,
):
    if not latency_filter:
        return ds
    allowed = {int(value) for value in latency_filter}
    if latency_column is None or latency_column not in ds.column_names:
        if default_latency is not None and int(default_latency) in allowed:
            return ds
        raise ValueError("latency_filter was requested, but the dataset has no latency column")
    return ds.filter(
        lambda row: _row_latency(
            row,
            latency_column=latency_column,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
            default_latency=default_latency,
        )
        in allowed
    )


def _episode_key(episode_idx: int, latency: int | None) -> EpisodeKey:
    return (episode_idx, int(latency)) if latency is not None else episode_idx


def _episode_sort_key(episode_key: EpisodeKey) -> tuple[int, int]:
    if isinstance(episode_key, tuple):
        return episode_key
    return (episode_key, -1)


def _load_index_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None,
    *,
    want_latency: bool,
    source_latency_column: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    latencies: list[int] | None = None,
):
    candidate_columns = _demon_attack_column_candidates(
        dataset_name,
        split,
        want_latency=want_latency,
        dataset_source_subdir=dataset_source_subdir,
    )
    if source_latency_column is not None:
        candidate_columns = [
            columns._replace(latency=source_latency_column)
            for columns in candidate_columns
        ]
    last_error: Exception | None = None
    for demon_attack_columns in candidate_columns:
        columns = ["episode_idx", demon_attack_columns.frame, "action_id", demon_attack_columns.reward, "prompt"]
        if demon_attack_columns.done is not None:
            columns.append(demon_attack_columns.done)
        if demon_attack_columns.latency is not None:
            columns.append(demon_attack_columns.latency)
        if demon_attack_columns.latency_ms is not None:
            columns.append(demon_attack_columns.latency_ms)
        try:
            return (
                _load_split(
                    dataset_name,
                    split,
                    cache_dir=cache_dir,
                    columns=columns,
                    dataset_config_name=dataset_config_name,
                    dataset_source_subdir=dataset_source_subdir,
                    latencies=latencies,
                ),
                demon_attack_columns,
            )
        except Exception as exc:
            last_error = exc
            if len(candidate_columns) == 1:
                raise

    try:
        ds = _load_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            latencies=latencies,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
        )
    except Exception:
        if last_error is not None:
            raise last_error
        raise
    available = set(ds.column_names)
    return (
        ds,
        DemonAttackColumns(
            frame=_resolve_required_column(available, ("t", "decision_step"), "frame index"),
            reward=_resolve_required_column(available, ("reward", "raw_reward"), "reward"),
            done=_resolve_optional_column(available, ("done",)),
            latency=source_latency_column if want_latency else None,
            latency_ms=_resolve_optional_column(available, ("latency_ms",)) if want_latency else None,
        ),
    )


def _canonical_prompt(
    row: dict[str, Any],
    *,
    prompt_map: dict[int, dict[str, Any]],
    latency_column: str | None,
    latency_ms_column: str | None,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    default_latency: int | None,
) -> tuple[str, int | None, Any]:
    latency = _row_latency(
        row,
        latency_column=latency_column,
        target_latency_unit=target_latency_unit,
        obs_stride_raw_frames=obs_stride_raw_frames,
        default_latency=default_latency,
    )
    if latency is not None and latency in prompt_map:
        entry = prompt_map[latency]
        return str(entry["prompt"]), latency, entry.get("latency_ms")
    latency_ms = row.get(latency_ms_column) if latency_ms_column is not None else None
    return str(row["prompt"]), latency, latency_ms


def _row_done(row: dict[str, Any], done_column: str | None, *, frame_idx: int, episode_length: int) -> bool:
    if done_column is not None and done_column in row:
        return bool(row[done_column])
    return frame_idx == episode_length - 1


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metadata(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    task_prompts: list[str],
    action_dim: int,
    action_labels: list[str],
    state_dim: int,
    state_labels: list[str],
    context_images_output_column: str | None,
    image_sequence_length: int | None,
    image_shape: list[int],
    fps: float,
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modality = {
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
        "video": {
            "image": {
                "original_key": "observation.image",
            }
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    info = {
        "codebase_version": "v2.0",
        "fps": fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": state_labels,
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
            "latency": {"dtype": "int64", "shape": [1]},
        },
    }
    if context_images_output_column is not None:
        info["features"][context_images_output_column] = {
            "dtype": "image_sequence",
            "shape": [int(image_sequence_length or 1) - 1, *image_shape],
            "names": ["time", "height", "width", "channel"],
            "video_info": {"video.fps": fps},
        }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    episodes = [
        {"episode_index": idx, "length": int(length)}
        for idx, length in enumerate(episode_lengths)
    ]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": idx, "task": prompt} for idx, prompt in enumerate(task_prompts)],
    )


def _write_episode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    action_dim: int,
    state_dim: int,
    context_images_output_column: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.image": pa.array(
            [{"bytes": row["image_bytes"], "path": None} for row in rows],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "observation.state": pa.array(
            [[0.0] * state_dim for _ in rows],
            type=pa.list_(pa.float32(), state_dim),
        ),
        "action": pa.array(
            [row["action"] for row in rows],
            type=pa.list_(pa.float32(), action_dim),
        ),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        "latency": pa.array([row["latency"] for row in rows], type=pa.int64()),
        "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
        "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
        "action_id": pa.array([row["action_id"] for row in rows], type=pa.int64()),
    }
    if context_images_output_column is not None:
        image_struct_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        columns[context_images_output_column] = pa.array(
            [row["context_images"] for row in rows],
            type=pa.list_(image_struct_type),
        )
    table = pa.table(columns)
    pq.write_table(table, path)


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
    latency_filter: list[int] | None = None,
    train_latency_filter: list[int] | None = None,
    eval_latency_filter: list[int] | None = None,
    episodes_per_latency: int | None = None,
    train_episodes_per_latency: int | None = None,
    eval_episodes_per_latency: int | None = None,
    episodes_per_latency_by_latency: dict[int, int] | None = None,
    train_episodes_per_latency_by_latency: dict[int, int] | None = None,
    eval_episodes_per_latency_by_latency: dict[int, int] | None = None,
    prompt_map_override: dict[str, Any] | dict[int, Any] | None = None,
    default_latency: int | None = None,
    action_carrier: str = "native",
    context_images_column: str | None = None,
    context_images_output_column: str | None = DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
    image_sequence_length: int = 4,
    fps: float,
    obs_stride_raw_frames: int,
    source_latency_column: str | None,
    target_latency_unit: str,
    source_rows_unit: str,
) -> dict[str, Any]:
    if source_rows_unit != "decision_step":
        raise ValueError(f"source_rows_unit must be decision_step, got {source_rows_unit!r}")
    action_carrier = _normalize_action_carrier(action_carrier)
    action_dim = _action_dim(action_carrier)
    action_labels = _action_labels(action_carrier)
    state_dim = _state_dim(action_carrier)
    state_labels = _state_labels(action_carrier)
    prompt_map_override = _normalize_prompt_map(prompt_map_override)
    if train_latency_filter is None:
        train_latency_filter = latency_filter
    if eval_latency_filter is None:
        eval_latency_filter = latency_filter
    if train_episodes_per_latency is None:
        train_episodes_per_latency = episodes_per_latency
    if eval_episodes_per_latency is None:
        eval_episodes_per_latency = episodes_per_latency
    if train_episodes_per_latency_by_latency is None:
        train_episodes_per_latency_by_latency = episodes_per_latency_by_latency
    if eval_episodes_per_latency_by_latency is None:
        eval_episodes_per_latency_by_latency = episodes_per_latency_by_latency
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)

    def _convert_split(
        split: str,
        split_output_dir: Path,
        *,
        split_latency_filter: list[int] | None,
        split_episodes_per_latency: int | None,
        split_episodes_per_latency_by_latency: dict[int, int] | None,
    ) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        want_latency = bool(
            source_latency_column is not None
            or require_latency_prompt_map
            or split_latency_filter
            or prompt_map_override
            or split_episodes_per_latency is not None
            or split_episodes_per_latency_by_latency is not None
        )
        ds_meta, demon_attack_columns = _load_index_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            want_latency=want_latency,
            source_latency_column=source_latency_column,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
            latencies=split_latency_filter,
        )
        ds_meta = _filter_latency(
            ds_meta,
            split_latency_filter,
            latency_column=demon_attack_columns.latency,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
            default_latency=default_latency,
        )
        if len(ds_meta) == 0:
            raise ValueError(f"{dataset_name} has no {split} rows")

        episode_indices: dict[EpisodeKey, list[tuple[int, int]]] = {}
        episode_latencies: dict[EpisodeKey, int] = {}
        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(tqdm(ds_meta, desc=f"Indexing Demon Attack {split} rows")):
            episode_idx = int(row["episode_idx"])
            latency = _row_latency(
                row,
                latency_column=demon_attack_columns.latency,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
                default_latency=default_latency,
            )
            episode_key = _episode_key(episode_idx, latency)
            episode_indices.setdefault(episode_key, []).append((int(row[demon_attack_columns.frame]), row_idx))
            if latency is not None:
                existing = episode_latencies.setdefault(episode_key, int(latency))
                if existing != int(latency):
                    raise ValueError(f"episode_key={episode_key!r} has inconsistent latencies: {existing} and {latency}")

        original_episode_ids = sorted(episode_indices, key=_episode_sort_key)
        original_episode_ids = _select_episode_ids(
            original_episode_ids,
            episode_latencies,
            max_episodes=max_episodes,
            require_latency_prompt_map=require_latency_prompt_map,
            episodes_per_latency=split_episodes_per_latency,
            episodes_per_latency_by_latency=split_episodes_per_latency_by_latency,
        )
        for episode_id in original_episode_ids:
            episode_indices[episode_id].sort(key=lambda item: item[0])

        ds_full = _filter_latency(
            _load_split(
                dataset_name,
                split,
                cache_dir=cache_dir,
                dataset_config_name=dataset_config_name,
                dataset_source_subdir=dataset_source_subdir,
                latencies=split_latency_filter,
            ),
            split_latency_filter,
            latency_column=demon_attack_columns.latency,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
            default_latency=default_latency,
        )
        if "image" in ds_full.column_names:
            ds_full = ds_full.cast_column("image", datasets.Image(decode=False))
        if context_images_column is not None:
            if context_images_column not in ds_full.column_names:
                raise ValueError(f"Configured context_images_column={context_images_column!r} is missing from {dataset_name}")
            ds_full = ds_full.cast_column(context_images_column, datasets.Sequence(datasets.Image(decode=False)))
        episode_lengths: list[int] = []

        for new_episode_idx, original_episode_idx in enumerate(tqdm(original_episode_ids, desc=f"Writing Demon Attack {split} LeRobot episodes")):
            row_indices = [row_idx for _, row_idx in episode_indices[original_episode_idx]]
            episode = ds_full.select(row_indices)
            out_rows = []
            for frame_idx, row in enumerate(episode):
                prompt, latency, latency_ms = _canonical_prompt(
                    row,
                    prompt_map=prompt_map_override,
                    latency_column=demon_attack_columns.latency,
                    latency_ms_column=demon_attack_columns.latency_ms,
                    target_latency_unit=target_latency_unit,
                    obs_stride_raw_frames=obs_stride_raw_frames,
                    default_latency=default_latency,
                )
                if prompt not in prompt_to_task_index:
                    prompt_to_task_index[prompt] = len(task_prompts)
                    task_prompts.append(prompt)
                if latency is not None:
                    latency_rows.append({
                        "latency": latency,
                        "latency_ms": latency_ms,
                        "prompt": prompt,
                    })
                out_row = {
                    "image_bytes": _png_bytes(row["image"]),
                    "action": _one_hot(int(row["action_id"]), action_dim=action_dim),
                    "timestamp": frame_idx / fps,
                    "episode_index": new_episode_idx,
                    "frame_index": frame_idx,
                    "task_index": prompt_to_task_index[prompt],
                    "latency": int(latency) if latency is not None else int(default_latency or 0),
                    "done": _row_done(
                        row,
                        demon_attack_columns.done,
                        frame_idx=frame_idx,
                        episode_length=len(episode),
                    ),
                    "reward": float(row[demon_attack_columns.reward]),
                    "action_id": int(row["action_id"]),
                }
                if context_images_column is not None:
                    out_row["context_images"] = _context_images_from_context(
                        row,
                        context_images_column=context_images_column,
                        image_sequence_length=image_sequence_length,
                    )
                out_rows.append(out_row)
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            _write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
                action_dim=action_dim,
                state_dim=state_dim,
                context_images_output_column=context_images_output_column if context_images_column is not None else None,
            )

        _write_metadata(
            split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            action_dim=action_dim,
            action_labels=action_labels,
            state_dim=state_dim,
            state_labels=state_labels,
            context_images_output_column=context_images_output_column if context_images_column is not None else None,
            image_sequence_length=image_sequence_length if context_images_column is not None else None,
            image_shape=[84, 84, 3],
            fps=fps,
        )

        if latency_rows:
            latency_prompt_map = build_latency_prompt_map(
                latency_rows,
                latency_column="latency",
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
            (split_output_dir / "latency_prompt_map.json").write_text(
                json.dumps(latency_prompt_map, indent=2),
                encoding="utf-8",
            )
        elif require_latency_prompt_map:
            raise ValueError(f"{dataset_name} {split} split has no latency rows; cannot build latency_prompt_map.json")

        manifest = {
            "dataset_name": split_output_dir.name,
            "split": split,
            "source": dataset_name,
            "source_config": dataset_config_name,
            "source_subdir": dataset_source_subdir,
            "latency_subdirs": [str(s) for s in resolve_latency_subdirs(dataset_source_subdir, split_latency_filter)],
            "format": "starvla_lerobot_v2_image_parquet",
            "action_labels": action_labels,
            "action_dim": action_dim,
            "active_action_dim": ACTION_DIM,
            "action_carrier": action_carrier,
            "bridge_action_dim": BRIDGE_ACTION_DIM if action_carrier == "bridge" else None,
            "latency_metadata": True,
            "source_latency_column": demon_attack_columns.latency,
            "source_latency_unit": (
                "raw_frames" if demon_attack_columns.latency == "latency_raw_frames"
                else "observation_steps" if demon_attack_columns.latency == "latency"
                else None
            ),
            "target_latency_unit": target_latency_unit,
            "obs_stride_raw_frames": obs_stride_raw_frames,
            "source_rows_unit": source_rows_unit,
            "fps": fps,
            "latency_filter": [int(value) for value in split_latency_filter] if split_latency_filter else None,
            "episodes_per_latency": int(split_episodes_per_latency) if split_episodes_per_latency is not None else None,
            "episodes_per_latency_by_latency": (
                {str(k): int(v) for k, v in sorted(split_episodes_per_latency_by_latency.items())}
                if split_episodes_per_latency_by_latency is not None
                else None
            ),
            "max_episodes": int(max_episodes) if max_episodes is not None else None,
            "prompt_override": bool(prompt_map_override),
            "default_latency": default_latency,
            "state_dim": state_dim,
            "active_state_dim": STATE_DIM,
            "state_carrier": action_carrier,
            "context_images_column": context_images_column,
            "context_images_output_column": context_images_output_column if context_images_column is not None else None,
            "image_sequence_length": int(image_sequence_length) if context_images_column is not None else None,
            "episodes": len(episode_lengths),
            "frames": int(sum(episode_lengths)),
            "task_prompts": task_prompts,
        }
        (split_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    train_manifest = _convert_split(
        "train",
        output_dir,
        split_latency_filter=train_latency_filter,
        split_episodes_per_latency=train_episodes_per_latency,
        split_episodes_per_latency_by_latency=train_episodes_per_latency_by_latency,
    )
    val_manifest = _convert_split(
        "validation",
        val_output_dir,
        split_latency_filter=eval_latency_filter,
        split_episodes_per_latency=eval_episodes_per_latency,
        split_episodes_per_latency_by_latency=eval_episodes_per_latency_by_latency,
    )
    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    train_manifest["validation_latency_filter"] = val_manifest["latency_filter"]
    train_manifest["validation_episodes_per_latency"] = val_manifest["episodes_per_latency"]
    train_manifest["validation_episodes_per_latency_by_latency"] = val_manifest["episodes_per_latency_by_latency"]
    (output_dir / "manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    return train_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--latency-filter", "--latency_filter", default=None)
    parser.add_argument("--episodes-per-latency", "--episodes_per_latency", type=int, default=None)
    parser.add_argument("--episodes-per-latency-by-latency", "--episodes_per_latency_by_latency", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", "--action_carrier", choices=["native", "bridge"], default="native")
    parser.add_argument("--context-images-column", "--context_images_column", default=None)
    parser.add_argument("--context-images-output-column", "--context_images_output_column", default=DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN)
    parser.add_argument("--image-sequence-length", "--image_sequence_length", type=int, default=4)
    parser.add_argument("--source-metadata")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--obs-stride-raw-frames", type=int)
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"])
    parser.add_argument(
        "--target-latency-unit",
        choices=["raw_frames", "observation_steps"],
        required=True,
    )
    args = parser.parse_args()
    fps, obs_stride_raw_frames, source_rows_unit = source_timing_from_args(
        source_metadata=args.source_metadata,
        source_fps=args.source_fps,
        obs_stride_raw_frames=args.obs_stride_raw_frames,
    )
    latency_filter = None
    if args.latency_filter:
        latency_filter = [int(item.strip()) for item in str(args.latency_filter).split(",") if item.strip()]
    episodes_per_latency_by_latency = None
    if args.episodes_per_latency_by_latency:
        episodes_per_latency_by_latency = {}
        for item in str(args.episodes_per_latency_by_latency).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "--episodes-per-latency-by-latency must use 'latency:episodes' items, "
                    f"got {args.episodes_per_latency_by_latency!r}"
                )
            latency, episodes = item.split(":", 1)
            episodes_per_latency_by_latency[int(latency.strip())] = int(episodes.strip())

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        dataset_config_name=args.dataset_config_name,
        dataset_source_subdir=args.dataset_source_subdir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=False,
        latency_filter=latency_filter,
        episodes_per_latency=args.episodes_per_latency,
        episodes_per_latency_by_latency=episodes_per_latency_by_latency,
        action_carrier=args.action_carrier,
        context_images_column=args.context_images_column,
        context_images_output_column=args.context_images_output_column,
        image_sequence_length=args.image_sequence_length,
        fps=fps,
        obs_stride_raw_frames=obs_stride_raw_frames,
        source_latency_column=args.source_latency_column,
        target_latency_unit=args.target_latency_unit,
        source_rows_unit=source_rows_unit,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
