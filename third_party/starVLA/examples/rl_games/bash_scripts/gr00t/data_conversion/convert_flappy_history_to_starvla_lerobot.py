#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.bash_scripts.gr00t.data_conversion import (
    convert_flappy_to_starvla_lerobot as flappy_converter,
)


DEFAULT_DATASET_NAME = "latency-sensitive-bench/benchmark-datasets"
DEFAULT_DATASET_CONFIG_NAME = "flappy_fixed_latency_3_200ep_7k2steps"
DEFAULT_OUTPUT_DIR = Path("data/flappy_fix_latency_3_200ep_context5/flappy_train__bridge")
SOURCE_OBSERVATION_FPS = 30.0
SOURCE_ENV_FPS = 30.0
SOURCE_ENV_FRAMESKIP = 1
SOURCE_COLUMNS = (
    "episode_idx",
    "decision_step",
    "action_id",
    "image",
    "prompt",
    "raw_reward",
    "latency_raw_frames",
    "latency_ms",
    "env_name",
    "split",
)


def _source_shard_paths(
    repo_files: list[str],
    dataset_config_name: str,
    source_split: str,
) -> list[str]:
    prefix = f"{dataset_config_name}/{source_split}-"
    paths = sorted(
        path
        for path in repo_files
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No parquet shards found for config={dataset_config_name!r}, split={source_split!r}"
        )
    return paths


def _downloaded_hub_shards(
    dataset_name: str,
    repo_paths: list[str],
    cache_dir: str | None,
    dataset_revision: str,
) -> Iterator[Path]:
    for repo_path in tqdm(repo_paths, desc="Downloading/caching Flappy shards"):
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=repo_path,
            repo_type="dataset",
            cache_dir=cache_dir,
            revision=dataset_revision,
        )
        yield Path(local_path)


def _iter_parquet_rows(
    parquet_paths: Iterable[Path],
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        available_columns = set(parquet.schema_arrow.names)
        missing_columns = set(SOURCE_COLUMNS) - available_columns
        if missing_columns:
            raise ValueError(
                f"Source shard {parquet_path} is missing columns={sorted(missing_columns)}; "
                f"available={sorted(available_columns)}"
            )
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(SOURCE_COLUMNS)):
            yield from batch.to_pylist()


def _context_image_entries(
    history: deque[bytes],
    first_image_bytes: bytes,
    context_image_count: int,
) -> list[dict[str, bytes | str | None]]:
    history_values = list(history)
    padding = [first_image_bytes] * (context_image_count - len(history_values))
    return [
        {"bytes": image_bytes, "path": None}
        for image_bytes in [*padding, *history_values]
    ]


def _validate_source_row(
    row: dict[str, Any],
    source_split: str,
) -> None:
    if str(row["env_name"]) != "flappy":
        raise ValueError(f"Expected env_name='flappy', got {row['env_name']!r}")
    allowed_splits = {"train"} if source_split == "train" else {"val", "validation"}
    if str(row["split"]).lower() not in allowed_splits:
        raise ValueError(
            f"Source split {source_split!r} contains row split={row['split']!r}"
        )


def _convert_episode(
    source_rows: list[dict[str, Any]],
    new_episode_idx: int,
    action_dim: int,
    image_sequence_length: int,
    prompt_to_task_index: dict[str, int],
    task_prompts: list[str],
    latency_prompt_entries: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    if not source_rows:
        raise ValueError("Cannot convert an empty Flappy episode")

    context_image_count = image_sequence_length - 1
    history: deque[bytes] = deque(maxlen=context_image_count)
    first_image_bytes: bytes | None = None
    previous_decision_step: int | None = None
    out_rows: list[dict[str, Any]] = []
    image_shape: list[int] | None = None

    for frame_idx, row in enumerate(source_rows):
        decision_step = int(row["decision_step"])
        if previous_decision_step is not None and decision_step <= previous_decision_step:
            raise ValueError(
                f"episode_idx={row['episode_idx']} decision_step must be strictly increasing; "
                f"got {previous_decision_step} then {decision_step}"
            )
        previous_decision_step = decision_step

        image_bytes = flappy_converter._png_bytes(row["image"])
        if first_image_bytes is None:
            first_image_bytes = image_bytes
            image_shape = flappy_converter._png_image_shape(image_bytes)

        prompt = str(row["prompt"])
        if prompt not in prompt_to_task_index:
            prompt_to_task_index[prompt] = len(task_prompts)
            task_prompts.append(prompt)
        latency = int(row["latency_raw_frames"])
        latency_prompt_entry = {
            "latency": latency,
            "latency_raw_frames": latency,
            "latency_ms": float(row["latency_ms"]),
            "prompt": prompt,
        }
        latency_prompt_key = (latency, prompt)
        existing_latency_prompt = latency_prompt_entries.get(latency_prompt_key)
        if (
            existing_latency_prompt is not None
            and existing_latency_prompt != latency_prompt_entry
        ):
            raise ValueError(
                f"Inconsistent latency metadata for latency={latency}, prompt={prompt!r}: "
                f"{existing_latency_prompt} and {latency_prompt_entry}"
            )
        latency_prompt_entries[latency_prompt_key] = latency_prompt_entry

        out_rows.append(
            {
                "image_bytes": image_bytes,
                "context_images": _context_image_entries(
                    history,
                    first_image_bytes,
                    context_image_count,
                ),
                "action": flappy_converter._one_hot(
                    int(row["action_id"]),
                    action_dim=action_dim,
                ),
                "timestamp": float(frame_idx) / SOURCE_OBSERVATION_FPS,
                "episode_index": new_episode_idx,
                "frame_index": frame_idx,
                "decision_step": decision_step,
                "task_index": prompt_to_task_index[prompt],
                "latency": latency,
                "done": frame_idx == len(source_rows) - 1,
                "reward": float(row["raw_reward"]),
                "action_id": int(row["action_id"]),
            }
        )
        history.append(image_bytes)

    if image_shape is None:
        raise ValueError("Converted Flappy episode has no image shape")
    return out_rows, image_shape


def _convert_split(
    parquet_paths: Iterable[Path],
    split_output_dir: Path,
    dataset_name: str,
    dataset_config_name: str,
    source_split: str,
    max_episodes: int | None,
    action_carrier: str,
    image_sequence_length: int,
    context_images_output_column: str,
    batch_size: int,
) -> dict[str, Any]:
    action_dim = flappy_converter._action_dim(action_carrier)
    action_labels = flappy_converter._action_labels(action_carrier)
    state_dim = flappy_converter._state_dim(action_carrier)
    state_labels = flappy_converter._state_labels(action_carrier)
    prompt_to_task_index: dict[str, int] = {}
    task_prompts: list[str] = []
    latency_prompt_entries: dict[tuple[int, str], dict[str, Any]] = {}
    episode_lengths: list[int] = []
    image_shape: list[int] | None = None
    written_episode_ids: set[int] = set()
    current_episode_id: int | None = None
    current_rows: list[dict[str, Any]] = []
    reached_episode_limit = False

    def _write_current_episode() -> None:
        nonlocal current_episode_id
        nonlocal current_rows
        nonlocal image_shape
        if current_episode_id is None:
            return
        if current_episode_id in written_episode_ids:
            raise ValueError(
                f"episode_idx={current_episode_id} appears after it was already written; "
                "source rows must be episode-contiguous"
            )
        new_episode_idx = len(episode_lengths)
        out_rows, episode_image_shape = _convert_episode(
            current_rows,
            new_episode_idx,
            action_dim,
            image_sequence_length,
            prompt_to_task_index,
            task_prompts,
            latency_prompt_entries,
        )
        if image_shape is None:
            image_shape = episode_image_shape
        elif image_shape != episode_image_shape:
            raise ValueError(
                f"Inconsistent image shapes across episodes: {image_shape} and {episode_image_shape}"
            )
        episode_chunk = new_episode_idx // 1000
        flappy_converter._write_episode(
            split_output_dir
            / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
            out_rows,
            action_dim=action_dim,
            state_dim=state_dim,
            context_images_output_column=context_images_output_column,
        )
        episode_lengths.append(len(out_rows))
        written_episode_ids.add(current_episode_id)
        current_episode_id = None
        current_rows = []

    for row in _iter_parquet_rows(parquet_paths, batch_size):
        _validate_source_row(row, source_split)
        episode_id = int(row["episode_idx"])
        if current_episode_id is None:
            if episode_id in written_episode_ids:
                raise ValueError(
                    f"episode_idx={episode_id} appears after it was already written; "
                    "source rows must be episode-contiguous"
                )
            current_episode_id = episode_id
        if episode_id != current_episode_id:
            _write_current_episode()
            if max_episodes is not None and len(episode_lengths) >= max_episodes:
                reached_episode_limit = True
                break
            if episode_id in written_episode_ids:
                raise ValueError(
                    f"episode_idx={episode_id} appears after it was already written; "
                    "source rows must be episode-contiguous"
                )
            current_episode_id = episode_id
        current_rows.append(row)

    if not reached_episode_limit:
        _write_current_episode()
    if not episode_lengths or image_shape is None:
        raise ValueError(
            f"{dataset_name}/{dataset_config_name} has no selected {source_split} episodes"
        )

    flappy_converter._write_metadata(
        split_output_dir,
        episode_lengths=episode_lengths,
        task_prompts=task_prompts,
        action_dim=action_dim,
        action_labels=action_labels,
        state_dim=state_dim,
        state_labels=state_labels,
        image_shape=image_shape,
        context_images_output_column=context_images_output_column,
        image_sequence_length=image_sequence_length,
        fps=SOURCE_OBSERVATION_FPS,
    )
    latency_prompt_map = flappy_converter.build_latency_prompt_map(
        list(latency_prompt_entries.values()),
        latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=SOURCE_ENV_FRAMESKIP,
    )
    for entry in latency_prompt_map.values():
        entry["latency_raw_frames"] = int(entry["latency"])
    (split_output_dir / "latency_prompt_map.json").write_text(
        json.dumps(latency_prompt_map, indent=2),
        encoding="utf-8",
    )

    source_latencies = sorted(
        {
            int(entry["latency_raw_frames"])
            for entry in latency_prompt_entries.values()
        }
    )
    manifest = {
        "dataset_name": split_output_dir.name,
        "split": "train" if source_split == "train" else "validation",
        "source": dataset_name,
        "source_config": dataset_config_name,
        "source_subdir": None,
        "latency_subdirs": [None],
        "source_split": source_split,
        "format": "starvla_lerobot_v2_image_parquet",
        "action_labels": action_labels,
        "action_dim": action_dim,
        "active_action_dim": flappy_converter.ACTION_DIM,
        "action_carrier": action_carrier,
        "bridge_action_dim": (
            flappy_converter.BRIDGE_ACTION_DIM
            if action_carrier == "bridge"
            else None
        ),
        "state_dim": state_dim,
        "active_state_dim": flappy_converter.STATE_DIM,
        "state_carrier": action_carrier,
        "context_source": "previous_episode_rows",
        "context_images_output_column": context_images_output_column,
        "image_sequence_length": image_sequence_length,
        "source_observation_fps": SOURCE_OBSERVATION_FPS,
        "source_env_fps": SOURCE_ENV_FPS,
        "source_env_frameskip": SOURCE_ENV_FRAMESKIP,
        "latency_metadata": True,
        "latency_unit": "raw_frames",
        "latency_raw_frames": source_latencies,
        "latency_env_steps": source_latencies,
        "latency_filter": source_latencies,
        "episodes_per_latency": max_episodes,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "max_episodes": max_episodes,
        "task_prompts": task_prompts,
    }
    (split_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def convert_hub_dataset(
    dataset_name: str,
    dataset_config_name: str,
    output_dir: Path,
    cache_dir: str | None,
    max_episodes: int | None,
    force: bool,
    action_carrier: str,
    image_sequence_length: int,
    context_images_output_column: str,
    batch_size: int,
    dataset_revision: str,
) -> dict[str, Any]:
    if image_sequence_length < 2:
        raise ValueError(
            f"image_sequence_length must be at least 2, got {image_sequence_length}"
        )
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    action_carrier = flappy_converter._normalize_action_carrier(action_carrier)
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    existing_outputs = [path for path in (output_dir, val_output_dir) if path.exists()]
    if existing_outputs and not force:
        raise FileExistsError(
            f"Output paths already exist: {[str(path) for path in existing_outputs]}; "
            "pass --force to replace them"
        )
    if force:
        for path in existing_outputs:
            shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)

    repo_files = HfApi().list_repo_files(dataset_name, repo_type="dataset", revision=dataset_revision)
    train_repo_paths = _source_shard_paths(
        repo_files,
        dataset_config_name,
        "train",
    )
    val_repo_paths = _source_shard_paths(
        repo_files,
        dataset_config_name,
        "val",
    )

    train_manifest = _convert_split(
        _downloaded_hub_shards(dataset_name, train_repo_paths, cache_dir, dataset_revision),
        output_dir,
        dataset_name,
        dataset_config_name,
        "train",
        max_episodes,
        action_carrier,
        image_sequence_length,
        context_images_output_column,
        batch_size,
    )
    val_manifest = _convert_split(
        _downloaded_hub_shards(dataset_name, val_repo_paths, cache_dir, dataset_revision),
        val_output_dir,
        dataset_name,
        dataset_config_name,
        "val",
        max_episodes,
        action_carrier,
        image_sequence_length,
        context_images_output_column,
        batch_size,
    )
    train_manifest["source_revision"] = dataset_revision
    val_manifest["source_revision"] = dataset_revision
    (val_output_dir / "manifest.json").write_text(
        json.dumps(val_manifest, indent=2), encoding="utf-8"
    )
    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    (output_dir / "manifest.json").write_text(
        json.dumps(train_manifest, indent=2),
        encoding="utf-8",
    )
    return train_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert row-history Flappy rollouts directly into the existing "
            "StarVLA LeRobot context-image format."
        )
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--dataset-config-name",
        default=DEFAULT_DATASET_CONFIG_NAME,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--action-carrier",
        choices=["native", "bridge"],
        default="bridge",
    )
    parser.add_argument("--image-sequence-length", type=int, default=5)
    parser.add_argument(
        "--context-images-output-column",
        default=flappy_converter.DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    manifest = convert_hub_dataset(
        args.dataset_name,
        args.dataset_config_name,
        args.output_dir,
        args.cache_dir,
        args.max_episodes,
        args.force,
        args.action_carrier,
        args.image_sequence_length,
        args.context_images_output_column,
        args.batch_size,
        args.dataset_revision,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
