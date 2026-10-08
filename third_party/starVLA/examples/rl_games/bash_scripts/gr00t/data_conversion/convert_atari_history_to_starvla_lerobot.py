#!/usr/bin/env python
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from typing import Any, ContextManager

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from tqdm import tqdm


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


def source_shard_paths(
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


def downloaded_hub_shards(
    dataset_name: str,
    repo_paths: list[str],
    cache_dir: str | None,
    *,
    desc: str,
    revision: str,
) -> Iterator[Path]:
    for repo_path in tqdm(repo_paths, desc=desc):
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
        )
        yield Path(local_path)


def iter_parquet_rows(
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
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=list(SOURCE_COLUMNS),
        ):
            yield from batch.to_pylist()


def context_image_entries(
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


def validate_source_row(
    row: dict[str, Any],
    *,
    source_env_name: str,
    source_split: str,
) -> None:
    if str(row["env_name"]) != source_env_name:
        raise ValueError(f"Expected env_name={source_env_name!r}, got {row['env_name']!r}")
    allowed_splits = {"train"} if source_split == "train" else {"val", "validation"}
    if str(row["split"]).lower() not in allowed_splits:
        raise ValueError(
            f"Source split {source_split!r} contains row split={row['split']!r}"
        )


def convert_episode(
    source_rows: list[dict[str, Any]],
    new_episode_idx: int,
    *,
    base_converter: Any,
    display_name: str,
    action_dim: int,
    image_sequence_length: int,
    source_observation_fps: float,
    prompt_to_task_index: dict[str, int],
    task_prompts: list[str],
    latency_prompt_entries: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    if not source_rows:
        raise ValueError(f"Cannot convert an empty {display_name} episode")

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

        image_bytes = base_converter._png_bytes(row["image"])
        if first_image_bytes is None:
            first_image_bytes = image_bytes
            image_shape = base_converter._image_shape(image_bytes)

        prompt = str(row["prompt"])
        if prompt not in prompt_to_task_index:
            prompt_to_task_index[prompt] = len(task_prompts)
            task_prompts.append(prompt)
        latency_raw_frames = int(row["latency_raw_frames"])
        latency_prompt_entry = {
            "latency": latency_raw_frames,
            "latency_raw_frames": latency_raw_frames,
            "latency_ms": float(row["latency_ms"]),
            "prompt": prompt,
        }
        latency_prompt_key = (latency_raw_frames, prompt)
        existing_latency_prompt = latency_prompt_entries.get(latency_prompt_key)
        if (
            existing_latency_prompt is not None
            and existing_latency_prompt != latency_prompt_entry
        ):
            raise ValueError(
                "Inconsistent latency metadata for "
                f"latency_raw_frames={latency_raw_frames}, prompt={prompt!r}: "
                f"{existing_latency_prompt} and {latency_prompt_entry}"
            )
        latency_prompt_entries[latency_prompt_key] = latency_prompt_entry

        out_rows.append(
            {
                "image_bytes": image_bytes,
                "context_images": context_image_entries(
                    history,
                    first_image_bytes,
                    context_image_count,
                ),
                "action": base_converter._one_hot(
                    int(row["action_id"]),
                    action_dim=action_dim,
                ),
                "timestamp": float(frame_idx) / source_observation_fps,
                "episode_index": new_episode_idx,
                "frame_index": frame_idx,
                "task_index": prompt_to_task_index[prompt],
                "latency": latency_raw_frames,
                "done": frame_idx == len(source_rows) - 1,
                "reward": float(row["raw_reward"]),
                "action_id": int(row["action_id"]),
            }
        )
        history.append(image_bytes)

    if image_shape is None:
        raise ValueError(f"Converted {display_name} episode has no image shape")
    return out_rows, image_shape


def convert_split(
    parquet_paths: Iterable[Path],
    split_output_dir: Path,
    *,
    dataset_name: str,
    dataset_config_name: str,
    source_split: str,
    source_env_name: str,
    display_name: str,
    base_converter: Any,
    max_episodes: int | None,
    action_carrier: str,
    image_sequence_length: int,
    context_images_output_column: str,
    batch_size: int,
    source_observation_fps: float,
    source_env_fps: float,
    source_env_frameskip: int,
    action_layout: str | None = None,
) -> dict[str, Any]:
    action_dim = base_converter._action_dim(action_carrier)
    action_labels = base_converter._action_labels(action_carrier)
    state_dim = base_converter._state_dim(action_carrier)
    state_labels = base_converter._state_labels(action_carrier)
    active_action_dim = int(base_converter.ACTION_DIM)
    active_state_dim = int(base_converter.STATE_DIM)
    bridge_action_dim = (
        int(base_converter.BRIDGE_ACTION_DIM)
        if action_carrier == "bridge"
        else None
    )
    prompt_to_task_index: dict[str, int] = {}
    task_prompts: list[str] = []
    latency_prompt_entries: dict[tuple[int, str], dict[str, Any]] = {}
    episode_lengths: list[int] = []
    image_shape: list[int] | None = None
    written_episode_ids: set[int] = set()
    current_episode_id: int | None = None
    current_rows: list[dict[str, Any]] = []
    reached_episode_limit = False

    def write_current_episode() -> None:
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
        out_rows, episode_image_shape = convert_episode(
            current_rows,
            new_episode_idx,
            base_converter=base_converter,
            display_name=display_name,
            action_dim=action_dim,
            image_sequence_length=image_sequence_length,
            source_observation_fps=source_observation_fps,
            prompt_to_task_index=prompt_to_task_index,
            task_prompts=task_prompts,
            latency_prompt_entries=latency_prompt_entries,
        )
        if image_shape is None:
            image_shape = episode_image_shape
        elif image_shape != episode_image_shape:
            raise ValueError(
                f"Inconsistent image shapes across episodes: {image_shape} and {episode_image_shape}"
            )
        episode_chunk = new_episode_idx // 1000
        base_converter._write_episode(
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

    for row in iter_parquet_rows(parquet_paths, batch_size):
        validate_source_row(
            row,
            source_env_name=source_env_name,
            source_split=source_split,
        )
        episode_id = int(row["episode_idx"])
        if current_episode_id is None:
            if episode_id in written_episode_ids:
                raise ValueError(
                    f"episode_idx={episode_id} appears after it was already written; "
                    "source rows must be episode-contiguous"
                )
            current_episode_id = episode_id
        if episode_id != current_episode_id:
            write_current_episode()
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
        write_current_episode()
    if not episode_lengths or image_shape is None:
        raise ValueError(
            f"{dataset_name}/{dataset_config_name} has no selected {source_split} episodes"
        )

    base_converter._write_metadata(
        split_output_dir,
        episode_lengths=episode_lengths,
        task_prompts=task_prompts,
        action_dim=action_dim,
        action_labels=action_labels,
        state_dim=state_dim,
        state_labels=state_labels,
        context_images_output_column=context_images_output_column,
        image_sequence_length=image_sequence_length,
        image_shape=image_shape,
        fps=source_observation_fps,
    )
    latency_prompt_map = base_converter.build_latency_prompt_map(
        list(latency_prompt_entries.values()),
        latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=source_env_frameskip,
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
        "source_split": source_split,
        "format": "starvla_lerobot_v2_image_parquet",
        "action_labels": action_labels,
        "action_dim": action_dim,
        "active_action_dim": active_action_dim,
        "action_carrier": action_carrier,
        "bridge_action_dim": bridge_action_dim,
        "state_dim": state_dim,
        "active_state_dim": active_state_dim,
        "state_carrier": action_carrier,
        "context_source": "previous_episode_rows",
        "context_images_output_column": context_images_output_column,
        "image_sequence_length": image_sequence_length,
        "source_observation_fps": source_observation_fps,
        "source_env_fps": source_env_fps,
        "source_env_frameskip": source_env_frameskip,
        "latency_unit": "raw_frames",
        "latency_raw_frames": source_latencies,
        "latency_env_steps": [
            latency / source_env_frameskip for latency in source_latencies
        ],
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "max_episodes": max_episodes,
        "task_prompts": task_prompts,
    }
    if action_layout:
        manifest["action_layout"] = action_layout
    (split_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def convert_hub_dataset(
    dataset_name: str,
    dataset_config_name: str,
    output_dir: Path,
    *,
    source_env_name: str,
    display_name: str,
    base_converter: Any,
    dataset_source_subdir: str,
    dataset_revision: str,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    action_carrier: str = "bridge",
    image_sequence_length: int = 5,
    context_images_output_column: str,
    batch_size: int = 256,
    source_observation_fps: float,
    source_env_fps: float,
    source_env_frameskip: int,
    conversion_context: Callable[[], ContextManager[Any]] | None = None,
    action_layout: str | None = None,
) -> dict[str, Any]:
    if image_sequence_length < 2:
        raise ValueError(
            f"image_sequence_length must be at least 2, got {image_sequence_length}"
        )
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    action_carrier = base_converter._normalize_action_carrier(action_carrier)
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
    train_repo_paths = source_shard_paths(repo_files, dataset_source_subdir, "train")
    val_repo_paths = source_shard_paths(repo_files, dataset_source_subdir, "val")
    context_factory = conversion_context or nullcontext

    with context_factory():
        train_manifest = convert_split(
            downloaded_hub_shards(
                dataset_name,
                train_repo_paths,
                cache_dir,
                desc=f"Downloading/caching {display_name} train shards",
                revision=dataset_revision,
            ),
            output_dir,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            source_split="train",
            source_env_name=source_env_name,
            display_name=display_name,
            base_converter=base_converter,
            max_episodes=max_episodes,
            action_carrier=action_carrier,
            image_sequence_length=image_sequence_length,
            context_images_output_column=context_images_output_column,
            batch_size=batch_size,
            source_observation_fps=source_observation_fps,
            source_env_fps=source_env_fps,
            source_env_frameskip=source_env_frameskip,
            action_layout=action_layout,
        )
        val_manifest = convert_split(
            downloaded_hub_shards(
                dataset_name,
                val_repo_paths,
                cache_dir,
                desc=f"Downloading/caching {display_name} val shards",
                revision=dataset_revision,
            ),
            val_output_dir,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            source_split="val",
            source_env_name=source_env_name,
            display_name=display_name,
            base_converter=base_converter,
            max_episodes=max_episodes,
            action_carrier=action_carrier,
            image_sequence_length=image_sequence_length,
            context_images_output_column=context_images_output_column,
            batch_size=batch_size,
            source_observation_fps=source_observation_fps,
            source_env_fps=source_env_fps,
            source_env_frameskip=source_env_frameskip,
            action_layout=action_layout,
        )

    for manifest, directory in [(train_manifest, output_dir), (val_manifest, val_output_dir)]:
        manifest["source_subdir"] = dataset_source_subdir
        manifest["source_revision"] = dataset_revision
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    (output_dir / "manifest.json").write_text(
        json.dumps(train_manifest, indent=2),
        encoding="utf-8",
    )
    return train_manifest
