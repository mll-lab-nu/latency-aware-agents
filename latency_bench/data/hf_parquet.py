"""Hugging Face parquet IO helpers for canonical rollout datasets."""

from __future__ import annotations

import json
import shutil
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Mapping, Sequence

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
from PIL import Image
from tqdm import tqdm

from latency_bench.data.action_layouts import ActionSpec
from latency_bench.data.rollout_schema import (
    empty_records_with_context_images,
    rollout_features,
    rollout_features_with_context_images,
    sort_rows,
)


def _embedded_image(path: str) -> dict[str, Any]:
    return {"bytes": Path(path).read_bytes(), "path": None}


def embed_image_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Embed image file bytes into row dictionaries for HF parquet writing.

    Args:
        rows: Canonical rollout rows whose ``image`` and optional ``context_images`` fields are local paths.

    Returns:
        Row copies with image path fields converted to Hugging Face image structs.
    """
    embedded_rows = []
    for row in rows:
        embedded = dict(row)
        embedded["image"] = _embedded_image(str(row["image"]))
        if "context_images" in row:
            embedded["context_images"] = [_embedded_image(str(path)) for path in row["context_images"]]
        embedded_rows.append(embedded)
    return embedded_rows


def _rows_include_context_images(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("context_images" in row for row in rows)


def _shards_include_context_images(output_dir: Path) -> bool:
    return any("context_images" in row for split in ("train", "val") for row in iter_episode_rows(output_dir, split))


def _rows_include_state(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("state" in row for row in rows)


def _shards_include_state(output_dir: Path) -> bool:
    return any("state" in row for split in ("train", "val") for row in iter_episode_rows(output_dir, split))


def _rows_include_termination(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("termination" in row for row in rows)


def _rows_include_actor_observation(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("actor_observation" in row for row in rows)


def _rows_include_latency_trace(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any("command_admitted" in row for row in rows)


def _shards_include_termination(output_dir: Path) -> bool:
    return any("termination" in row for split in ("train", "val") for row in iter_episode_rows(output_dir, split))


def _shards_include_actor_observation(output_dir: Path) -> bool:
    return any("actor_observation" in row for split in ("train", "val") for row in iter_episode_rows(output_dir, split))


def _shards_include_latency_trace(output_dir: Path) -> bool:
    return any("command_admitted" in row for split in ("train", "val") for row in iter_episode_rows(output_dir, split))


def dataset_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_spec: ActionSpec,
    include_context_images: bool,
    include_state: bool = False,
    include_termination: bool = False,
    include_actor_observation: bool = False,
    include_latency_trace: bool = False,
) -> Dataset:
    """Create a Hugging Face dataset from canonical rollout rows.

    Args:
        rows: Canonical rollout rows for one environment.
        action_spec: Action spec used to build the dataset features.
        include_context_images: Whether the dataset schema includes historical context images.

    Returns:
        Hugging Face ``Dataset`` with embedded images.
    """
    features = (
        rollout_features_with_context_images(
            action_spec,
            include_state=include_state,
            include_termination=include_termination,
            include_actor_observation=include_actor_observation,
            include_latency_trace=include_latency_trace,
        )
        if include_context_images
        else rollout_features(
            action_spec,
            include_state=include_state,
            include_termination=include_termination,
            include_actor_observation=include_actor_observation,
            include_latency_trace=include_latency_trace,
        )
    )
    if rows:
        embedded_rows = embed_image_rows(rows)
        return Dataset.from_dict(
            {
                name: [row[name] for row in embedded_rows]
                for name in features
            },
            features=features,
        )
    return Dataset.from_dict(
        empty_records_with_context_images(
            action_spec,
            include_context_images,
            include_state=include_state,
            include_termination=include_termination,
            include_actor_observation=include_actor_observation,
            include_latency_trace=include_latency_trace,
        ),
        features=features,
    )


def write_rows_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    action_spec: ActionSpec,
    include_context_images: bool,
    include_state: bool = False,
    include_termination: bool = False,
    include_actor_observation: bool = False,
    include_latency_trace: bool = False,
    row_batch_size: int = 256,
    keep_dataset: bool = True,
) -> Dataset | None:
    """Write canonical rows to a parquet file in stable row order.

    Args:
        path: Destination parquet path.
        rows: Canonical rollout rows for one split.
        action_spec: Action spec used to build the dataset features.
        include_context_images: Whether the dataset schema includes historical context images.
        row_batch_size: Number of rows embedded and written per batch.
        keep_dataset: Whether to return an in-memory ``Dataset``.

    Returns:
        Written dataset when requested, otherwise ``None``.
    """
    sorted_rows = sort_rows(rows)
    if not sorted_rows:
        dataset = dataset_from_rows(
            [],
            action_spec=action_spec,
            include_context_images=include_context_images,
            include_state=include_state,
            include_termination=include_termination,
            include_actor_observation=include_actor_observation,
            include_latency_trace=include_latency_trace,
        )
        pq.write_table(dataset.data.table, path)
        return dataset if keep_dataset else None

    writer = None
    datasets_to_return = []
    try:
        for start in range(0, len(sorted_rows), int(row_batch_size)):
            dataset = dataset_from_rows(
                sorted_rows[start : start + int(row_batch_size)],
                action_spec=action_spec,
                include_context_images=include_context_images,
                include_state=include_state,
                include_termination=include_termination,
                include_actor_observation=include_actor_observation,
                include_latency_trace=include_latency_trace,
            )
            if keep_dataset:
                datasets_to_return.append(dataset)
            table = dataset.data.table
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if not keep_dataset:
        return None
    if len(datasets_to_return) == 1:
        return datasets_to_return[0]
    return concatenate_datasets(datasets_to_return)


def write_parquet_splits(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    action_spec: ActionSpec,
    keep_dataset: bool = True,
):
    """Write train and validation parquet files.

    Args:
        output_dir: Directory receiving ``train.parquet`` and ``val.parquet``.
        train_rows: Canonical training rows.
        val_rows: Canonical validation rows.
        action_spec: Action spec shared by both splits.
        keep_dataset: Whether to return an in-memory ``DatasetDict``.

    Returns:
        ``DatasetDict`` when requested, otherwise ``None``.
    """
    datasets.disable_progress_bars()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_context_images = _rows_include_context_images([*train_rows, *val_rows])
    include_state = _rows_include_state([*train_rows, *val_rows])
    include_termination = _rows_include_termination([*train_rows, *val_rows])
    include_actor_observation = _rows_include_actor_observation([*train_rows, *val_rows])
    include_latency_trace = _rows_include_latency_trace([*train_rows, *val_rows])
    train_ds = write_rows_parquet(
        output_dir / "train.parquet",
        train_rows,
        action_spec=action_spec,
        include_context_images=include_context_images,
        include_state=include_state,
        include_termination=include_termination,
        include_actor_observation=include_actor_observation,
        include_latency_trace=include_latency_trace,
        keep_dataset=keep_dataset,
    )
    val_ds = write_rows_parquet(
        output_dir / "val.parquet",
        val_rows,
        action_spec=action_spec,
        include_context_images=include_context_images,
        include_state=include_state,
        include_termination=include_termination,
        include_actor_observation=include_actor_observation,
        include_latency_trace=include_latency_trace,
        keep_dataset=keep_dataset,
    )
    if not keep_dataset:
        return None
    return DatasetDict({"train": train_ds, "val": val_ds})


def iter_episode_rows(output_dir: Path, split: str):
    """Yield rows from per-episode JSONL shards.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.
        split: Split name, usually ``train`` or ``val``.

    Yields:
        Canonical rollout rows from the requested split.
    """
    for shard_path in sorted((output_dir / "episode_rows" / str(split)).glob("episode_*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def completed_episode_indices_from_shards(output_dir: Path) -> set[int]:
    """Read episode ids already materialized as JSONL shards.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.

    Returns:
        Episode indices with at least one shard row.
    """
    completed_episode_indices: set[int] = set()
    for split in ("train", "val"):
        for shard_path in sorted((output_dir / "episode_rows" / split).glob("episode_*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        completed_episode_indices.add(int(json.loads(line)["episode_idx"]))
                        break
    return completed_episode_indices


def episode_row_split_stats(output_dir: Path) -> dict[str, int]:
    """Count rows and episodes in train/val JSONL shards.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.

    Returns:
        Counts for train, val, and totals.
    """
    stats: dict[str, int] = {}
    for split in ("train", "val"):
        row_count = 0
        episode_indices: set[int] = set()
        for row in iter_episode_rows(output_dir, split):
            row_count += 1
            episode_indices.add(int(row["episode_idx"]))
        stats[f"{split}_rows"] = row_count
        stats[f"{split}_episodes"] = len(episode_indices)
    stats["total_rows"] = stats["train_rows"] + stats["val_rows"]
    stats["total_episodes"] = stats["train_episodes"] + stats["val_episodes"]
    return stats


def first_episode_row(output_dir: Path) -> dict[str, Any] | None:
    """Return the first row found in train or validation shards.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.

    Returns:
        First canonical row, or ``None`` when no shards exist.
    """
    for split in ("train", "val"):
        for row in iter_episode_rows(output_dir, split):
            return row
    return None


def image_metadata_from_shards(output_dir: Path) -> dict[str, Any] | None:
    """Infer image resolution metadata from the first sharded row.

    Args:
        output_dir: Rollout export directory containing images and shards.

    Returns:
        ``image_resolution`` metadata, or ``None`` when no row exists.
    """
    row = first_episode_row(output_dir)
    if row is None:
        return None
    with Image.open(row["image"]) as first_image:
        channels = len(first_image.getbands())
        return {"image_resolution": [int(first_image.height), int(first_image.width), int(channels)]}


def _write_parquet_row_batch(
    writer: Any,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    action_spec: ActionSpec,
    include_context_images: bool,
    include_state: bool,
    include_termination: bool,
    include_actor_observation: bool,
    include_latency_trace: bool,
) -> Any:
    """Append one row batch to a parquet writer.

    Args:
        writer: Existing parquet writer, or ``None`` for the first batch.
        path: Temporary parquet path used when creating the writer.
        rows: Canonical rollout rows in the batch.
        action_spec: Action spec used to build the dataset features.
        include_context_images: Whether the dataset schema includes historical context images.

    Returns:
        Open parquet writer after writing the batch.
    """
    dataset = dataset_from_rows(
        rows,
        action_spec=action_spec,
        include_context_images=include_context_images,
        include_state=include_state,
        include_termination=include_termination,
        include_actor_observation=include_actor_observation,
        include_latency_trace=include_latency_trace,
    )
    table = dataset.data.table
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _write_parquet_split_from_shards(
    output_dir: Path,
    split: str,
    *,
    action_spec: ActionSpec,
    include_context_images: bool,
    include_state: bool,
    include_termination: bool,
    include_actor_observation: bool,
    include_latency_trace: bool,
    row_count: int,
    row_batch_size: int = 256,
) -> None:
    """Write one split parquet file from per-episode shards.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.
        split: Split name to materialize.
        action_spec: Action spec used to build the dataset features.
        include_context_images: Whether the dataset schema includes historical context images.
        row_count: Number of rows for progress reporting.
        row_batch_size: Number of rows embedded and written per batch.
    """
    path = output_dir / f"{split}.parquet"
    tmp_path = path.with_name(f"{path.name}.tmp")
    writer = None
    batch: list[dict[str, Any]] = []
    progress = tqdm(total=int(row_count), desc=f"write {path.name}", unit="row")
    try:
        for row in iter_episode_rows(output_dir, split):
            batch.append(row)
            if len(batch) >= int(row_batch_size):
                writer = _write_parquet_row_batch(
                    writer,
                    tmp_path,
                    batch,
                    action_spec=action_spec,
                    include_context_images=include_context_images,
                    include_state=include_state,
                    include_termination=include_termination,
                    include_actor_observation=include_actor_observation,
                    include_latency_trace=include_latency_trace,
                )
                progress.update(len(batch))
                batch = []
        if batch:
            writer = _write_parquet_row_batch(
                writer,
                tmp_path,
                sort_rows(batch),
                action_spec=action_spec,
                include_context_images=include_context_images,
                include_state=include_state,
                include_termination=include_termination,
                include_actor_observation=include_actor_observation,
                include_latency_trace=include_latency_trace,
            )
            progress.update(len(batch))
        if writer is None:
            dataset = dataset_from_rows(
                [],
                action_spec=action_spec,
                include_context_images=include_context_images,
                include_state=include_state,
                include_termination=include_termination,
                include_actor_observation=include_actor_observation,
                include_latency_trace=include_latency_trace,
            )
            pq.write_table(dataset.data.table, tmp_path)
    finally:
        if writer is not None:
            writer.close()
        progress.close()
    tmp_path.replace(path)


def write_parquet_splits_from_shards(
    output_dir: Path,
    *,
    action_spec: ActionSpec,
    keep_dataset: bool = True,
):
    """Materialize train/val parquet files from episode row shards.

    Args:
        output_dir: Rollout export directory containing JSONL shards.
        action_spec: Action spec shared by both splits.
        keep_dataset: Whether to return an in-memory ``DatasetDict``.

    Returns:
        ``DatasetDict`` when requested, otherwise ``None``.
    """
    datasets.disable_progress_bars()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = episode_row_split_stats(output_dir)
    include_context_images = _shards_include_context_images(output_dir)
    include_state = _shards_include_state(output_dir)
    include_termination = _shards_include_termination(output_dir)
    include_actor_observation = _shards_include_actor_observation(output_dir)
    include_latency_trace = _shards_include_latency_trace(output_dir)
    _write_parquet_split_from_shards(
        output_dir,
        "train",
        action_spec=action_spec,
        include_context_images=include_context_images,
        include_state=include_state,
        include_termination=include_termination,
        include_actor_observation=include_actor_observation,
        include_latency_trace=include_latency_trace,
        row_count=stats["train_rows"],
    )
    _write_parquet_split_from_shards(
        output_dir,
        "val",
        action_spec=action_spec,
        include_context_images=include_context_images,
        include_state=include_state,
        include_termination=include_termination,
        include_actor_observation=include_actor_observation,
        include_latency_trace=include_latency_trace,
        row_count=stats["val_rows"],
    )
    if not keep_dataset:
        return None
    return DatasetDict(
        {
            "train": Dataset(pq.read_table(output_dir / "train.parquet")),
            "val": Dataset(pq.read_table(output_dir / "val.parquet")),
        }
    )


def cleanup_rollout_intermediates(output_dir: Path) -> None:
    """Remove image and JSONL shard intermediates after parquet materialization.

    Args:
        output_dir: Rollout export directory.
    """
    for name in ("images", "episode_rows", "probe_state.json"):
        path = output_dir / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def episode_row_shard_path(output_dir: Path, split: str, episode_idx: int) -> Path:
    """Build the JSONL shard path for one episode.

    Args:
        output_dir: Rollout export directory.
        split: Row split name.
        episode_idx: Episode index.

    Returns:
        Path to the episode shard.
    """
    return output_dir / "episode_rows" / str(split) / f"episode_{int(episode_idx):06d}.jsonl"


def write_episode_row_shard(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write one episode's canonical rows as an atomic JSONL shard.

    Args:
        output_dir: Rollout export directory.
        rows: Canonical rows from a single completed episode.
    """
    split = str(rows[0]["split"])
    episode_idx = int(rows[0]["episode_idx"])
    shard_path = episode_row_shard_path(output_dir, split, episode_idx)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_name(f"{shard_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in sort_rows(rows):
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
    tmp_path.replace(shard_path)


def load_episode_row_shards(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    """Load all JSONL shards into train/val row lists.

    Args:
        output_dir: Rollout export directory containing ``episode_rows``.

    Returns:
        Train rows, validation rows, and completed episode indices.
    """
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    completed_episode_indices: set[int] = set()
    for split in ("train", "val"):
        for shard_path in sorted((output_dir / "episode_rows" / split).glob("episode_*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if rows:
                completed_episode_indices.add(int(rows[0]["episode_idx"]))
                if split == "train":
                    train_rows.extend(rows)
                else:
                    val_rows.extend(rows)
    return sort_rows(train_rows), sort_rows(val_rows), completed_episode_indices
