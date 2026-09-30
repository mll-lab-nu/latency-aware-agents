"""Hugging Face parquet IO helpers for canonical rollout datasets."""

from __future__ import annotations

import json
import shutil
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Mapping, Sequence

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm

from latency_bench.data.action_layouts import ActionSpec
from latency_bench.data.rollout_schema import rollout_features, sort_rows


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


def _dataset_from_rows(rows, features) -> Dataset:
    embedded_rows = embed_image_rows(rows)
    return Dataset.from_dict(
        {name: [row[name] for row in embedded_rows] for name in features},
        features=features,
    )


def load_parquet_splits(output_dir: Path) -> DatasetDict:
    """Read embedded images and HF features without loading the full dataset into RAM."""
    splits = {}
    for split in ("train", "val"):
        path = output_dir / f"{split}.parquet"
        # HF's parquet builder rejects a valid empty validation split.
        if pq.ParquetFile(path).metadata.num_rows == 0:
            splits[split] = Dataset(pq.read_table(path))
        else:
            splits[split] = Dataset.from_parquet(str(path))
    return DatasetDict(splits)


def _inspect_episode_shards(output_dir: Path):
    """Collect split counts, optional columns and image size in one shard scan."""
    stats = {}
    columns = set()
    image_resolution = None
    for split in ("train", "val"):
        row_count = 0
        episode_indices = set()
        for row in iter_episode_rows(output_dir, split):
            row_count += 1
            episode_indices.add(row["episode_idx"])
            columns.update(row)
            if image_resolution is None:
                with Image.open(row["image"]) as image:
                    image_resolution = [image.height, image.width, len(image.getbands())]
        stats[f"{split}_rows"] = row_count
        stats[f"{split}_episodes"] = len(episode_indices)
    stats["total_rows"] = stats["train_rows"] + stats["val_rows"]
    stats["total_episodes"] = stats["train_episodes"] + stats["val_episodes"]
    stats["image_resolution"] = image_resolution
    return stats, columns


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


def _write_parquet_split_from_shards(output_dir, split, *, features, row_count):
    """Atomically materialize one split with bounded image embedding batches."""
    path = output_dir / f"{split}.parquet"
    tmp_path = path.with_name(f"{path.name}.tmp")
    schema = _dataset_from_rows([], features).data.table.schema
    with pq.ParquetWriter(tmp_path, schema) as writer, tqdm(
        total=row_count, desc=f"write {path.name}", unit="row"
    ) as progress:
        batch = []
        for row in iter_episode_rows(output_dir, split):
            batch.append(row)
            if len(batch) == 256:
                writer.write_table(_dataset_from_rows(batch, features).data.table)
                progress.update(len(batch))
                batch = []
        if batch:
            writer.write_table(_dataset_from_rows(batch, features).data.table)
            progress.update(len(batch))
    tmp_path.replace(path)


def write_parquet_splits_from_shards(output_dir: Path, *, action_spec: ActionSpec):
    """Materialize train/val shards and return the written split counts and image size."""
    datasets.disable_progress_bars()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, columns = _inspect_episode_shards(output_dir)
    features = rollout_features(
        action_spec,
        include_context_images="context_images" in columns,
        include_state="state" in columns,
        include_latency_trace="command_admitted" in columns,
    )
    for split in ("train", "val"):
        _write_parquet_split_from_shards(
            output_dir, split, features=features, row_count=stats[f"{split}_rows"],
        )
    return stats


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
