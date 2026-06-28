"""End-to-end export orchestration for Sample Factory teacher datasets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.data.hf_parquet import (
    cleanup_rollout_intermediates,
    episode_row_split_stats,
    image_metadata_from_shards,
    write_parquet_splits_from_shards,
)
from latency_bench.data.prompts import BASE_PROMPTS, IMAGE_TRANSFORM_CONTEXTS, TIMING_SUFFIX
from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, FILTER_PRESET_NONE
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.data.sf_teacher_config import load_rollout_training_config
from latency_bench.data.sf_teacher_rollout import collect_teacher_dataset
from latency_bench.utils.hf import resolve_token


def push_dataset_to_hub(
    dataset_dict: Any,
    *,
    repo_id: str,
    private: bool,
    token: str | None,
    max_shard_size: str,
    config_name: str = "",
) -> None:
    """Push the exported HF dataset dict to the Hub.

    Args:
        dataset_dict: Hugging Face ``DatasetDict`` to upload.
        repo_id: Destination dataset repository id.
        private: Whether the repository should be private.
        token: Optional Hugging Face token.
        max_shard_size: Hub upload shard size.
        config_name: Optional HF dataset config name for multi-config repos.
    """
    push_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "private": bool(private),
        "max_shard_size": str(max_shard_size),
    }
    if token:
        push_kwargs["token"] = token
    if config_name:
        push_kwargs["config_name"] = str(config_name)
    dataset_dict.push_to_hub(**push_kwargs)


def _load_written_parquet_splits(output_dir: Path) -> Any:
    """Load on-disk train/val parquet memory-mapped for a low-RAM hub push.

    The parquet files embed image bytes. Reading them fully into memory
    (``pq.read_table``) and then pushing rebuilds the whole image-embedded dataset in RAM,
    which OOMs on large datasets. ``Dataset.from_parquet`` memory-maps the Arrow data from
    disk so the push streams shard-by-shard with bounded memory. Features (incl. the
    embedded Image column) are restored from the parquet's Hugging Face metadata.

    Args:
        output_dir: Directory containing ``train.parquet`` and ``val.parquet``.

    Returns:
        A ``DatasetDict`` with memory-mapped ``train``/``val`` splits.
    """
    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_parquet(str(output_dir / "train.parquet")),
            "val": Dataset.from_parquet(str(output_dir / "val.parquet")),
        }
    )


def export_sample_factory_teacher_dataset(
    *,
    checkpoint_root: str | Path,
    output_dir: Path,
    latency_config_path: str | Path | None,
    env_name: str,
    max_episodes: int,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    replay_num_envs: int = 16,
    device: str,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    push_to_hub: bool,
    hf_repo_id: str,
    hf_private: bool,
    hf_token: str,
    hf_max_shard_size: str,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    filter_kwargs: dict[str, Any] | None = None,
    hf_config_name: str = "",
    rollout_seed_offset: int = 0,
    probe_only: bool = False,
    image_transform_config: GhostTrailConfig | None = None,
) -> dict[str, Any]:
    """Collect and write a canonical Sample Factory teacher dataset.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.
        output_dir: Directory receiving parquet files and metadata.
        latency_config_path: Optional latency-only YAML override.
        env_name: Optional canonical environment name override.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Step cap per episode.
        val_fraction: Fraction assigned to validation.
        split_seed: RNG seed for split assignment.
        num_envs: Number of probe rollout environments or agents.
        replay_num_envs: Number of replay rollout environments or agents.
        device: Optional rollout device override.
        deterministic: Whether to use deterministic teacher actions.
        export_image_size: Optional ``(width, height)`` resize target.
        image_writer_workers: Number of async image writer threads.
        row_buffer_size: Pending image write threshold.
        push_to_hub: Whether to upload the written dataset.
        hf_repo_id: Destination Hugging Face dataset repository id.
        hf_private: Whether the uploaded repository should be private.
        hf_token: Explicit Hugging Face token.
        hf_max_shard_size: Hub upload shard size.
        image_transform_config: Optional image transform applied during replay
            before row filtering.

    Returns:
        Final metadata written to ``metadata.json``.
    """
    config, config_metadata = load_rollout_training_config(
        checkpoint_root,
        latency_config_path=latency_config_path,
        device_override=device or None,
    )
    if env_name:
        config["env_name"] = str(env_name)
    rollout_metadata = collect_teacher_dataset(
        config,
        output_dir=output_dir,
        max_episodes=int(max_episodes),
        max_steps_per_episode=int(max_steps_per_episode),
        val_fraction=float(val_fraction),
        split_seed=int(split_seed),
        num_envs=int(num_envs),
        replay_num_envs=int(replay_num_envs),
        deterministic=bool(deterministic),
        export_image_size=export_image_size,
        image_writer_workers=int(image_writer_workers),
        row_buffer_size=int(row_buffer_size),
        filter_preset=str(filter_preset),
        filter_kwargs=filter_kwargs,
        rollout_seed_offset=int(rollout_seed_offset),
        probe_only=bool(probe_only),
        image_transform_config=image_transform_config,
    )
    if probe_only:
        # Sharded-probe worker: probe_state.json holds this shard's accepted specs.
        # Skip parquet/image materialization and hub upload; the merge + replay step
        # handles the combined dataset.
        return rollout_metadata
    split_stats = episode_row_split_stats(output_dir)
    shard_image_metadata = image_metadata_from_shards(output_dir)
    write_parquet_splits_from_shards(
        output_dir,
        action_spec=action_spec_for_env(str(config["env_name"])),
        keep_dataset=False,
    )
    metadata = {
        **config_metadata,
        **rollout_metadata,
        "env_name": config["env_name"],
        "prompt_source": "latency_bench.data.prompts",
        "base_prompt": BASE_PROMPTS[str(config["env_name"])],
        "image_transform_prompt_contexts": IMAGE_TRANSFORM_CONTEXTS,
        "prompt_timing_suffix": TIMING_SUFFIX,
        "max_episodes": int(max_episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "num_envs": int(num_envs),
        "replay_num_envs": int(replay_num_envs),
        "filter_preset": rollout_metadata["filter_preset"],
        "filter_report": rollout_metadata["filter_report"],
        "reward_field": rollout_metadata["reward_field"],
        "episode_return_field": rollout_metadata["episode_return_field"],
        "reward_semantics": rollout_metadata["reward_semantics"],
        "image_source": rollout_metadata["image_source"],
        "train_rows": split_stats["train_rows"],
        "val_rows": split_stats["val_rows"],
        "total_rows": split_stats["total_rows"],
        "rows_unit": "decision_step",
        "train_episodes": split_stats["train_episodes"],
        "val_episodes": split_stats["val_episodes"],
        "total_episodes": split_stats["total_episodes"],
        "image_resolution": shard_image_metadata["image_resolution"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(rollout_metadata["filter_report"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    cleanup_rollout_intermediates(output_dir)

    if push_to_hub:
        # Load the just-written parquet memory-mapped for the push. Building the
        # DatasetDict via pq.read_table (the old keep_dataset path) pulls the whole
        # image-embedded dataset into RAM and OOMs on large datasets; from_parquet
        # memory-maps the Arrow data so push_to_hub streams shard-by-shard.
        push_dataset_to_hub(
            _load_written_parquet_splits(output_dir),
            repo_id=str(hf_repo_id),
            private=bool(hf_private),
            token=resolve_token(str(hf_token)),
            max_shard_size=str(hf_max_shard_size),
            config_name=str(hf_config_name),
        )
    return deepcopy(metadata)
