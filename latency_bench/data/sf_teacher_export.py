"""End-to-end export orchestration for Sample Factory teacher datasets."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from latency_bench.data.hf_parquet import (
    cleanup_rollout_intermediates,
    load_parquet_splits,
    write_parquet_splits_from_shards,
)
from latency_bench.data.prompts import BASE_PROMPTS, IMAGE_TRANSFORM_CONTEXTS, TIMING_SUFFIX
from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1,
    action_spec_for_env,
)
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, FILTER_PRESET_NONE, demon_raw_score_bin_counts
from latency_bench.data.rollout_schema import SCHEMA_VERSION
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.data.sf_teacher_config import load_rollout_training_config
from latency_bench.data.sf_teacher_rollout import collect_teacher_dataset
from latency_bench.utils.hf import resolve_token


def export_sample_factory_teacher_dataset(
    *,
    checkpoint_root: str | Path,
    output_dir: Path,
    latency_config_path: str | Path | None,
    gym_camera_name: str | None = None,
    gym_make_kwargs_override: Mapping[str, Any] | None = None,
    env_name: str,
    max_episodes: int,
    max_attempts: int | None = None,
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
    demon_episode_threshold: float | None = None,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
    hf_config_name: str = "",
    hf_data_dir: str | None = None,
    context_window: int = 1,
    rollout_seed_offset: int = 0,
    probe_only: bool = False,
    image_transform_config: GhostTrailConfig | None = None,
    export_zero_action_history: int | None = None,
) -> dict[str, Any]:
    """Collect and write a canonical Sample Factory teacher dataset.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.
        output_dir: Directory receiving parquet files and metadata.
        latency_config_path: Optional latency-only YAML override.
        gym_camera_name: Optional camera name override for Gymnasium checkpoints.
        gym_make_kwargs_override: Optional Gymnasium reset/runtime overrides.
        env_name: Optional canonical environment name override.
        max_episodes: Number of episodes to collect.
        max_steps_per_episode: Raw environment frame cap per episode.
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
        hf_data_dir: Optional repository directory for the published Parquet splits.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.
        image_transform_config: Optional image transform applied during replay
            before row filtering.
        export_zero_action_history: Student history depth for a zero-latency teacher
            whose own observation input does not contain history.

    Returns:
        Final metadata written to ``metadata.json``.
    """
    config, config_metadata = load_rollout_training_config(
        checkpoint_root,
        latency_config_path=latency_config_path,
        device_override=device or None,
        gym_camera_name=gym_camera_name,
        gym_make_kwargs_override=gym_make_kwargs_override,
    )
    if export_zero_action_history is not None:
        config["export_zero_action_history"] = export_zero_action_history
    if env_name:
        if config["integration_name"] == "gymnasium":
            raise ValueError(
                "Gymnasium task identity is fixed by the teacher checkpoint; "
                "do not pass --env-name"
            )
        config["env_name"] = str(env_name)
        config["action_spec"] = action_spec_for_env(config["env_name"])
        config["base_prompt"] = BASE_PROMPTS[config["env_name"]]
    action_spec = config["action_spec"]
    if action_spec.layout in (
        ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
        ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1,
    ):
        filter_preset = FILTER_PRESET_NONE
    probe_state = collect_teacher_dataset(
        config,
        output_dir=output_dir,
        max_episodes=int(max_episodes),
        max_attempts=max_attempts,
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
        demon_episode_threshold=demon_episode_threshold,
        episode_return_gt=episode_return_gt,
        episode_acceptance=episode_acceptance,
        context_window=int(context_window),
        rollout_seed_offset=int(rollout_seed_offset),
        probe_only=bool(probe_only),
        image_transform_config=image_transform_config,
    )
    if probe_only:
        # Sharded-probe worker mode: the accepted_specs (with their reset seeds) are now
        # persisted in probe_state.json; skip replay/image writing. A separate merge step
        # combines shards and the single-process replay path materializes the dataset.
        metadata = {
            "probe_only": True,
            "env_name": config["env_name"],
            "accepted_episodes": len(probe_state["accepted_specs"]),
            "attempted_episodes": int(probe_state["attempted_episodes"]),
            "rejected_episodes": int(probe_state["rejected_episodes"]),
            "attempt_budget": max_attempts,
            "rollout_seed_offset": int(rollout_seed_offset),
            "filter_preset": filter_preset,
        }
        if demon_episode_threshold is not None:
            metadata["demon_episode_threshold"] = demon_episode_threshold
        if episode_acceptance is not None:
            metadata["episode_acceptance"] = dict(episode_acceptance)
        if episode_return_gt is not None:
            metadata["episode_return_gate"] = {
                "field": action_spec.episode_return_field,
                "operator": ">",
                "threshold": float(episode_return_gt),
            }
        return metadata

    split_stats = write_parquet_splits_from_shards(output_dir, action_spec=action_spec)
    image_transform_config = GhostTrailConfig() if image_transform_config is None else image_transform_config
    runtime_metadata = probe_state["runtime_metadata"]
    env_fps = runtime_metadata["env_fps"]
    obs_fps = runtime_metadata["obs_fps"]
    filter_report = _filter_report_from_probe_state(
        split_stats,
        probe_state,
        env_name=config["integration_name"],
        filter_preset=filter_preset,
        episode_return_field=action_spec.episode_return_field,
        episode_return_gt=episode_return_gt,
        episode_acceptance=episode_acceptance,
    )
    if demon_episode_threshold is not None:
        filter_report["demon_episode_threshold"] = demon_episode_threshold
    rollout_metadata = {
        "schema_version": SCHEMA_VERSION,
        "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": action_spec.reward_field,
        "episode_return_field": action_spec.episode_return_field,
        "reward_semantics": action_spec.reward_semantics,
        "filter_preset": filter_preset,
        "filter_report": filter_report,
        "checkpoint_train_step": runtime_metadata["checkpoint_train_step"],
        "frame_stack": runtime_metadata["frame_stack"],
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": int(env_fps / obs_fps),
        "max_steps_unit": "raw_frame",
        "image_source": "env_raw_rgb"
        if image_transform_config.image_transform == "raw_rgb"
        else f"env_raw_rgb_{image_transform_config.image_transform}",
        "image_transform": asdict(image_transform_config),
        "image_resolution": split_stats["image_resolution"],
        "context_window": context_window,
        "context_image_count": max(0, context_window - 1),
        "replay_num_envs": replay_num_envs,
        "train_episodes": split_stats["train_episodes"],
        "val_episodes": split_stats["val_episodes"],
    }
    if "flat_cfg" in config and config["flat_cfg"].get("action_history_decisions") is not None:
        rollout_metadata["action_history_decisions"] = config["flat_cfg"]["action_history_decisions"]
    if "export_zero_action_history" in config:
        rollout_metadata["action_history_decisions"] = config["export_zero_action_history"]
    if ("flat_cfg" in config and config["flat_cfg"].get("latency_type") == "profile_sample"
            and rollout_metadata.get("action_history_decisions") == 8):
        rollout_metadata["profile_seed_by_reset"] = "env_reset_seed"
    if episode_return_gt is not None:
        rollout_metadata["episode_return_gate"] = filter_report["episode_return_gate"]
    metadata = {
        **config_metadata,
        **rollout_metadata,
        "env_name": config["env_name"],
        "prompt_source": (
            "sample_factory_checkpoint:gym_base_prompt"
            if config["integration_name"] == "gymnasium"
            else "sample_factory_checkpoint:base_prompt"
            if "action_history_decisions" in rollout_metadata
            else "latency_bench.data.prompts"
        ),
        "base_prompt": config["base_prompt"],
        "image_transform_prompt_contexts": IMAGE_TRANSFORM_CONTEXTS,
        "max_episodes": int(max_episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "num_envs": int(num_envs),
        "train_rows": split_stats["train_rows"],
        "val_rows": split_stats["val_rows"],
        "total_rows": split_stats["total_rows"],
        "rows_unit": "decision_step",
        "total_episodes": split_stats["total_episodes"],
    }
    if config["integration_name"] == "gymnasium":
        metadata["integration_name"] = "gymnasium"
    if (action_spec.layout != ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1
            and "action_history_decisions" not in rollout_metadata):
        metadata["prompt_timing_suffix"] = TIMING_SUFFIX
    if episode_acceptance is not None:
        metadata["episode_acceptance"] = dict(episode_acceptance)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(rollout_metadata["filter_report"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    cleanup_rollout_intermediates(output_dir)

    if push_to_hub:
        # Memory-mapped loading preserves embedded image features without a full RAM copy.
        push_kwargs = {"repo_id": hf_repo_id, "private": hf_private,
                       "max_shard_size": hf_max_shard_size}
        token = resolve_token(hf_token)
        if token is not None:
            push_kwargs["token"] = token
        if hf_config_name:
            push_kwargs["config_name"] = hf_config_name
        if hf_data_dir is not None:
            push_kwargs["data_dir"] = hf_data_dir
        load_parquet_splits(output_dir).push_to_hub(**push_kwargs)
    return deepcopy(metadata)


def _filter_report_from_probe_state(
    split_stats: Mapping[str, Any],
    probe_state: Mapping[str, Any],
    *,
    env_name: str,
    episode_return_field: str,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build final filter report from probe state and written row shards.

    Args:
        split_stats: Counts returned by the canonical Parquet writer.
        probe_state: Persisted probe state with accepted specs and attempt counters.
        env_name: Canonical benchmark environment name.
        filter_preset: Filter preset name recorded in the report.
        episode_return_field: Canonical episode-return field used by the acceptance gate.
        episode_return_gt: Optional strict lower bound for accepted episode returns.
        episode_acceptance: Explicit task-specific episode acceptance contract.

    Returns:
        Filter report written to metadata and ``filter_report.json``.
    """
    accepted_specs = probe_state["accepted_specs"]
    train_specs = [spec for spec in accepted_specs if spec["split"] == "train"]
    val_specs = [spec for spec in accepted_specs if spec["split"] == "val"]
    report = {
        "preset": filter_preset,
        "attempted_episodes": probe_state["attempted_episodes"],
        "accepted_episodes": len(accepted_specs),
        "rejected_episodes": probe_state["rejected_episodes"],
        "accepted_train_episodes": len(train_specs),
        "accepted_val_episodes": len(val_specs),
        "train_rows_before_row_filter": sum(spec["episode_length"] for spec in train_specs),
        "train_rows_after_row_filter": split_stats["train_rows"],
        "val_rows_before_row_filter": sum(spec["episode_length"] for spec in val_specs),
        "val_rows_after_row_filter": split_stats["val_rows"],
    }
    if episode_acceptance is not None:
        report["episode_acceptance"] = dict(episode_acceptance)
    if episode_return_gt is not None:
        report["episode_return_gate"] = {
            "field": episode_return_field,
            "operator": ">",
            "threshold": episode_return_gt,
        }
    if env_name == "demon_attack":
        report["raw_score_bins"] = demon_raw_score_bin_counts(accepted_specs)
    return report
