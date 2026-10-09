"""End-to-end export orchestration for Sample Factory teacher datasets."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from latency_bench.data.hf_parquet import load_parquet_splits
from latency_bench.data.rollout_export import finalize_dataset

from latency_bench.data.prompts import BASE_PROMPTS, IMAGE_TRANSFORM_CONTEXTS, TIMING_SUFFIX
from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1,
    action_spec_for_env,
)
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, FILTER_PRESET_NONE
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config
from latency_bench.integrations.sample_factory.teacher_rollout import collect_teacher_dataset, collection_task
from latency_bench.utils.hf import resolve_token
from latency_bench.utils.io import write_json_atomic


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
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "resolved_config.yaml", {
        **config, "action_spec": asdict(action_spec),
        "collection": dict(max_episodes=max_episodes, max_attempts=max_attempts,
                           max_steps_per_episode=max_steps_per_episode, val_fraction=val_fraction,
                           split_seed=split_seed, num_envs=num_envs, replay_num_envs=replay_num_envs,
                           deterministic=deterministic, filter_preset=filter_preset,
                           episode_return_gt=episode_return_gt, episode_acceptance=episode_acceptance,
                           demon_episode_threshold=demon_episode_threshold,
                           export_image_size=export_image_size, image_writer_workers=image_writer_workers,
                           row_buffer_size=row_buffer_size, context_window=context_window,
                           rollout_seed_offset=rollout_seed_offset, probe_only=probe_only,
                           image_transform=asdict(GhostTrailConfig() if image_transform_config is None else image_transform_config)),
    })
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

    image_transform_config = GhostTrailConfig() if image_transform_config is None else image_transform_config
    task = collection_task(config)
    source_metadata = {
        **config_metadata,
        "prompt_source": (
            "sample_factory_checkpoint:gym_base_prompt" if config["integration_name"] == "gymnasium"
            else "sample_factory_checkpoint:base_prompt" if task["prompt_override"] is not None
            else "latency_bench.data.prompts"
        ),
        "image_transform_prompt_contexts": IMAGE_TRANSFORM_CONTEXTS,
    }
    flat = config["flat_cfg"]
    if flat.get("action_history_decisions") is not None:
        source_metadata["action_history_decisions"] = flat["action_history_decisions"]
    if "export_zero_action_history" in config:
        source_metadata["action_history_decisions"] = config["export_zero_action_history"]
    if flat.get("latency_type") == "profile_sample" and source_metadata.get("action_history_decisions") == 8:
        source_metadata["profile_seed_by_reset"] = "env_reset_seed"
    if action_spec.layout != ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1 and "action_history_decisions" not in source_metadata:
        source_metadata["prompt_timing_suffix"] = TIMING_SUFFIX
    metadata = finalize_dataset(
        output_dir=output_dir, task=task, probe_state=probe_state, source_metadata=source_metadata,
        options=dict(max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode,
                     val_fraction=val_fraction, split_seed=split_seed, num_envs=num_envs,
                     replay_num_envs=replay_num_envs, context_window=context_window,
                     filter_preset=filter_preset, episode_return_gt=episode_return_gt,
                     episode_acceptance=episode_acceptance, demon_episode_threshold=demon_episode_threshold,
                     image_transform_config=image_transform_config),
    )

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
