"""End-to-end export orchestration for Sample Factory teacher datasets."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from latency_bench.utils.io import write_yaml

from latency_bench.core.config import resolve_runtime_config
from latency_bench.data.rollout_config import resolve_collection_options
from latency_bench.policy.model_source import model_file, resolve_model_source
from latency_bench.data.rollout_probe import load_probe_state

from latency_bench.data.rollout_export import finalize_dataset

from latency_bench.data.prompts import IMAGE_TRANSFORM_CONTEXTS, TIMING_SUFFIX
from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1,
)
from latency_bench.data.filters import FILTER_PRESET_CLEAN_V1, FILTER_PRESET_NONE
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config
from latency_bench.integrations.sample_factory.teacher_rollout import collect_teacher_dataset, collection_task
from latency_bench.utils.io import write_json_atomic


def _export_dataset(
    config, config_metadata, *,
    output_dir: Path,
    max_episodes: int,
    max_attempts: int | None = None,
    max_steps_per_episode: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    replay_num_envs: int = 16,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    filter_preset: str = FILTER_PRESET_CLEAN_V1,
    demon_episode_threshold: float | None = None,
    episode_return_gt: float | None = None,
    episode_acceptance: Mapping[str, Any] | None = None,
    context_window: int = 1,
    rollout_seed_offset: int = 0,
    probe_only: bool = False,
    image_transform_config: GhostTrailConfig | None = None,
    export_zero_action_history: int | None = None,
    two_pass: bool = True,
) -> dict[str, Any]:
    """Finalize the native SF adapter through the shared collection pipeline."""
    if export_zero_action_history is not None:
        config["export_zero_action_history"] = export_zero_action_history
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
        two_pass=two_pass,
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

    return deepcopy(metadata)


def collect(config):
    """Adapt the public run configuration to native SF collection."""

    config = resolve_model_source(config)
    resolve_runtime_config(config)
    if config["executor"]["mode"] not in {"paused", "simulated"}:
        raise ValueError("Teacher collection requires executor.mode=paused or simulated")
    policy = config["policy"]
    env = config["env"]
    options = resolve_collection_options(config)
    collection = config["collection"]
    if options["probe_only"] and not options["two_pass"]:
        raise ValueError("probe_only requires two_pass collection")
    if config["executor"]["inference_pool"]:
        raise ValueError("Native Sample Factory collection does not use an inference pool")
    devices = config["executor"]["inference_devices"]
    if len(devices) != 1:
        raise ValueError("Native Sample Factory collection uses one inference device")
    descriptor, source = load_rollout_training_config(
        model_file(policy, "checkpoint") if policy["checkpoint"] is not None else policy["root"],
        device_override=devices[0], gym_make_kwargs_override=env.get("make_kwargs"),
    )
    if descriptor["env_name"] != env["name"]:
        raise ValueError("env.name must match the teacher checkpoint task")
    flat = descriptor["flat_cfg"]
    flat["seed"] = config["experiment"]["seed"]
    for key in ("env_fps", "obs_fps", "simulator", "simulator_device"):
        if key in env:
            flat[key] = env[key]
    flat.update(config["scheduler"])
    descriptor["runtime_overrides"] = {
        "seed": config["experiment"]["seed"], **config["scheduler"],
        **{key: env[key] for key in ("env_fps", "obs_fps", "simulator", "simulator_device") if key in env},
        "simulated_worker_capacity": config["executor"]["simulated_worker_capacity"],
    }
    descriptor["latency_override"] = (deepcopy(config["latency"])
        if config["executor"]["mode"] == "simulated" else {"method": "zero"})
    source["latency_override"] = deepcopy(descriptor["latency_override"])
    source["latency_source"] = "configuration"
    output_dir = Path(config["logging"]["output_dir"]).expanduser().absolute()
    metadata = _export_dataset(
        descriptor, source, output_dir=output_dir,
        max_episodes=config["evaluation"]["episodes"],
        max_steps_per_episode=config["evaluation"]["max_env_steps"],
        num_envs=config["executor"]["parallel_envs"],
        deterministic=policy["deterministic"],
        export_zero_action_history=collection.get("export_zero_action_history"),
        image_transform_config=GhostTrailConfig(**collection.get("image_transform", {})),
        **options,
    )
    write_yaml(output_dir / "resolved_config.yaml", config)
    if options["probe_only"]:
        return {"output_dir": str(output_dir), "probe_state": load_probe_state(output_dir, completed_episode_indices=set())}
    return {"output_dir": str(output_dir), "metadata": metadata}
