"""Adapt Sample Factory teachers to the shared collection and export pipeline."""

from copy import deepcopy
from pathlib import Path

from latency_bench.core.config import resolve_runtime_config
from latency_bench.data.action_layouts import (
    ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1,
    ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1,
)
from latency_bench.data.filters import FILTER_PRESET_NONE
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.data.prompts import IMAGE_TRANSFORM_CONTEXTS, TIMING_SUFFIX
from latency_bench.data.rollout_config import resolve_collection_options
from latency_bench.data.rollout_export import finalize_dataset
from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config
from latency_bench.integrations.sample_factory.teacher_rollout import collect_teacher_dataset, collection_task
from latency_bench.policy.model_source import model_file, resolve_model_source
from latency_bench.utils.io import write_yaml
from training.common.checkpoints import find_checkpoint


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
    if policy["checkpoint"] is None and "checkpoint_kind" in policy:
        root = Path(policy["root"])
        checkpoint = find_checkpoint(root.parent, root.name, kind=policy["checkpoint_kind"])
        policy["checkpoint"] = str(checkpoint.relative_to(root))
        config["provenance"]["model_artifact"]["checkpoint"] = policy["checkpoint"]
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
    if "export_zero_action_history" in collection:
        descriptor["export_zero_action_history"] = collection["export_zero_action_history"]
    action_spec = descriptor["action_spec"]
    if action_spec.layout in (ACTION_LAYOUT_GYMNASIUM_CONTINUOUS_V1, ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1):
        collection["filter_preset"] = options["filter_preset"] = FILTER_PRESET_NONE
    image_transform = GhostTrailConfig(**collection.get("image_transform", {}))
    options.update(
        max_episodes=config["evaluation"]["episodes"],
        max_steps_per_episode=config["evaluation"]["max_env_steps"],
        num_envs=config["executor"]["parallel_envs"],
        deterministic=policy["deterministic"],
        image_transform_config=image_transform,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "resolved_config.yaml", config)
    probe_state = collect_teacher_dataset(descriptor, output_dir=output_dir, **options)
    if options["probe_only"]:
        return {"output_dir": str(output_dir), "probe_state": probe_state}
    task = collection_task(descriptor)
    source.update(
        prompt_source=(
            "sample_factory_checkpoint:gym_base_prompt" if descriptor["integration_name"] == "gymnasium"
            else "sample_factory_checkpoint:base_prompt" if task["prompt_override"] is not None
            else "latency_bench.data.prompts"
        ),
        image_transform_prompt_contexts=IMAGE_TRANSFORM_CONTEXTS,
    )
    if flat.get("action_history_decisions") is not None:
        source["action_history_decisions"] = flat["action_history_decisions"]
    if "export_zero_action_history" in descriptor:
        source["action_history_decisions"] = descriptor["export_zero_action_history"]
    if flat.get("latency_type") == "profile_sample" and source.get("action_history_decisions") == 8:
        source["profile_seed_by_reset"] = "env_reset_seed"
    if action_spec.layout != ACTION_LAYOUT_G1_SONIC_ABSOLUTE_JOINT_TARGET_V1 and "action_history_decisions" not in source:
        source["prompt_timing_suffix"] = TIMING_SUFFIX
    metadata = finalize_dataset(
        output_dir=output_dir, task=task, options=options,
        probe_state=probe_state, source_metadata=source,
    )
    return {"output_dir": str(output_dir), "metadata": metadata}
