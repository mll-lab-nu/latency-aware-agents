"""Resolve public collection settings and the existing canonical task contract."""

from latency_bench.core.config import resolve_runtime_config
from copy import deepcopy
from pathlib import Path

from latency_bench.core.clock import validate_clock_rates
from latency_bench.envs.factory import resolve_env_config, environment_integration
from latency_bench.core.actions import ActionResolver
from latency_bench.data.action_layouts import (
    action_spec_for_env, gymnasium_discrete_action_spec, gymnasium_continuous_action_spec,
)
from latency_bench.data.ghost_trail import GhostTrailConfig
from latency_bench.data.prompts import BASE_PROMPTS
from latency_bench.eval.config import _episode_seed
from latency_bench.policy.model_source import resolve_model_source


def resolve_collection(config):
    config = resolve_model_source(config)
    if config["executor"]["mode"] not in {"paused", "simulated"}:
        raise ValueError("Teacher collection requires executor.mode=paused or simulated")
    resolve_runtime_config(config)
    resolve_env_config(config)
    env = config["env"]
    validate_clock_rates(env["env_fps"], env["obs_fps"])
    collection = config["collection"]
    options = resolve_collection_options(config)
    student = collection["student_observation"]
    for key, value in {"image_key": None, "state_key": None, "state_source": "metadata"}.items():
        if key not in student:
            student[key] = value
    if student["state_source"] not in {"data", "metadata"}:
        raise ValueError("student_observation.state_source must be data or metadata")
    output_dir = Path(config["logging"]["output_dir"]).expanduser().absolute()
    config["logging"]["output_dir"] = str(output_dir)
    name = env["name"]
    if environment_integration(config) == "gymnasium":
        if "action_space" in env and env["action_space"]["type"] == "box":
            spec = gymnasium_continuous_action_spec(env["action_space"]["labels"])
        else:
            spec = gymnasium_discrete_action_spec(env["action_order"])
        integration = "gymnasium"
        base_prompt = env["base_prompt"]
        task_metadata = {
            "task_name": name, "base_prompt": base_prompt,
            "env_fps": env["env_fps"], "obs_fps": env["obs_fps"],
            "frame_stack": env["frame_stack"] if "frame_stack" in env else 1,
            **{key: env[key] for key in ("env_id", "registration_imports", "make_kwargs") if key in env},
        }
        if "action_space" in env and env["action_space"]["type"] == "box":
            task_metadata.update(action_space=env["action_space"], noop_action=env["noop_action"])
        else:
            task_metadata.update(action_labels=list(spec.labels),
                                 action_values=[env["action_map"][key] for key in env["action_order"]],
                                 noop_action_id=[env["action_map"][key] for key in env["action_order"]].index(ActionResolver(env).noop_action.value))
        if student["state_key"] is not None:
            task_metadata["state_space"] = env["state_space"]
        source = {"integration_name": integration, "gymnasium_task": task_metadata}
    else:
        spec = action_spec_for_env(name)
        integration = name
        base_prompt = env["base_prompt"] if "base_prompt" in env else BASE_PROMPTS[name]
        source = {}
    task = dict(env_name=name, integration_name=integration, action_spec=spec,
                base_prompt=base_prompt, prompt_override=base_prompt, images_pretransformed=False,
                action_values=source["gymnasium_task"]["action_values"]
                if integration == "gymnasium" and "action_id" in spec.row_fields else None,
                image_source="env_raw_rgb" if student["image_key"] is None else f"observation.data.{student['image_key']}")
    source.update(teacher_source=deepcopy(config["policy"]), student_observation=student,
                  latency=deepcopy(config["latency"]) if config["executor"]["mode"] == "simulated" else {"method": "zero"}, scheduler=deepcopy(config["scheduler"]),
                  prompt_source="task:base_prompt")
    options.update(max_episodes=config["evaluation"]["episodes"],
                   max_steps_per_episode=config["evaluation"]["max_env_steps"],
                   num_envs=config["executor"]["parallel_envs"], base_seed=_episode_seed(config, 0),
                   image_transform_config=GhostTrailConfig(**collection.get("image_transform", {})))
    if options["probe_only"] and not options["two_pass"]:
        raise ValueError("probe_only requires two_pass collection")
    return config, task, options, source


def resolve_collection_options(config):
    """Resolve shared collection defaults once for generic and native execution."""
    collection = config["collection"]
    defaults = {
        "val_fraction": 0.0, "split_seed": 0, "max_attempts": None,
        "two_pass": False, "replay_num_envs": config["executor"]["parallel_envs"],
        "filter_preset": "none", "episode_return_gt": None, "episode_acceptance": None,
        "demon_episode_threshold": None, "context_window": 1, "probe_only": False,
        "export_image_size": None, "image_writer_workers": 0, "row_buffer_size": 256,
    }
    for key, value in defaults.items():
        if key not in collection:
            collection[key] = value
    return {key: collection[key] for key in defaults}
