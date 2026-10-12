"""Select the existing Balance label protocol for public collection."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
from latency_bench.utils.io import write_yaml

from latency_bench.core.config import resolve_runtime_config
from latency_bench.envs.factory import resolve_env_config
from latency_bench.policy.model_source import model_file, resolve_model_source
from latency_bench.policy.action_prefix import resolve_action_prefix


def collect(config):
    config = resolve_model_source(config)
    resolve_runtime_config(config)
    resolve_env_config(config)
    options = config["collection"]
    output = Path(config["logging"]["output_dir"])
    teacher = (resolve_model_source({"policy": options["teacher"]})["policy"]
               if "teacher" in options else config["policy"])
    checkpoint = Path(model_file(teacher, "checkpoint"))
    teacher_config = Path(model_file(teacher, "model_config"))
    mode = options["label_mode"]
    if config["executor"]["mode"] != "simulated":
        raise ValueError("Balance label collection requires simulated execution")
    if mode in {"zero_full", "profile_conditioned"}:
        from latency_bench.envs.tasks.humanoidbench.profile_collection import INFERENCE_SEED, rollout_batch
        horizon = options["action_horizon"]
        if horizon not in (8, 16):
            raise ValueError("Balance profile labels require an action_horizon of 8 or 16")
        config["training"]["ppo"].update(chunk_horizon=horizon, active_heads=[0, horizon])
        config["policy"]["action_prefix"] = resolve_action_prefix(
            config["policy"]["action_prefix"], env_fps=config["env"]["env_fps"],
            action_horizon=horizon, action_dim=43)
        if (mode == "zero_full") != (config["policy"]["action_prefix"]["mode"] == "none"):
            raise ValueError("Balance label_mode and policy.action_prefix disagree")
        resources = config["executor"]
        student = "teacher" in options
        args = SimpleNamespace(
            smoke=options.get("smoke", False), teacher=checkpoint, teacher_config=teacher_config,
            teacher_revision=options["teacher_revision"],
            label_mode=mode, horizon=options["action_horizon"],
            collection_seed=config["experiment"]["seed"],
            rollout_workers=resources["rollout_workers"], inference_batch_size=resources["inference_batch_size"],
            rollout_overlap=resources["rollout_overlap"], num_gpus=len(resources["inference_devices"]),
            collect_bootstrap_only=True,
        )
        inference_config = deepcopy(config) if student else None
        if inference_config is not None:
            inference_config["experiment"]["seed"] = INFERENCE_SEED
        resident = options.get("resident")
        rollout_batch(args, config, output,
            bundle=Path(config["policy"]["root"]) if student else None,
            run=None, count=options["row_budget"], inference_config=inference_config,
            resident_control_dir=Path(resident["control_dir"]) if resident is not None else None,
            resident_step=resident["step"] if resident is not None else None)
    else:
        raise ValueError(f"Unsupported Balance collection.label_mode: {mode}")
    write_yaml(output / "resolved_config.yaml", config)
    return {"output_dir": str(output), "metadata": json.loads((output / "metadata.json").read_text())}
