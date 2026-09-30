"""Evaluate a task bundle with an explicit, frozen action-prefix protocol."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from latency_bench.core.config import load_config
from latency_bench.data.starvla_tasks import task_spec
from latency_bench.policy.action_prefix import resolve_action_prefix
from scripts.starvla import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("balance", "mikasa"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-mode", choices=("none", "profile_mean"), required=True)
    parser.add_argument("--prefix-profile", type=Path)
    parser.add_argument("--vla-python", type=Path, required=True)
    parser.add_argument("--task-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=820001)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    if args.prefix_mode == "profile_mean" and args.prefix_profile is None:
        parser.error("profile_mean requires --prefix-profile")
    if args.task == "balance":
        # MuJoCo chooses its GL implementation on import, before env construction.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    contract = json.loads((args.bundle / "task_contract.json").read_text())
    config = load_config(args.environment)
    settings = {"mode": args.prefix_mode}
    if args.prefix_mode == "profile_mean":
        settings["profile_path"] = str(args.prefix_profile)
    config["policy"]["action_prefix"] = resolve_action_prefix(
        settings, env_fps=config["env"]["env_fps"],
        action_horizon=contract["action_horizon"], action_dim=contract["action_dim"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "bundle": str(args.bundle), "task": args.task,
        "action_prefix": config["policy"]["action_prefix"],
        "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
    }
    if args.task == "balance" and config["executor"]["mode"] == "simulated" and contract["robot_type"] in (
        "latency_balance_profile_h8", "latency_balance_profile_h16"
    ):
        # This optional adapter owns Balance's augmented state and teacher wrapper.
        from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import (
            INFERENCE_SEED, TEACHER, TEACHER_CONFIG, policy_config, rollout_batch,
        )
        config = policy_config(config, args.bundle, args.vla_python, 8, args.backbone)
        config["training"]["ppo"]["chunk_horizon"] = contract["action_horizon"]
        config["training"]["ppo"]["active_heads"] = [0, contract["action_horizon"]]
        provenance.update(
            render_backend=os.environ["MUJOCO_GL"], inference_seed=INFERENCE_SEED,
            inference_batch_size=8, rollout_workers=16,
        )
        (args.output / "environment.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        rollout_args = SimpleNamespace(
            smoke=False, inference_batch_size=8, rollout_overlap=True, rollout_workers=16,
            num_gpus=1, vla_python=args.vla_python, backbone=args.backbone,
            teacher=TEACHER, teacher_config=TEACHER_CONFIG,
            horizon=contract["action_horizon"], collection_seed=0,
        )
        result = rollout_batch(rollout_args, config, args.output, bundle=args.bundle, run=None,
                               seeds=provenance["seeds"])
        summary = {key: result[key] for key in ("n", "mean_length", "mean_return", "ge100", "ge500")}
        summary.update(
            length_std=np.std([e["length"] for e in result["episodes"]]).item(),
            return_std=np.std([e["return"] for e in result["episodes"]]).item(),
            underflow_count=sum(e["underflow_count"] for e in result["episodes"]),
            underflow_frames=sum(e["underflow_frames"] for e in result["episodes"]),
        )
        if args.prefix_mode == "profile_mean":
            k = config["policy"]["action_prefix"]["frames"]
            delays = {"lt_k": 0, "eq_k": 0, "gt_k": 0}
            for episode in result["episodes"]:
                for delay, count in episode["action_ready_frame_counts"].items():
                    d = int(delay)  # Integer frame counts are JSON object keys.
                    delays["lt_k" if d < k else "eq_k" if d == k else "gt_k"] += count
            summary["delay_relative_to_k"] = delays
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    else:
        input_keys = runtime.TASK_INPUT_KEYS[task_spec(args.task).name]
        config["policy"].update(
            type="starvla", checkpoint_path=str(args.bundle / "checkpoints/model.pt"),
            model_config_path=str(args.bundle / "config.yaml"),
            task_contract_path=str(args.bundle / "task_contract.json"),
            worker_python_executable=str(args.vla_python), backbone_path=str(args.backbone),
            device="cuda:0",
            state_info_key=input_keys["state_info_key"],
            image_views_info_key=input_keys["image_views_info_key"],
        )
        config["evaluation"]["eval_episodes"] = args.episodes
        config["experiment"]["seed"] = args.seed_start
        config["logging"]["output_dir"] = str(args.output)
        config_path = args.output / "eval_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        provenance["render_configuration"] = config["env"]
        if args.task == "balance":
            provenance["render_backend"] = os.environ["MUJOCO_GL"]
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        runtime.run_command(
            runtime.evaluation_command(args, config_path),
            args.output / "stdout.log",
            env=runtime.environment(),
        )


if __name__ == "__main__":
    main()
