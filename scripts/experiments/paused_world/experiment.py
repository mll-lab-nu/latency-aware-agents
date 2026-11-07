"""Prepare and summarize the six fixed-checkpoint paused-world evaluations.

The Bash entrypoint owns installation, activation and sequential evaluation.
This module reuses the existing sim2real checkpoint registry and asset loader.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json, write_csv, write_json, write_yaml
from latency_bench.utils.paths import REPO_ROOT


MODEL_REVISION = "b17aad08bc4549fdbab5af96e698b3c637a2095a"
TASKS = ("flappy", "demon_attack")


def paused_config(base, output, *, episodes, parallel_envs, inference_devices, batch_size):
    """Keep the checkpoint's task/input contract and remove inference delay."""
    config = deepcopy(base)
    config["experiment"]["name"] += "-paused"
    config["latency"] = {"method": "zero"}
    config["executor"].update(
        mode="simulated", simulated_worker_capacity=1, inference_pool=True,
        parallel_envs=parallel_envs, inference_devices=inference_devices,
        inference_batch_size=batch_size,
    )
    config["policy"]["worker_python_executable"] = sys.executable
    config["evaluation"].update(episodes=episodes, suites=[])
    config["logging"].update(output_dir=str(output), wandb_project=None)
    return config


def prepare(args):
    from huggingface_hub import hf_hub_download

    from scripts.experiments.sim2real.assets import prepare_model_assets

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    registry = load_config(REPO_ROOT / "configs/experiments/sim2real.yaml")
    registry["model_revision"] = MODEL_REVISION
    manifest = {
        "model_repo": registry["model_repo"], "model_revision": MODEL_REVISION,
        "backbone": registry["backbone"],
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "condition": "paused_zero_delay",
        "comparison_condition": "archived_temporal_profile_simulation",
        "cases": [],
    }
    config_paths = []
    for task in TASKS:
        for model in registry["models"]:
            entry = registry["tasks"][task][model]
            case_dir = output / task / model
            case_dir.mkdir(parents=True)
            print(f"[paused-world] Preparing {task}/{model}: {entry['bundle']}", flush=True)
            base = prepare_model_assets(registry, entry)
            config = paused_config(
                base, case_dir, episodes=args.episodes, parallel_envs=args.parallel_envs,
                inference_devices=args.inference_devices, batch_size=args.inference_batch_size,
            )
            config_path = case_dir / "eval_config.yaml"
            write_yaml(config_path, config)
            config_paths.append(str(config_path))
            profile_summary_path = f"{entry['bundle']}/evaluations/profile/summary.json"
            archived = hf_hub_download(
                repo_id=registry["model_repo"], revision=MODEL_REVISION,
                filename=profile_summary_path,
            )
            shutil.copyfile(archived, case_dir / "archived_profile_summary.json")
            manifest["cases"].append({
                "task": task, "model": model, "bundle": entry["bundle"],
                "checkpoint": entry["checkpoint"],
                "config": str(config_path.relative_to(output)),
                "archived_profile_summary": profile_summary_path,
            })
    write_json(output / "experiment.json", manifest)
    (output / "eval_configs.txt").write_text("\n".join(config_paths) + "\n")
    print(f"[paused-world] All six checkpoints and configs ready: {output}", flush=True)


def summarize(output):
    manifest = read_json(output / "experiment.json")
    rows = []
    for case in manifest["cases"]:
        config_path = output / case["config"]
        config = load_config(config_path)
        case_dir = config_path.parent
        records = [json.loads(line) for line in (case_dir / "episode_metrics.jsonl").read_text().splitlines()]
        expected_seeds = list(range(
            config["experiment"]["seed"], config["experiment"]["seed"] + config["evaluation"]["episodes"],
        ))
        # Results are an external boundary: never report a partial/wrong-condition run as complete.
        seeds = sorted(record["metadata"]["episode_seed"] for record in records)
        if seeds != expected_seeds:
            raise ValueError(f"Incomplete or mismatched episode cohort: {case_dir}")
        if any(record["mean_latency_ms"] != 0 for record in records):
            raise ValueError(f"Nonzero inference delay in paused evaluation: {case_dir}")
        if any(record["survival_steps"] > config["evaluation"]["max_env_steps"] for record in records):
            raise ValueError(f"Episode exceeds the configured raw-frame cap: {case_dir}")
        returns = np.asarray([record["episode_return"] for record in records])
        bootstrap = np.random.default_rng(42).choice(returns, size=(10000, len(returns))).mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        archived = read_json(case_dir / "archived_profile_summary.json")
        rows.append({
            "task": case["task"], "model": case["model"], "episodes": len(records),
            "paused_mean": float(returns.mean()), "paused_std": float(returns.std()),
            "paused_ci95_low": float(low), "paused_ci95_high": float(high),
            "archived_profile_mean": archived["mean_return"],
            "archived_profile_std": archived["std_return"],
            "archived_profile_episodes": len(archived["returns"]),
            "mean_raw_frames": sum(record["survival_steps"] for record in records) / len(records),
            "max_raw_frames": config["evaluation"]["max_env_steps"],
            "seed_start": expected_seeds[0], "seed_end": expected_seeds[-1],
            "env_fps": config["env"]["env_fps"], "obs_fps": config["env"]["obs_fps"],
            "checkpoint_bundle": case["bundle"],
        })
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        for condition in ("paused", "archived_profile"):
            for row in task_rows:
                row[f"{condition}_rank"] = 1 + sum(
                    other[f"{condition}_mean"] > row[f"{condition}_mean"] for other in task_rows
                )
    write_json(output / "results.json", {
        "experiment": manifest, "status": "complete",
        "uncertainty": "Population SD; 95% percentile bootstrap CI of episode mean, 10000 resamples, seed 42."
                       " One training seed per checkpoint; ranks use point estimates.",
        "results": rows,
    })
    write_csv(output / "results.csv", rows)
    print("\nScores: mean +/- episode SD. Profile scores are archived simulation results.")
    print(f"{'Task':<14} {'Model':<12} {'Episodes':>8} {'Paused':>22} "
          f"{'Archived profile':>22} {'Rank paused/profile':>20}")
    for row in rows:
        paused = f"{row['paused_mean']:.2f} +/- {row['paused_std']:.2f}"
        profile = f"{row['archived_profile_mean']:.2f} +/- {row['archived_profile_std']:.2f}"
        print(f"{row['task']:<14} {row['model']:<12} {row['episodes']:>8} {paused:>22} {profile:>22} "
              f"{row['paused_rank']}/{row['archived_profile_rank']}")
    print(f"\nResults: {output / 'results.csv'}\nFull evidence: {output}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Download artifacts and write six evaluation configs")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--episodes", type=int, default=100)
    prepare_parser.add_argument("--parallel-envs", type=int, default=32)
    prepare_parser.add_argument("--inference-batch-size", type=int, default=16)
    prepare_parser.add_argument("--inference-devices", nargs="+", default=["cuda:0", "cuda:1"])
    summary_parser = commands.add_parser("summarize", help="Summarize completed runs without downloading or evaluating")
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        summarize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
