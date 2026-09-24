"""Evaluate the selected QwenOFT H8 models and assemble the train/eval latency grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import yaml
from huggingface_hub import snapshot_download

from scripts.starvla.train_task import (
    ROOT, code_revisions, eval_config, evaluation_summary, mikasa_h8_execution_audit,
)


ZERO_REVISION = "5b533937a72e9fa923ecf28b1c1be5a16bad488d"
FIXED_TWO_REVISION = "795468e53947d26314e891cbc375775c6e0bd491"
FIXED_TWO_PATH = (
    "fixed-2/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/"
    "h8-conditioned-dagger-20260923"
)
EPISODES = 200
START_SEED = 4242424242


def selected_model(train_latency: int, model_dir: Path) -> tuple[Path, dict]:
    """Use the published Fixed-2 bundle or the completed DAgger run's best model."""
    if train_latency == 2:
        snapshot = snapshot_download(
            "latency-sensitive-bench/benchmark-models", revision=FIXED_TWO_REVISION,
            allow_patterns=[f"{FIXED_TWO_PATH}/**"], local_dir=model_dir,
        )
        return Path(snapshot) / FIXED_TWO_PATH, {
            "type": "benchmark_models", "repo_id": "latency-sensitive-bench/benchmark-models",
            "revision": FIXED_TWO_REVISION, "path": FIXED_TWO_PATH, "selected_step": 5000,
        }
    result = json.loads((model_dir / "result.json").read_text())
    run = json.loads((model_dir / "run.json").read_text())
    sft = json.loads((Path(run["source_run"]) / "run.json").read_text())
    assert result["dagger_steps"] == 5000
    assert sft["training"]["sft_updates"] == 5000
    step = result["best_step"]
    bundle = model_dir / "bundles" / ("best" if step else "source")
    return bundle, {
        "type": "local_dagger", "run": str(model_dir), "selected_step": step,
        "source_run": run["source_run"], "condition": run["condition"],
    }


def build_config(args, bundle: Path, output: Path) -> dict:
    runtime = SimpleNamespace(
        task="mikasa_conditioned_h8", model="qwenoft", condition=f"fixed-{args.train_latency}",
        eval_seed=START_SEED, inference_device="cuda:1", simulator_device="cuda:0",
        smoke=False, vla_python=args.vla_python, backbone_path=args.backbone_path,
    )
    return eval_config(runtime, bundle, Path("unused"), output,
                       eval_episodes=EPISODES, eval_latency_frames=args.eval_latency)


def summarize_cell(directory: Path, manifest: dict) -> dict:
    evaluation = evaluation_summary(directory)
    episodes = evaluation["episodes"]
    assert [episode["metadata"]["episode_seed"] for episode in episodes] == list(
        range(START_SEED, START_SEED + EPISODES)
    )
    assert [episode["episode_id"] for episode in episodes] == list(range(EPISODES))
    assert all(episode["survival_steps"] == 60 for episode in episodes)
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    by_episode = {episode["episode_id"]: [] for episode in episodes}
    for step in steps:
        by_episode[step["episode_id"]].append(step["env_step"])
    assert all(frames == list(range(60)) for frames in by_episode.values())
    config = yaml.safe_load((directory / "eval_config.yaml").read_text())
    assert config["policy"]["action_prefix"] == manifest["action_prefix"]
    actions = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    assert all(action["latency_raw_frames"] == manifest["eval_latency"] for action in actions)
    success_count = sum(episode["metadata"]["task_metrics"]["success"] for episode in episodes)
    return {
        "train_latency": manifest["train_latency"], "latency_frames": manifest["eval_latency"],
        "n_episodes": EPISODES, "seed_start": START_SEED, "seed_end": START_SEED + EPISODES - 1,
        "success_count": success_count, "success_rate": success_count / EPISODES,
        "mean_return": evaluation["mean_return"], "std_return": evaluation["std_return"],
        "artifact": {**manifest["artifact"], "bundle_path": manifest["bundle"]},
        "inference_code": manifest["inference_code"], "action_prefix": manifest["action_prefix"],
        "evidence_path": str(directory), "execution_audit": mikasa_h8_execution_audit(directory),
    }


def evaluate_cell(args) -> None:
    bundle, source = selected_model(args.train_latency, args.model_dir)
    output = args.output / f"train_{args.train_latency}" / f"eval_{args.eval_latency}"
    config = build_config(args, bundle, output)
    # These are artifact/protocol audits: the column must be the trained condition.
    assert config["policy"]["action_prefix"]["frames"] == args.train_latency
    if args.train_latency != 2:
        assert source["condition"] == f"fixed-{args.train_latency}"
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "requested_eval_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "train_latency": args.train_latency, "eval_latency": args.eval_latency,
        "artifact": source, "bundle": str(bundle),
        "action_prefix": config["policy"]["action_prefix"],
        "inference_code": code_revisions("mikasa_conditioned_h8"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.devices,
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "third_party/MIKASA-Robo"))),
        "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
    }
    with (output / "stdout.log").open("w") as log:
        subprocess.run(
            [str(args.task_python), str(ROOT / "scripts/mikasa/evaluate.py"),
             "latency-eval", "--eval-config", str(config_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    summary = summarize_cell(output, manifest)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def aggregate(zero_results: Path, output: Path) -> dict:
    baseline = json.loads(zero_results.read_text())["curves"]["qwenoft_h8"]
    assert baseline["artifact"]["revision"] == ZERO_REVISION
    cells = []
    for train_latency in range(5):
        artifacts = []
        for eval_latency in range(5):
            if train_latency == 0:
                cell = {**baseline["latencies"][str(eval_latency)],
                        "artifact": baseline["artifact"], "action_prefix": {"mode": "none"},
                        "train_latency": 0, "reused_from": str(zero_results)}
            else:
                directory = output / f"train_{train_latency}" / f"eval_{eval_latency}"
                manifest = json.loads((directory / "manifest.json").read_text())
                cell = summarize_cell(directory, manifest)
                assert cell == json.loads((directory / "summary.json").read_text())
                assert cell["action_prefix"]["frames"] == train_latency
            assert cell["train_latency"] == train_latency
            assert cell["latency_frames"] == eval_latency
            assert cell["n_episodes"] == EPISODES
            assert (cell["seed_start"], cell["seed_end"]) == (START_SEED, START_SEED + EPISODES - 1)
            assert cell["execution_audit"]["passed"]
            artifacts.append(cell["artifact"])
            cells.append(cell)
        assert all(artifact == artifacts[0] for artifact in artifacts)
    result = {
        "protocol": {
            "model": "qwenoft", "action_horizon": 8, "env_fps": 20, "obs_fps": 20,
            "episodes_per_cell": EPISODES, "seed_start": START_SEED,
            "seed_end": START_SEED + EPISODES - 1, "eval_max_steps": 60,
            "simulated_worker_capacity": 1, "eval_parallel_envs": 10,
            "inference_batch_size": 8, "prefix": "fixed by training contract",
            "columns": "train_latency", "rows": "eval_latency", "metric": "success_rate",
        },
        "cells": cells,
    }
    (output / "grid.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "train_latency", "eval_latency", "success_count", "episodes",
            "success_rate", "mean_return", "std_return",
        ])
        writer.writeheader()
        for cell in cells:
            writer.writerow({
                "train_latency": cell["train_latency"], "eval_latency": cell["latency_frames"],
                "success_count": cell["success_count"], "episodes": cell["n_episodes"],
                "success_rate": cell["success_rate"], "mean_return": cell["mean_return"],
                "std_return": cell["std_return"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--train-latency", type=int, choices=(1, 2, 3, 4), required=True)
    evaluate.add_argument("--eval-latency", type=int, choices=range(5), required=True)
    evaluate.add_argument("--model-dir", type=lambda value: Path(value).absolute(), required=True)
    evaluate.add_argument("--backbone-path", type=lambda value: Path(value).absolute(), required=True)
    evaluate.add_argument("--vla-python", type=lambda value: Path(value).absolute(), required=True)
    evaluate.add_argument("--task-python", type=lambda value: Path(value).absolute(), required=True)
    evaluate.add_argument("--devices", required=True, help="Two physical GPUs: simulator,inference")
    evaluate.add_argument("--output", type=lambda value: Path(value).absolute(), required=True)
    summarize = commands.add_parser("aggregate")
    summarize.add_argument("--zero-results", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "eval":
        evaluate_cell(args)
    else:
        aggregate(args.zero_results, args.output)


if __name__ == "__main__":
    main()
