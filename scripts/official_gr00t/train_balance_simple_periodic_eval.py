#!/usr/bin/env python3
"""Train balance_simple from GR00T-N1.7 with periodic profile eval."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import wandb
import yaml

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.stats import generate_stats
from gr00t.policy import Gr00tPolicy
from latency_bench.core.config import load_config


REPO = Path(__file__).resolve().parents[2]
GROOT = REPO / "third_party/Isaac-GR00T"
EVAL_CONFIG = (
    REPO
    / "configs/examples/gymnasium/g1_sonic_balance_simple/"
    "official_gr00t_eval_simulated.yaml"
)


def train_to_next_checkpoint(
    *,
    base_model: Path,
    dataset: Path,
    output: Path,
    run_id: str,
    step: int,
    max_steps: int = 15000,
    step_interval: int = 500,
    resume: bool = False,
    num_gpus: int = 2,
    global_batch_size: int = 128,
) -> Path:
    command = [
        str(GROOT / ".venv/bin/torchrun"),
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(GROOT / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(base_model),
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(REPO / "latency_bench/policy/official_gr00t_modality.py"),
        "--num-gpus",
        str(num_gpus),
        "--output-dir",
        str(output),
        "--max-state-dim",
        "198",
        "--action-horizon",
        "40",
        "--repeat-current-action",
        "--fixed-action-noise-seed",
        "0",
        "--one-step-action-regression",
        "--action-head-dropout",
        "0",
        "--state-dropout-prob",
        "0",
        "--tune-visual",
        "--load-bf16",
        "--no-backbone-trainable-params-fp32",
        "--optim",
        "adamw_torch_fused",
        "--learning-rate",
        "0.0001",
        "--warmup-ratio",
        "0.016666666666666666",
        "--max-steps",
        str(max_steps),
        "--global-batch-size",
        str(global_batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "8",
        "--save-steps",
        str(step_interval),
        "--stop-every-steps",
        str(step_interval),
        "--save-total-limit",
        "2",
        "--use-wandb",
        "--experiment-name",
        run_id,
    ]
    if resume:
        command.append("--resume-from-checkpoint")

    env = os.environ | {
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
    }
    subprocess.run(command, cwd=REPO, env=env, check=True)
    return output / run_id / f"checkpoint-{step}"


def validation_metrics(checkpoint: Path, val_dataset: Path) -> dict[str, float]:
    generate_stats(val_dataset)
    policy = Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=checkpoint,
        device="cuda:0",
    )
    loader = LeRobotEpisodeLoader(val_dataset, policy.get_modality_config())
    episode = loader[0]
    count = min(128, len(episode))
    states = np.stack(episode["state.state"].iloc[:count]).astype(np.float32)[:, None]
    videos = np.stack([np.asarray(x) for x in episode["video.ego_view"].iloc[:count]])[
        :, None
    ]
    language = [
        [x] for x in episode["language.annotation.human.task_description"].iloc[:count]
    ]
    targets = np.stack(episode["action.action"].iloc[:count]).astype(np.float32)

    def predict(limit: int) -> np.ndarray:
        chunks = []
        for start in range(0, limit, 8):
            end = min(start + 8, limit)
            action, _ = policy.get_action(
                {
                    "video": {"ego_view": videos[start:end]},
                    "state": {"state": states[start:end]},
                    "language": {
                        "annotation.human.task_description": language[start:end]
                    },
                }
            )
            chunks.append(action["action"][:, 0])
        return np.concatenate(chunks)

    predictions = predict(count)
    repeated = np.stack([predict(8) for _ in range(4)])
    metrics: dict[str, float] = {}
    for name, width in (("body", 29), ("all", 43)):
        target = targets[:, :width].reshape(-1)
        prediction = predictions[:, :width].reshape(-1)
        metrics[f"{name}_mae"] = float(np.mean(np.abs(prediction - target)))
        metrics[f"{name}_slope"] = float(
            np.cov(target, prediction, bias=True)[0, 1] / np.var(target)
        )
    metrics["repeat_std_max"] = float(repeated.std(axis=0).max())
    return metrics


def closed_loop_metrics(
    checkpoint: Path,
    step: int | str,
    *,
    output: Path,
    eval_config: Path,
    latency_profile_path: Path,
    latency_config_path: Path | None = None,
    latency_method: str = "temporal",
) -> dict[str, float]:
    config = load_config(eval_config)
    if latency_config_path is not None:
        latency_config = load_config(latency_config_path)
        for section in ("executor", "latency", "scheduler"):
            config[section].update(latency_config[section])
    eval_output = output / "evals" / f"eval_step{step}_{latency_method}"
    config["experiment"]["name"] = (
        f"balance_simple_gr00t_step{step}_{latency_method}"
    )
    config["policy"]["model_path"] = str(checkpoint)
    config["latency"]["method"] = latency_method
    if latency_method == "temporal" and latency_config_path is None:
        config["latency"]["profile_path"] = str(latency_profile_path)
    config["logging"]["output_dir"] = str(eval_output)
    config["logging"]["video"] = {"enabled": True, "num_bins": 1}

    with tempfile.TemporaryDirectory(prefix="gr00t-periodic-eval-") as temp_dir:
        config_path = Path(temp_dir) / "eval.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        }
        subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "latency",
                "python",
                "-m",
                "latency_bench.run",
                "--config",
                str(config_path),
            ],
            cwd=REPO,
            env=env,
            check=True,
        )

    episodes = [json.loads(line) for line in (eval_output / "episode_metrics.jsonl").read_text().splitlines()]
    returns = np.asarray([x["episode_return"] for x in episodes])
    lengths = np.asarray([x["survival_steps"] for x in episodes])
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(lengths.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-model-path", type=Path)
    parser.add_argument(
        "--latency-profile-path",
        type=Path,
        required=True,
        help="Temporal latency profile used by every periodic closed-loop evaluation.",
    )
    parser.add_argument(
        "--latency-config",
        type=Path,
        help="Resolved training latency config copied into periodic evaluations.",
    )
    parser.add_argument("--start-step", type=int, default=500)
    parser.add_argument("--max-step", type=int, default=15000)
    parser.add_argument("--step-interval", type=int, default=500)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=128)
    args = parser.parse_args()

    val_dataset = Path(f"{args.dataset_path}__val")

    for step in range(args.start_step, args.max_step + 1, args.step_interval):
        checkpoint = train_to_next_checkpoint(
            base_model=args.base_model_path,
            dataset=args.dataset_path,
            output=args.output_dir,
            run_id=args.run_id,
            step=step,
            max_steps=args.max_step,
            step_interval=args.step_interval,
            resume=step > args.step_interval,
            num_gpus=args.num_gpus,
            global_batch_size=args.global_batch_size,
        )
        metrics = validation_metrics(checkpoint, val_dataset)
        gc.collect()
        torch.cuda.empty_cache()
        for latency_name, latency_method in (
            ("zero", "zero"),
            ("profile", "temporal"),
        ):
            metrics.update(
                {
                    f"new_{latency_name}_{key}": value
                    for key, value in closed_loop_metrics(
                        checkpoint,
                        step,
                        output=args.output_dir,
                        eval_config=EVAL_CONFIG,
                        latency_profile_path=args.latency_profile_path,
                        latency_config_path=args.latency_config,
                        latency_method=latency_method,
                    ).items()
                }
            )
        print(json.dumps({"step": step, **metrics}, indent=2), flush=True)

        run = wandb.init(
            project="finetune-gr00t-n1d7",
            id=args.run_id,
            name=args.run_id,
            resume="allow",
        )
        run.define_metric("periodic_eval/*", step_metric="train/global_step")
        run.log(
            {
                "train/global_step": step,
                **{
                    f"periodic_eval/{key}": value
                    for key, value in metrics.items()
                },
            }
        )
        run.finish()
        wandb.teardown()

    if args.baseline_model_path is not None:
        baseline_metrics = {}
        for latency_name, latency_method in (
            ("zero", "zero"),
            ("profile", "temporal"),
        ):
            baseline_metrics.update(
                {
                    f"original_{latency_name}_{key}": value
                    for key, value in closed_loop_metrics(
                        args.baseline_model_path,
                        "baseline",
                        output=args.output_dir,
                        eval_config=EVAL_CONFIG,
                        latency_profile_path=args.latency_profile_path,
                        latency_config_path=args.latency_config,
                        latency_method=latency_method,
                    ).items()
                }
            )
        print(json.dumps({"checkpoint": "original", **baseline_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
