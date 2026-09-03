#!/usr/bin/env python3
"""Train balance_simple from GR00T-N1.7 with periodic zero-delay eval."""

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
from gr00t.policy import Gr00tPolicy


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
) -> Path:
    command = [
        str(GROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=2",
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
        "2",
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
        "15000",
        "--global-batch-size",
        "128",
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "8",
        "--save-steps",
        "500",
        "--stop-every-steps",
        "500",
        "--save-total-limit",
        "2",
        "--use-wandb",
        "--experiment-name",
        run_id,
    ]
    if step > 500:
        command.append("--resume-from-checkpoint")

    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
    }
    subprocess.run(command, cwd=REPO, env=env, check=True)
    return output / f"checkpoint-{step}"


def validation_metrics(checkpoint: Path, val_dataset: Path) -> dict[str, float]:
    policy = Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=checkpoint,
        device="cuda:0",
    )
    loader = LeRobotEpisodeLoader(val_dataset, policy.get_modality_config())
    episode = loader[0]
    count = min(128, len(episode) - 39)
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
    step: int,
    *,
    output: Path,
    eval_config: Path,
) -> dict[str, float]:
    config = yaml.safe_load(eval_config.read_text())
    eval_output = output / "evals" / f"eval_step{step}"
    config["experiment"]["name"] = f"balance_simple_gr00t_step{step}"
    config["policy"]["model_path"] = str(checkpoint)
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
    parser.add_argument("--start-step", type=int, default=500)
    args = parser.parse_args()

    val_dataset = Path(f"{args.dataset_path}__val")

    for step in range(args.start_step, 15001, 500):
        checkpoint = train_to_next_checkpoint(
            base_model=args.base_model_path,
            dataset=args.dataset_path,
            output=args.output_dir,
            run_id=args.run_id,
            step=step,
        )
        metrics = validation_metrics(checkpoint, val_dataset)
        gc.collect()
        torch.cuda.empty_cache()
        metrics.update(
            closed_loop_metrics(
                checkpoint,
                step,
                output=args.output_dir,
                eval_config=EVAL_CONFIG,
            )
        )
        metrics["epochs"] = step * 128 / 86048
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


if __name__ == "__main__":
    main()
