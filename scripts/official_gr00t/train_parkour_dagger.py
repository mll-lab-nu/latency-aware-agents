#!/usr/bin/env python3
"""Run the ten-round Extreme Parkour VLA DAgger pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PARKOUR_ROOT = REPO_ROOT / "third_party/Isaaclab_Parkour"
GROOT_ROOT = REPO_ROOT / "third_party/Isaac-GR00T"
PARKOUR_SCRIPT = PARKOUR_ROOT / "scripts/rsl_rl/parkour_vla.py"
PREPARE_SCRIPT = REPO_ROOT / "scripts/official_gr00t/prepare_parkour_dataset.py"
FINETUNE_SCRIPT = GROOT_ROOT / "gr00t/experiment/launch_finetune.py"
MODALITY_CONFIG = REPO_ROOT / "latency_bench/policy/official_gr00t_parkour_modality.py"

DAGGER_ROUNDS = 10
UPDATES_PER_ROUND = 500
TOTAL_UPDATES = DAGGER_ROUNDS * UPDATES_PER_ROUND
GLOBAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 36
COLLECTOR_RANKS = 2
ENVS_PER_COLLECTOR_RANK = 96
PARKOUR_DAGGER_ROWS_PER_ROUND = 2_304_000

DEFAULT_TEACHER_CHECKPOINT = PARKOUR_ROOT / (
    "logs/rsl_rl/unitree_go2_parkour/"
    "2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k/"
    "model_14999.pt"
)
DEFAULT_BOOTSTRAP_CHECKPOINT = REPO_ROOT / (
    "outputs/official_gr00t/extreme_parkour_go2_latent_h1_5k/checkpoint-5000"
)
DEFAULT_BOOTSTRAP_DATASET = REPO_ROOT / (
    "outputs/official_gr00t/extreme_parkour_go2/lerobot"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/official_gr00t/extreme_parkour_go2_dagger"
DEFAULT_POLICY_CONFIG = REPO_ROOT / (
    "configs/examples/extreme_parkour/official_gr00t_eval.yaml"
)
DEFAULT_PARKOUR_PYTHON = Path("/pfs/users/lxy/conda/envs/extreme-parkour/bin/python")


def checkpoint_dir(training_output: Path, completed_rounds: int) -> Path:
    return training_output / f"checkpoint-{completed_rounds * UPDATES_PER_ROUND}"


def write_policy_config(
    source: Path,
    destination: Path,
    *,
    model_path: Path,
    inference_device: str,
) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["policy"]["model_path"] = str(model_path)
    config["policy"]["device"] = inference_device
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def collector_command(
    parkour_python: Path,
    *,
    mode: str,
    teacher_checkpoint: Path,
    policy_config: Path,
    output_dir: Path,
    inference_device: str,
    num_envs: int,
    seed: int,
    round_index: int,
    student_actor_checkpoint: Path | None,
    dagger_row_budget: int = PARKOUR_DAGGER_ROWS_PER_ROUND,
) -> list[str]:
    command = [
        str(parkour_python),
        str(PARKOUR_SCRIPT),
        mode,
        "--headless",
        "--device",
        "cuda:0",
        "--checkpoint",
        str(teacher_checkpoint),
        "--policy_config",
        str(policy_config),
        "--output_dir",
        str(output_dir),
        "--inference_device",
        inference_device,
        "--inference_batch_size",
        "8",
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--dagger_round",
        str(round_index),
    ]
    if mode == "dagger-collect":
        command.extend(["--dagger_row_budget", str(dagger_row_budget)])
    if student_actor_checkpoint is not None:
        command.extend(
            ["--student_actor_checkpoint", str(student_actor_checkpoint)]
        )
    return command


def prepare_command(
    raw_dir: Path,
    dataset_dir: Path,
    *,
    round_index: int,
    normalization_source: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PREPARE_SCRIPT),
        str(raw_dir),
        str(dataset_dir),
        "--dagger",
        "--round-index",
        str(round_index),
        "--expected-rows",
        str(PARKOUR_DAGGER_ROWS_PER_ROUND),
        "--normalization-source",
        str(normalization_source),
    ]


def training_command(
    *,
    bootstrap_checkpoint: Path,
    teacher_checkpoint: Path,
    dataset_dir: Path,
    training_output: Path,
    round_index: int,
) -> list[str]:
    command = [
        str(GROOT_ROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=2",
        str(FINETUNE_SCRIPT),
        "--base-model-path",
        str(bootstrap_checkpoint),
        "--dataset-path",
        str(dataset_dir),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(MODALITY_CONFIG),
        "--num-gpus",
        "2",
        "--output-dir",
        str(training_output),
        "--max-state-dim",
        "53",
        "--parkour-dagger",
        "--parkour-actor-checkpoint",
        str(teacher_checkpoint),
        "--action-horizon",
        "1",
        "--fixed-action-noise-seed",
        "0",
        "--one-step-action-regression",
        "--action-head-dropout",
        "0",
        "--state-dropout-prob",
        "0",
        "--no-use-percentiles",
        "--tune-visual",
        "--load-bf16",
        "--no-backbone-trainable-params-fp32",
        "--optim",
        "adamw_torch_fused",
        "--learning-rate",
        "0.0001",
        "--warmup-ratio",
        "0.025",
        "--max-steps",
        str(TOTAL_UPDATES),
        "--global-batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--gradient-accumulation-steps",
        str(GRADIENT_ACCUMULATION_STEPS),
        "--episode-sampling-rate",
        "1.0",
        "--dataloader-num-workers",
        "0",
        "--save-steps",
        str(UPDATES_PER_ROUND),
        "--stop-every-steps",
        str(UPDATES_PER_ROUND),
        "--save-total-limit",
        str(DAGGER_ROUNDS),
        "--use-wandb",
    ]
    if round_index > 0:
        command.extend(
            [
                "--resume-checkpoint-path",
                str(checkpoint_dir(training_output, round_index)),
            ]
        )
    return command


def run_pipeline(args: argparse.Namespace) -> None:
    training_output = args.output_root / "training"
    parkour_env = os.environ | {
        "PYTHONPATH": str(REPO_ROOT),
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    training_env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPATH": f"{PARKOUR_ROOT}{os.pathsep}{REPO_ROOT}",
        "WANDB_RUN_ID": "extreme_parkour_go2_vla_dagger_joint_5k",
        "WANDB_RESUME": "allow",
    }

    for round_index in range(args.start_round, DAGGER_ROUNDS):
        round_dir = args.output_root / f"round_{round_index:02d}"
        raw_dir = round_dir / "raw"
        dataset_dir = round_dir / "lerobot"
        policy_config = round_dir / "policy.yaml"
        current_model = (
            args.bootstrap_checkpoint
            if round_index == 0
            else checkpoint_dir(training_output, round_index)
        )
        current_actor = (
            None
            if round_index == 0
            else current_model / "parkour_student_actor.pt"
        )
        write_policy_config(
            args.policy_config,
            policy_config,
            model_path=current_model,
            inference_device=args.inference_device,
        )
        for collector_rank in range(COLLECTOR_RANKS):
            subprocess.run(
                collector_command(
                    args.parkour_python,
                    mode="dagger-collect",
                    teacher_checkpoint=args.teacher_checkpoint,
                    policy_config=policy_config,
                    output_dir=raw_dir / f"rank_{collector_rank:02d}",
                    inference_device=args.inference_device,
                    num_envs=ENVS_PER_COLLECTOR_RANK,
                    seed=collector_rank + 1,
                    round_index=round_index,
                    student_actor_checkpoint=current_actor,
                    dagger_row_budget=(
                        PARKOUR_DAGGER_ROWS_PER_ROUND // COLLECTOR_RANKS
                    ),
                ),
                cwd=PARKOUR_ROOT,
                env=parkour_env,
                check=True,
            )
        subprocess.run(
            prepare_command(
                raw_dir,
                dataset_dir,
                round_index=round_index,
                normalization_source=args.bootstrap_dataset,
            ),
            cwd=REPO_ROOT,
            check=True,
        )
        trained_model = checkpoint_dir(training_output, round_index + 1)
        if not (trained_model / "parkour_student_actor.pt").is_file():
            subprocess.run(
                training_command(
                    bootstrap_checkpoint=args.bootstrap_checkpoint,
                    teacher_checkpoint=args.teacher_checkpoint,
                    dataset_dir=dataset_dir,
                    training_output=training_output,
                    round_index=round_index,
                ),
                cwd=REPO_ROOT,
                env=training_env,
                check=True,
            )

        eval_policy_config = round_dir / "eval_policy.yaml"
        write_policy_config(
            args.policy_config,
            eval_policy_config,
            model_path=trained_model,
            inference_device=args.inference_device,
        )
        subprocess.run(
            collector_command(
                args.parkour_python,
                mode="dagger-eval",
                teacher_checkpoint=args.teacher_checkpoint,
                policy_config=eval_policy_config,
                output_dir=round_dir / "eval",
                inference_device=args.inference_device,
                num_envs=50,
                seed=2,
                round_index=round_index,
                student_actor_checkpoint=(
                    trained_model / "parkour_student_actor.pt"
                ),
            ),
            cwd=PARKOUR_ROOT,
            env=parkour_env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--bootstrap-checkpoint", type=Path, default=DEFAULT_BOOTSTRAP_CHECKPOINT)
    parser.add_argument("--bootstrap-dataset", type=Path, default=DEFAULT_BOOTSTRAP_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--parkour-python", type=Path, default=DEFAULT_PARKOUR_PYTHON)
    parser.add_argument("--inference-device", default="cuda:1")
    parser.add_argument("--start-round", type=int, default=0)
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
