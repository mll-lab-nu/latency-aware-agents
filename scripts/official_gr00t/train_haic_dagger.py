#!/usr/bin/env python3
"""Run the HAIC privileged-oracle bootstrap and GR00T VLA DAgger pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HAIC_ROOT = REPO_ROOT / "third_party/HAIC"
HAIC_SCRIPT = HAIC_ROOT / "scripts/haic_vla.py"
GROOT_ROOT = REPO_ROOT / "third_party/Isaac-GR00T"
GR00T_PYTHON = GROOT_ROOT / ".venv/bin/python"
PREPARE_SCRIPT = REPO_ROOT / "scripts/official_gr00t/prepare_haic_dataset.py"
FINETUNE_SCRIPT = GROOT_ROOT / "gr00t/experiment/launch_finetune.py"
MODALITY_CONFIG = REPO_ROOT / "latency_bench/policy/official_gr00t_haic_modality.py"
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs/examples/haic/official_gr00t_eval.yaml"

BOOTSTRAP_TRAIN_EPISODES = 250
BOOTSTRAP_VAL_EPISODES = 25
BOOTSTRAP_UPDATES = 5_000
DAGGER_ROUNDS = 10
UPDATES_PER_ROUND = 500
TOTAL_UPDATES = DAGGER_ROUNDS * UPDATES_PER_ROUND
GLOBAL_BATCH_SIZE = 64
BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 72
COLLECTOR_RANKS = 2
ENVS_PER_COLLECTOR_RANK = 96
HAIC_DAGGER_ROWS_PER_ROUND = 2_304_000
HAIC_DAGGER_ROWS_PER_COLLECTOR = HAIC_DAGGER_ROWS_PER_ROUND // COLLECTOR_RANKS
EVAL_ENVS = 50
HAIC_CONTROL_STEPS_PER_VLA = 5

DEFAULT_TEACHER_CHECKPOINT = REPO_ROOT / (
    "outputs/haic/phase1_teacher/checkpoint_final.pt"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "outputs/official_gr00t/models/GR00T-N1.7-3B"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/official_gr00t/haic_pull_cart_vla_dagger"
DEFAULT_HAIC_PYTHON = Path("/pfs/users/lxy/conda/envs/haic/bin/python")


def checkpoint_dir(training_output: Path, completed_updates: int) -> Path:
    return training_output / f"checkpoint-{completed_updates}"


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


def export_teacher_actor(
    haic_python: Path,
    teacher_checkpoint: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(haic_python),
            "-m",
            "active_adaptation.learning.ppo.haic_actor",
            str(teacher_checkpoint),
            str(output_path),
        ],
        cwd=HAIC_ROOT,
    )
    return output_path


def collector_command(
    haic_python: Path,
    *,
    mode: str,
    teacher_checkpoint: Path,
    output_dir: Path,
    inference_device: str,
    num_envs: int,
    seed: int,
    round_index: int,
    policy_config: Path | None = None,
    split: str | None = None,
    student_actor_checkpoint: Path | None = None,
    row_budget: int | None = None,
    episode_budget: int | None = None,
    max_steps: int | None = None,
) -> list[str]:
    command = [
        str(haic_python),
        str(HAIC_SCRIPT),
        mode,
        "--headless",
        "--device",
        "cuda:0",
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--output-dir",
        str(output_dir),
        "--inference-device",
        inference_device,
        "--inference-batch-size",
        "8",
        "--num-envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--dagger-round",
        str(round_index),
        "--vla-cadence",
        str(HAIC_CONTROL_STEPS_PER_VLA),
    ]
    if split is not None:
        command.extend(["--split", split])
    if policy_config is not None:
        command.extend(["--policy-config", str(policy_config)])
    if row_budget is not None:
        command.extend(["--row-budget", str(row_budget)])
    if episode_budget is not None:
        command.extend(["--episode-budget", str(episode_budget)])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if student_actor_checkpoint is not None:
        command.extend(["--student-actor-checkpoint", str(student_actor_checkpoint)])
    return command


def prepare_command(
    raw_dir: Path,
    dataset_dir: Path,
    *,
    dagger: bool,
    round_index: int,
    expected_rows: int | None = None,
    normalization_source: Path | None = None,
) -> list[str]:
    command = [
        str(GR00T_PYTHON),
        str(PREPARE_SCRIPT),
        str(raw_dir),
        str(dataset_dir),
        "--force",
        "--round-index",
        str(round_index),
    ]
    if dagger:
        command.append("--dagger")
        command.extend(["--normalization-source", str(normalization_source)])
    if expected_rows is not None:
        command.extend(["--expected-rows", str(expected_rows)])
    return command


def _training_flags(
    *,
    base_model: Path,
    dataset_dir: Path,
    output_dir: Path,
    max_steps: int,
    gradient_accumulation_steps: int,
    checkpoint_interval: int,
    teacher_checkpoint: Path | None,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        str(GR00T_PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(FINETUNE_SCRIPT),
        "--base-model-path",
        str(base_model),
        "--dataset-path",
        str(dataset_dir),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(MODALITY_CONFIG),
        "--num-gpus",
        "1",
        "--output-dir",
        str(output_dir),
        "--max-state-dim",
        "605",
        "--max-action-dim",
        "275",
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
        str(max_steps),
        "--global-batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--episode-sampling-rate",
        "1.0",
        "--dataloader-num-workers",
        "0",
        "--save-steps",
        str(checkpoint_interval),
        "--stop-every-steps",
        str(checkpoint_interval),
        "--save-total-limit",
        "2",
        "--use-wandb",
    ]
    if teacher_checkpoint is not None:
        command.extend(["--haic-dagger", "--haic-actor-checkpoint", str(teacher_checkpoint)])
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint-path", str(resume_checkpoint)])
    return command


def bootstrap_training_command(
    *,
    base_model: Path,
    dataset_dir: Path,
    training_output: Path,
    max_steps: int = BOOTSTRAP_UPDATES,
    gradient_accumulation_steps: int = BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS,
) -> list[str]:
    """Build the single-GPU privileged-oracle bootstrap training command."""
    return _training_flags(
        base_model=base_model,
        dataset_dir=dataset_dir,
        output_dir=training_output,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_interval=max_steps,
        teacher_checkpoint=None,
        resume_checkpoint=None,
    )


def training_command(
    *,
    bootstrap_checkpoint: Path,
    actor_checkpoint: Path,
    dataset_dir: Path,
    training_output: Path,
    resume_checkpoint: Path | None,
    max_steps: int = TOTAL_UPDATES,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    checkpoint_interval: int = UPDATES_PER_ROUND,
) -> list[str]:
    """Build one HAIC DAgger training command with exact round resume."""
    return _training_flags(
        base_model=bootstrap_checkpoint,
        dataset_dir=dataset_dir,
        output_dir=training_output,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_interval=checkpoint_interval,
        teacher_checkpoint=actor_checkpoint,
        resume_checkpoint=resume_checkpoint,
    )


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    process_env = os.environ | {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}"
    }
    if env is not None:
        process_env.update(env)
    subprocess.run(command, cwd=cwd, env=process_env, check=True)


def run_bootstrap(args: argparse.Namespace, *, smoke: bool) -> Path:
    output_root = args.output_root / "bootstrap"
    raw_dir = output_root / "raw"
    dataset_dir = output_root / "lerobot"
    episodes_train = 2 if smoke else BOOTSTRAP_TRAIN_EPISODES
    episodes_val = 1 if smoke else BOOTSTRAP_VAL_EPISODES
    for split, episodes, seed in (("train", episodes_train, 0), ("val", episodes_val, 1)):
        _run(
            collector_command(
                args.haic_python,
                mode="bootstrap-collect",
                teacher_checkpoint=args.teacher_checkpoint,
                output_dir=raw_dir,
                inference_device=args.inference_device,
                num_envs=8 if smoke else ENVS_PER_COLLECTOR_RANK,
                seed=seed,
                round_index=0,
                split=split,
                episode_budget=episodes,
            ),
            cwd=HAIC_ROOT,
        )
    _run(
        prepare_command(raw_dir, dataset_dir, dagger=False, round_index=0),
        cwd=REPO_ROOT,
    )
    training_output = output_root / "training"
    _run(
        bootstrap_training_command(
            base_model=args.base_model,
            dataset_dir=dataset_dir,
            training_output=training_output,
            max_steps=1 if smoke else BOOTSTRAP_UPDATES,
        ),
        cwd=REPO_ROOT,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
            **({"WANDB_MODE": "offline"} if smoke else {}),
        },
    )
    return training_output / f"checkpoint-{1 if smoke else BOOTSTRAP_UPDATES}"


def run_smoke(args: argparse.Namespace) -> None:
    bootstrap_checkpoint = run_bootstrap(args, smoke=True)
    actor_checkpoint = export_teacher_actor(
        args.haic_python,
        args.teacher_checkpoint,
        args.output_root / "haic_teacher_actor.pt",
    )
    round_dir = args.output_root / "smoke_round_00"
    raw_dir = round_dir / "raw"
    dataset_dir = round_dir / "lerobot"
    policy_config = round_dir / "bootstrap_policy.yaml"
    write_policy_config(
        args.policy_config,
        policy_config,
        model_path=bootstrap_checkpoint,
        inference_device=args.inference_device,
    )
    _run(
        collector_command(
            args.haic_python,
            mode="dagger-collect",
            teacher_checkpoint=args.teacher_checkpoint,
            output_dir=raw_dir,
            inference_device=args.inference_device,
            num_envs=8,
            seed=2,
            round_index=0,
            policy_config=policy_config,
            split="train",
            row_budget=64,
        ),
        cwd=HAIC_ROOT,
    )
    _run(
        prepare_command(
            raw_dir,
            dataset_dir,
            dagger=True,
            round_index=0,
            expected_rows=64,
            normalization_source=args.output_root / "bootstrap/lerobot",
        ),
        cwd=REPO_ROOT,
    )
    training_output = round_dir / "training"
    _run(
        training_command(
            bootstrap_checkpoint=bootstrap_checkpoint,
            actor_checkpoint=actor_checkpoint,
            dataset_dir=dataset_dir,
            training_output=training_output,
            resume_checkpoint=None,
            max_steps=1,
            gradient_accumulation_steps=1,
            checkpoint_interval=1,
        ),
        cwd=REPO_ROOT,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
            "WANDB_MODE": "offline",
        },
    )
    eval_policy_config = round_dir / "eval_policy.yaml"
    write_policy_config(
        args.policy_config,
        eval_policy_config,
        model_path=training_output / "checkpoint-2",
        inference_device=args.inference_device,
    )
    _run(
        training_command(
            bootstrap_checkpoint=bootstrap_checkpoint,
            actor_checkpoint=actor_checkpoint,
            dataset_dir=dataset_dir,
            training_output=training_output,
            resume_checkpoint=checkpoint_dir(training_output, 1),
            max_steps=2,
            gradient_accumulation_steps=1,
            checkpoint_interval=1,
        ),
        cwd=REPO_ROOT,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
            "WANDB_MODE": "offline",
        },
    )
    _run(
        collector_command(
            args.haic_python,
            mode="dagger-eval",
            teacher_checkpoint=args.teacher_checkpoint,
            output_dir=round_dir / "eval",
            inference_device=args.inference_device,
            num_envs=1,
            seed=0,
            round_index=0,
            policy_config=eval_policy_config,
            student_actor_checkpoint=training_output / "checkpoint-2/haic_student_actor.pt",
            max_steps=600,
        ),
        cwd=HAIC_ROOT,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    bootstrap_checkpoint = args.bootstrap_checkpoint
    if args.run_bootstrap:
        bootstrap_checkpoint = run_bootstrap(args, smoke=False)
    actor_checkpoint = export_teacher_actor(
        args.haic_python,
        args.teacher_checkpoint,
        args.output_root / "haic_teacher_actor.pt",
    )
    training_output = args.output_root / "training"
    for round_index in range(args.start_round, DAGGER_ROUNDS):
        round_dir = args.output_root / f"round_{round_index:02d}"
        raw_dir = round_dir / "raw"
        dataset_dir = round_dir / "lerobot"
        current_model = (
            bootstrap_checkpoint
            if round_index == 0
            else checkpoint_dir(training_output, round_index * UPDATES_PER_ROUND)
        )
        current_actor = (
            None
            if round_index == 0
            else current_model / "haic_student_actor.pt"
        )
        policy_config = round_dir / "collector_policy.yaml"
        write_policy_config(
            args.policy_config,
            policy_config,
            model_path=current_model,
            inference_device=args.inference_device,
        )
        for collector_rank in range(COLLECTOR_RANKS):
            _run(
                collector_command(
                    args.haic_python,
                    mode="dagger-collect",
                    teacher_checkpoint=args.teacher_checkpoint,
                    output_dir=raw_dir / f"rank_{collector_rank:02d}",
                    inference_device=args.inference_device,
                    num_envs=ENVS_PER_COLLECTOR_RANK,
                    seed=collector_rank + 1,
                    round_index=round_index,
                    policy_config=policy_config,
                    split="train",
                    student_actor_checkpoint=current_actor,
                    row_budget=HAIC_DAGGER_ROWS_PER_COLLECTOR,
                ),
                cwd=HAIC_ROOT,
            )
        _run(
            prepare_command(
                raw_dir,
                dataset_dir,
                dagger=True,
                round_index=round_index,
                expected_rows=HAIC_DAGGER_ROWS_PER_ROUND,
                normalization_source=args.bootstrap_dataset,
            ),
            cwd=REPO_ROOT,
        )
        _run(
            training_command(
                bootstrap_checkpoint=bootstrap_checkpoint,
                actor_checkpoint=actor_checkpoint,
                dataset_dir=dataset_dir,
                training_output=training_output,
                resume_checkpoint=(current_model if round_index > 0 else None),
            ),
            cwd=REPO_ROOT,
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "NO_ALBUMENTATIONS_UPDATE": "1",
                "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
                "WANDB_RESUME": "allow",
            },
        )
        trained_model = checkpoint_dir(
            training_output, (round_index + 1) * UPDATES_PER_ROUND
        )
        eval_policy_config = round_dir / "eval_policy.yaml"
        write_policy_config(
            args.policy_config,
            eval_policy_config,
            model_path=trained_model,
            inference_device=args.inference_device,
        )
        _run(
            collector_command(
                args.haic_python,
                mode="dagger-eval",
                teacher_checkpoint=args.teacher_checkpoint,
                output_dir=round_dir / "eval",
                inference_device=args.inference_device,
                num_envs=EVAL_ENVS,
                seed=0,
                round_index=round_index,
                policy_config=eval_policy_config,
                split="val",
                student_actor_checkpoint=trained_model / "haic_student_actor.pt",
            ),
            cwd=HAIC_ROOT,
        )
        shutil.rmtree(raw_dir)
        shutil.rmtree(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-dataset", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--haic-python", type=Path, default=DEFAULT_HAIC_PYTHON)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_dataset is None:
        args.bootstrap_dataset = args.output_root / "bootstrap/lerobot"
    if args.smoke:
        run_smoke(args)
    else:
        if args.bootstrap_checkpoint is None and not args.run_bootstrap:
            args.bootstrap_checkpoint = (
                args.output_root / "bootstrap/training/checkpoint-5000"
            )
        run_pipeline(args)


if __name__ == "__main__":
    main()
