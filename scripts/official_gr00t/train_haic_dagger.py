#!/usr/bin/env python3
"""Run the HAIC privileged-oracle bootstrap and GR00T VLA DAgger pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import wandb
import yaml

from latency_bench.core.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
HAIC_ROOT = REPO_ROOT / "third_party/HAIC"
HAIC_SCRIPT = HAIC_ROOT / "scripts/haic_vla.py"
HAIC_ACTOR_MODULE = "active_adaptation.learning.ppo.haic_actor"
GROOT_ROOT = REPO_ROOT / "third_party/Isaac-GR00T"
GR00T_PYTHON = GROOT_ROOT / ".venv/bin/python"
PREPARE_SCRIPT = REPO_ROOT / "scripts/official_gr00t/prepare_haic_dataset.py"
FINETUNE_SCRIPT = GROOT_ROOT / "gr00t/experiment/launch_finetune.py"
MODALITY_CONFIG = REPO_ROOT / "latency_bench/policy/official_gr00t_haic_modality.py"
ISSUED_MODALITY_CONFIG = (
    REPO_ROOT / "latency_bench/policy/official_gr00t_haic_issued_modality.py"
)
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs/examples/haic/official_gr00t_eval.yaml"

BOOTSTRAP_TRAIN_EPISODES = 250
BOOTSTRAP_VAL_EPISODES = 25
BOOTSTRAP_UPDATES = 5_000
DAGGER_ROUNDS = 10
UPDATES_PER_ROUND = 500
TOTAL_UPDATES = DAGGER_ROUNDS * UPDATES_PER_ROUND
GLOBAL_BATCH_SIZE = 64
BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 2
COLLECTOR_RANKS = 2
ENVS_PER_COLLECTOR_RANK = 96
HAIC_DAGGER_ROWS_PER_ROUND = (
    UPDATES_PER_ROUND * GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)
HAIC_DAGGER_ROWS_PER_COLLECTOR = HAIC_DAGGER_ROWS_PER_ROUND // COLLECTOR_RANKS
HAIC_DAGGER_SHARD_ROWS = 1_000
DEFAULT_INFERENCE_BATCH_SIZE = 32
CHECKPOINT_COMPLETE_MARKER = ".dagger-round-complete"
EVAL_ENVS = 100
HAIC_CONTROL_STEPS_PER_VLA = 5
DEFAULT_WANDB_RUN_ID = "haic_pull_cart_vla_dagger_action_h40_official_flow_aux_5k"

DEFAULT_TEACHER_CHECKPOINT = REPO_ROOT / (
    "outputs/haic/phase1_teacher/checkpoint_final.pt"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "outputs/official_gr00t/models/GR00T-N1.7-3B"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "outputs/official_gr00t/haic_pull_cart_vla_action_h40_dagger"
)
DEFAULT_HAIC_PYTHON = Path("/pfs/users/lxy/conda/envs/haic/bin/python")


def checkpoint_dir(training_output: Path, completed_updates: int) -> Path:
    return training_output / f"checkpoint-{completed_updates}"


def training_dataset_dirs(
    output_root: Path, round_index: int
) -> list[Path]:
    return [
        output_root / f"round_{index:02d}" / "lerobot"
        for index in range(round_index + 1)
    ]


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
    haic_python: Path,
    *,
    mode: str,
    teacher_checkpoint: Path,
    output_dir: Path,
    inference_device: str,
    num_envs: int,
    seed: int,
    round_index: int,
    simulation_device: str = "cuda:0",
    policy_config: Path | None = None,
    split: str | None = None,
    row_budget: int | None = None,
    episode_budget: int | None = None,
    max_steps: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    eval_config: Path | None = None,
    latency_config: Path | None = None,
    keep_failed: bool = False,
) -> list[str]:
    if mode == "latency-eval":
        return [
            str(haic_python),
            str(HAIC_SCRIPT),
            mode,
            "--headless",
            "--eval-config",
            eval_config.as_posix(),
        ]
    command = [
        str(haic_python),
        str(HAIC_SCRIPT),
        mode,
        "--headless",
        "--device",
        simulation_device,
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--output-dir",
        str(output_dir),
        "--inference-device",
        inference_device,
        "--inference-batch-size",
        str(inference_batch_size),
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
    if latency_config is not None:
        command.extend(["--latency-config", str(latency_config)])
    if row_budget is not None:
        command.extend(["--row-budget", str(row_budget)])
    if episode_budget is not None:
        command.extend(["--episode-budget", str(episode_budget)])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if keep_failed:
        command.append("--keep-failed")
    return command


def prepare_command(
    raw_dir: Path,
    dataset_dir: Path,
    *,
    dagger: bool,
    round_index: int,
    expected_rows: int | None = None,
    normalization_source: Path | None = None,
    modality_config: Path | None = None,
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
    if modality_config is not None:
        command.extend(["--modality-config", str(modality_config)])
    return command


def _training_flags(
    *,
    base_model: Path,
    dataset_dirs: list[Path],
    actor_checkpoint: Path,
    replay_dataset: Path | None,
    output_dir: Path,
    experiment_name: str,
    max_steps: int,
    gradient_accumulation_steps: int,
    checkpoint_interval: int,
    resume_checkpoint: Path | None,
    modality_config: Path = MODALITY_CONFIG,
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
        os.pathsep.join(str(path) for path in dataset_dirs),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(modality_config),
        "--num-gpus",
        "1",
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        experiment_name,
        "--max-state-dim",
        "605",
        "--max-action-dim",
        "256",
        "--action-horizon",
        "40",
        "--num-inference-timesteps",
        "4",
        "--allow-padding",
        "--action-objective",
        "haic",
        "--action-actor-checkpoint",
        str(actor_checkpoint),
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
        "0.00005",
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
        "--shard-size",
        str(HAIC_DAGGER_SHARD_ROWS),
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
    if replay_dataset is not None:
        command.extend(["--replay-bootstrap-dataset", str(replay_dataset)])
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint-path", str(resume_checkpoint)])
    return command


def bootstrap_training_command(
    *,
    base_model: Path,
    dataset_dir: Path,
    actor_checkpoint: Path,
    training_output: Path,
    experiment_name: str,
    max_steps: int = BOOTSTRAP_UPDATES,
    gradient_accumulation_steps: int = BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS,
) -> list[str]:
    """Build the single-GPU privileged-oracle bootstrap training command."""
    return _training_flags(
        base_model=base_model,
        dataset_dirs=[dataset_dir],
        actor_checkpoint=actor_checkpoint,
        replay_dataset=None,
        output_dir=training_output,
        experiment_name=experiment_name,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_interval=max_steps,
        resume_checkpoint=None,
        modality_config=ISSUED_MODALITY_CONFIG,
    )


def training_command(
    *,
    bootstrap_checkpoint: Path,
    replay_dataset: Path,
    dataset_dirs: list[Path],
    actor_checkpoint: Path,
    training_output: Path,
    experiment_name: str,
    resume_checkpoint: Path | None,
    max_steps: int = TOTAL_UPDATES,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    checkpoint_interval: int = UPDATES_PER_ROUND,
    modality_config: Path = MODALITY_CONFIG,
) -> list[str]:
    """Build one HAIC DAgger training command with exact round resume."""
    return _training_flags(
        base_model=bootstrap_checkpoint,
        dataset_dirs=dataset_dirs,
        actor_checkpoint=actor_checkpoint,
        replay_dataset=replay_dataset,
        output_dir=training_output,
        experiment_name=experiment_name,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_interval=checkpoint_interval,
        resume_checkpoint=resume_checkpoint,
        modality_config=modality_config,
    )


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    process_env = os.environ | {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}"
    }
    if env is not None:
        process_env.update(env)
    subprocess.run(command, cwd=cwd, env=process_env, check=True)


def read_eval_metrics(eval_path: Path) -> dict[str, int | float]:
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    task_metrics = result["task_metrics"]
    success_values = task_metrics["success"]["values"]
    return {
        "dagger_eval/success_rate": task_metrics["success"]["mean"],
        "dagger_eval/episodes": len(result["returns"]),
        "dagger_eval/successes": sum(success_values),
        "dagger_eval/motion_progress_mean": task_metrics["motion_progress"]["mean"],
        "dagger_eval/motion_progress_std": task_metrics["motion_progress"]["std"],
        "dagger_eval/cart_progress_mean": task_metrics["cart_progress"]["mean"],
        "dagger_eval/cart_progress_std": task_metrics["cart_progress"]["std"],
        "dagger_eval/pullcart_score_mean": task_metrics["pullcart_score"]["mean"],
        "dagger_eval/pullcart_score_std": task_metrics["pullcart_score"]["std"],
    }


def log_eval_metrics(
    eval_path: Path, *, step: int, run_id: str, mode: str | None = None
) -> None:
    init_kwargs = {
        "project": "finetune-gr00t-n1d7",
        "id": run_id,
        "name": run_id,
        "resume": "allow",
    }
    if mode is not None:
        init_kwargs["mode"] = mode
    run = wandb.init(**init_kwargs)
    run.define_metric("dagger_eval/*", step_metric="train/global_step")
    run.log({"train/global_step": step, **read_eval_metrics(eval_path)})
    run.finish()
    wandb.teardown()


def write_latency_eval_config(
    source: Path,
    destination: Path,
    *,
    model_path: Path,
    runtime_checkpoint: Path,
    latency_profile_path: Path,
    inference_device: str,
    simulator_device: str,
    output_dir: Path,
    num_envs: int,
    max_steps: int,
    latency_config: Path | None = None,
) -> None:
    config = load_config(source)
    if latency_config is not None:
        latency_sections = load_config(latency_config)
        config["latency"] = latency_sections["latency"]
        config["scheduler"] = latency_sections["scheduler"]
        config["executor"].update(latency_sections["executor"])
    config["executor"].update(
        {
            "mode": "simulated",
            "inference_devices": [inference_device],
            "simulated_inference_pool": True,
        }
    )
    config["env"]["simulator_device"] = simulator_device
    config["env"]["runtime_checkpoint_path"] = str(runtime_checkpoint)
    if latency_config is None:
        config["latency"] = {"method": "temporal"}
    config["latency"].update(
        {"profile_path": str(latency_profile_path), "sync_cuda": False}
    )
    config["policy"]["model_path"] = str(model_path)
    config["policy"]["device"] = inference_device
    config["evaluation"]["eval_episodes"] = num_envs
    config["evaluation"]["eval_parallel_envs"] = num_envs
    config["evaluation"]["eval_max_steps"] = max_steps
    config["logging"]["output_dir"] = str(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    policy_config: Path,
    output_dir: Path,
    step: int,
    environment: dict[str, str],
    num_envs: int = EVAL_ENVS,
) -> None:
    eval_config = output_dir / "eval_config.yaml"
    runtime_checkpoint = args.runtime_checkpoint
    write_latency_eval_config(
        policy_config,
        eval_config,
        model_path=yaml.safe_load(policy_config.read_text(encoding="utf-8"))["policy"]["model_path"],
        runtime_checkpoint=(
            args.teacher_checkpoint
            if runtime_checkpoint is None
            else runtime_checkpoint
        ),
        latency_profile_path=args.latency_profile_path,
        latency_config=args.latency_config,
        inference_device=args.inference_device,
        simulator_device="cuda:0",
        output_dir=output_dir,
        num_envs=num_envs,
        max_steps=args.eval_max_steps,
    )
    subprocess.run(
        collector_command(
            args.haic_python,
            mode="latency-eval",
            teacher_checkpoint=args.teacher_checkpoint,
            output_dir=output_dir,
            inference_device=args.inference_device,
            num_envs=num_envs,
            seed=0,
            round_index=step // UPDATES_PER_ROUND,
            policy_config=policy_config,
            max_steps=args.eval_max_steps,
            inference_batch_size=args.inference_batch_size,
            eval_config=eval_config,
        ),
        cwd=HAIC_ROOT,
        env=environment,
        check=True,
    )
    log_eval_metrics(
        output_dir / "queue_eval_latency_profile_sample.json",
        step=step,
        run_id=args.wandb_run_id,
        mode=environment.get("WANDB_MODE"),
    )


def archive_and_remove_raw_data(
    round_dir: Path, raw_dir: Path, dataset_dir: Path
) -> None:
    shutil.copy2(dataset_dir / "manifest.json", round_dir / "dataset_manifest.json")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)


def run_bootstrap(args: argparse.Namespace, *, smoke: bool) -> Path:
    output_root = args.output_root / "bootstrap"
    raw_dir = output_root / "raw"
    dataset_dir = args.bootstrap_dataset
    episodes_train = 2 if smoke else BOOTSTRAP_TRAIN_EPISODES
    episodes_val = 1 if smoke else BOOTSTRAP_VAL_EPISODES
    if dataset_dir is None:
        dataset_dir = output_root / "lerobot"
        _run(
            collector_command(
                args.haic_python,
                mode="oracle-eval",
                teacher_checkpoint=args.teacher_checkpoint,
                output_dir=output_root / "oracle_eval",
                inference_device=args.inference_device,
                num_envs=1 if smoke else EVAL_ENVS,
                seed=0,
                round_index=0,
                max_steps=args.eval_max_steps,
                inference_batch_size=args.inference_batch_size,
            ),
            cwd=HAIC_ROOT,
        )
        _run(
            [
                str(args.haic_python),
                "-m",
                HAIC_ACTOR_MODULE,
                str(args.teacher_checkpoint),
                str(args.actor_checkpoint),
            ],
            cwd=HAIC_ROOT,
        )
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
                    inference_batch_size=args.inference_batch_size,
                    latency_config=args.latency_config,
                    keep_failed=smoke,
                ),
                cwd=HAIC_ROOT,
            )
        _run(
            prepare_command(
                raw_dir,
                dataset_dir,
                dagger=False,
                round_index=0,
                modality_config=ISSUED_MODALITY_CONFIG,
            ),
            cwd=REPO_ROOT,
        )
    else:
        dataset_dir = dataset_dir.resolve()
    training_output = output_root / "training"
    experiment_name = f"{args.wandb_run_id}_sft"
    updates = 2 if smoke else BOOTSTRAP_UPDATES
    _run(
        bootstrap_training_command(
            base_model=args.base_model,
            dataset_dir=dataset_dir,
            actor_checkpoint=args.actor_checkpoint,
            training_output=training_output,
            experiment_name=experiment_name,
            max_steps=updates,
        ),
        cwd=REPO_ROOT,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
            "WANDB_RUN_ID": experiment_name,
            "WANDB_RESUME": "allow",
            **({"WANDB_MODE": "offline"} if smoke else {}),
        },
    )
    return training_output / experiment_name / f"checkpoint-{updates}"


def run_smoke(args: argparse.Namespace) -> Path:
    bootstrap_checkpoint = run_bootstrap(args, smoke=True)
    policy_config = args.output_root / "bootstrap/eval_policy.yaml"
    write_policy_config(
        args.policy_config,
        policy_config,
        model_path=bootstrap_checkpoint,
        inference_device=args.inference_device,
    )
    smoke_environment = os.environ | {
        "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
        "WANDB_MODE": "offline",
    }
    evaluate_checkpoint(
        args,
        policy_config=policy_config,
        output_dir=args.output_root / "bootstrap/eval",
        step=2,
        environment=smoke_environment,
        num_envs=1,
    )
    return bootstrap_checkpoint


def run_pipeline(args: argparse.Namespace) -> Path:
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    args.base_model = args.base_model.resolve()
    args.output_root = args.output_root.resolve()
    args.policy_config = args.policy_config.resolve()
    args.haic_python = args.haic_python.resolve()
    args.actor_checkpoint = args.actor_checkpoint.resolve()
    bootstrap_checkpoint = args.bootstrap_checkpoint
    if args.run_bootstrap:
        bootstrap_checkpoint = run_bootstrap(args, smoke=False)
    bootstrap_checkpoint = bootstrap_checkpoint.resolve()
    if args.bootstrap_dataset is None:
        args.bootstrap_dataset = args.output_root / "bootstrap/lerobot"
    args.bootstrap_dataset = args.bootstrap_dataset.resolve()
    training_output = args.output_root / "training"
    training_run_output = training_output / args.wandb_run_id
    haic_environment = os.environ | {
        "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    training_environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPATH": f"{HAIC_ROOT}{os.pathsep}{REPO_ROOT}",
        "WANDB_RUN_ID": args.wandb_run_id,
        "WANDB_RESUME": "allow",
    }
    first_round_marker = (
        checkpoint_dir(training_run_output, UPDATES_PER_ROUND)
        / CHECKPOINT_COMPLETE_MARKER
    )
    if args.start_round == 0 and (
        args.end_round == 0 or not first_round_marker.is_file()
    ):
        bootstrap_policy_config = args.output_root / "bootstrap/eval_policy.yaml"
        write_policy_config(
            args.policy_config,
            bootstrap_policy_config,
            model_path=bootstrap_checkpoint,
            inference_device=args.inference_device,
        )
        evaluate_checkpoint(
            args,
            policy_config=bootstrap_policy_config,
            output_dir=args.output_root / "bootstrap/vla_eval",
            step=0,
            environment=haic_environment,
        )
    for round_index in range(args.start_round, args.end_round):
        round_dir = args.output_root / f"round_{round_index:02d}"
        raw_dir = round_dir / "raw"
        dataset_dir = round_dir / "lerobot"
        current_model = (
            bootstrap_checkpoint
            if round_index == 0
            else checkpoint_dir(training_run_output, round_index * UPDATES_PER_ROUND)
        )
        trained_model = checkpoint_dir(
            training_run_output, (round_index + 1) * UPDATES_PER_ROUND
        )
        complete_marker = trained_model / CHECKPOINT_COMPLETE_MARKER
        datasets = training_dataset_dirs(args.output_root, round_index)
        training_datasets_path = round_dir / "training_datasets.json"
        training_datasets_manifest = {
            "round_index": round_index,
            "datasets": [str(path) for path in datasets],
            "replay_dataset": str(args.bootstrap_dataset),
            "replay_ratio": 0.5,
        }
        policy_config = round_dir / "collector_policy.yaml"
        write_policy_config(
            args.policy_config,
            policy_config,
            model_path=current_model,
            inference_device=args.inference_device,
        )
        if not complete_marker.is_file():
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            if trained_model.exists():
                shutil.rmtree(trained_model)
            collector_commands = []
            collector_environments = []
            for collector_rank in range(COLLECTOR_RANKS):
                collector_device = (
                    None
                    if args.collector_devices is None
                    else args.collector_devices[collector_rank]
                )
                collector_commands.append(
                    collector_command(
                        args.haic_python,
                        mode="dagger-collect",
                        teacher_checkpoint=args.teacher_checkpoint,
                        output_dir=raw_dir / f"rank_{collector_rank:02d}",
                        inference_device=(
                            args.inference_device
                            if collector_device is None
                            else collector_device
                        ),
                        simulation_device=(
                            "cuda:0" if collector_device is None else collector_device
                        ),
                        num_envs=args.collector_num_envs,
                        seed=collector_rank + 1,
                        round_index=round_index,
                        policy_config=policy_config,
                        split="train",
                        row_budget=HAIC_DAGGER_ROWS_PER_COLLECTOR,
                        inference_batch_size=args.inference_batch_size,
                    )
                )
                collector_environments.append(
                    haic_environment
                    | (
                        {"LOCAL_RANK": collector_device.split(":", 1)[1]}
                        if args.collector_devices is not None
                        else {}
                    )
                )
            if args.collector_devices is None:
                subprocess.run(
                    collector_commands[0],
                    cwd=HAIC_ROOT,
                    env=collector_environments[0],
                    check=True,
                )
                subprocess.run(
                    collector_commands[1],
                    cwd=HAIC_ROOT,
                    env=collector_environments[1],
                    check=True,
                )
            else:
                processes = []
                try:
                    for command, environment in zip(
                        collector_commands, collector_environments
                    ):
                        processes.append(
                            subprocess.Popen(
                                command,
                                cwd=HAIC_ROOT,
                                env=environment,
                            )
                        )
                    for process, command in zip(processes, collector_commands):
                        return_code = process.wait()
                        if return_code:
                            raise subprocess.CalledProcessError(return_code, command)
                except BaseException:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    for process in processes:
                        process.wait()
                    raise
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
            subprocess.run(
                training_command(
                    bootstrap_checkpoint=bootstrap_checkpoint,
                    replay_dataset=args.bootstrap_dataset,
                    dataset_dirs=datasets,
                    actor_checkpoint=args.actor_checkpoint,
                    training_output=training_output,
                    experiment_name=args.wandb_run_id,
                    resume_checkpoint=(current_model if round_index > 0 else None),
                ),
                cwd=REPO_ROOT,
                env=training_environment,
                check=True,
            )
            training_datasets_path.write_text(
                json.dumps(training_datasets_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_marker = trained_model / f"{CHECKPOINT_COMPLETE_MARKER}.tmp"
            temporary_marker.write_text("complete\n", encoding="utf-8")
            temporary_marker.replace(complete_marker)
        else:
            recorded_training_datasets = json.loads(
                training_datasets_path.read_text(encoding="utf-8")
            )
            if recorded_training_datasets != training_datasets_manifest:
                raise RuntimeError(
                    f"{complete_marker} was produced with a different DAgger dataset aggregate"
                )
        eval_policy_config = round_dir / "eval_policy.yaml"
        write_policy_config(
            args.policy_config,
            eval_policy_config,
            model_path=trained_model,
            inference_device=args.inference_device,
        )
        evaluate_checkpoint(
            args,
            policy_config=eval_policy_config,
            output_dir=round_dir / "eval",
            step=(round_index + 1) * UPDATES_PER_ROUND,
            environment=haic_environment,
        )
        archive_and_remove_raw_data(round_dir, raw_dir, dataset_dir)
    if args.end_round == 0:
        return bootstrap_checkpoint
    return checkpoint_dir(training_run_output, args.end_round * UPDATES_PER_ROUND)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument(
        "--runtime-checkpoint",
        type=Path,
        help="Original frozen teacher checkpoint used by closed-loop evals.",
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-dataset", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--haic-python", type=Path, default=DEFAULT_HAIC_PYTHON)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument(
        "--collector-devices",
        nargs=COLLECTOR_RANKS,
        default=None,
        help=(
            "Run both collectors in parallel on these devices; omitted keeps serial "
            "single-GPU collection."
        ),
    )
    parser.add_argument(
        "--collector-num-envs",
        type=int,
        default=ENVS_PER_COLLECTOR_RANK,
        help="Number of environments per DAgger collector for render A/B tests.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    parser.add_argument("--eval-max-steps", type=int, default=1_000)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--wandb-run-id", default=DEFAULT_WANDB_RUN_ID)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, default=DAGGER_ROUNDS)
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--latency-profile-path",
        type=Path,
        required=True,
        help="Temporal latency profile used by every periodic closed-loop evaluation.",
    )
    parser.add_argument(
        "--latency-config",
        type=Path,
        help="Latency command config used for native H40 bootstrap collection.",
    )
    args = parser.parse_args()
    if args.actor_checkpoint is None:
        args.actor_checkpoint = args.output_root / "bootstrap/actor_adapt.pt"
    if args.bootstrap_dataset is None and not args.run_bootstrap:
        args.bootstrap_dataset = args.output_root / "bootstrap/lerobot"
    if args.smoke:
        model_path = run_smoke(args)
        print(f"MODEL_PATH={model_path}", flush=True)
    else:
        if args.bootstrap_checkpoint is None and not args.run_bootstrap:
            args.bootstrap_checkpoint = (
                args.output_root
                / "bootstrap/training"
                / f"{args.wandb_run_id}_sft"
                / "checkpoint-5000"
            )
        model_path = run_pipeline(args)
        print(f"MODEL_PATH={model_path}", flush=True)


if __name__ == "__main__":
    main()
