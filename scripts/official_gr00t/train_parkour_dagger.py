#!/usr/bin/env python3
"""Run the Extreme Parkour VLA action-supervised DAgger pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import wandb
import yaml

from latency_bench.data.parkour_dagger import PARKOUR_ACTION_HORIZON
from latency_bench.utils.stats import std as series_std


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
BOOTSTRAP_UPDATES = 5_000
GLOBAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
DAGGER_ENVS = 192
PARKOUR_DAGGER_ROWS_PER_ROUND = (
    UPDATES_PER_ROUND * GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)
PARKOUR_DAGGER_SHARD_ROWS = 1_000
DEFAULT_INFERENCE_BATCH_SIZE = 32
PARKOUR_EVAL_INFERENCE_BATCH_SIZE = 8
DEFAULT_PARKOUR_STARTUP_TIMEOUT_SEC = 300.0
PARKOUR_EVAL_EPISODES = 100
PARKOUR_EVAL_MAX_EPISODE_STEPS = 1500
CHECKPOINT_COMPLETE_MARKER = ".dagger-round-complete"
EVAL_METRICS = (
    "reward",
    "episode_length",
    "normalized_waypoint_progress",
    "edge_violation",
)

DEFAULT_TEACHER_CHECKPOINT = PARKOUR_ROOT / (
    "logs/rsl_rl/unitree_go2_parkour/"
    "2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k/"
    "model_14999.pt"
)
DEFAULT_BASE_MODEL = REPO_ROOT / "outputs/official_gr00t/models/GR00T-N1.7-3B"
DEFAULT_BOOTSTRAP_DATASET = REPO_ROOT / (
    "outputs/official_gr00t/extreme_parkour_go2_action_data/lerobot_h40_semantic_stats"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_flow_aux_dagger"
)
DEFAULT_POLICY_CONFIG = REPO_ROOT / (
    "configs/examples/extreme_parkour/official_gr00t_eval.yaml"
)
DEFAULT_PARKOUR_PYTHON = Path("/pfs/users/lxy/conda/envs/extreme-parkour/bin/python")


def checkpoint_dir(training_output: Path, completed_rounds: int) -> Path:
    return training_output / f"checkpoint-{completed_rounds * UPDATES_PER_ROUND}"


def bootstrap_checkpoint_dir(
    training_output: Path,
    experiment_name: str,
    updates: int = BOOTSTRAP_UPDATES,
) -> Path:
    return training_output / experiment_name / f"checkpoint-{updates}"


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
    inference_device: list[str],
) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["policy"]["model_path"] = str(model_path)
    config["policy"]["device"] = inference_device[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_eval_config(
    source: Path,
    destination: Path,
    *,
    model_path: Path,
    teacher_checkpoint: Path,
    latency_profile_path: Path,
    output_dir: Path,
    inference_device: list[str],
    eval_episodes: int = PARKOUR_EVAL_EPISODES,
    max_episode_steps: int = PARKOUR_EVAL_MAX_EPISODE_STEPS,
    num_envs: int = 50,
) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["executor"] = {
        "mode": "simulated",
        "simulated_inference_pool": True,
        "inference_devices": list(inference_device),
        "inference_batch_size": PARKOUR_EVAL_INFERENCE_BATCH_SIZE,
    }
    config["env"].update(
        {
            "runtime_checkpoint_path": str(teacher_checkpoint),
            "simulator_device": "cuda:0",
        }
    )
    config["policy"].update(
        {
            "model_path": str(model_path),
            "device": inference_device[0],
        }
    )
    config["latency"] = {
        "method": "temporal",
        "profile_path": str(latency_profile_path),
        "sync_cuda": False,
    }
    config["evaluation"].update(
        {
            "eval_episodes": int(eval_episodes),
            "eval_parallel_envs": int(num_envs),
            "eval_max_steps": int(max_episode_steps),
            "eval_deterministic": True,
        }
    )
    config["logging"]["output_dir"] = str(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def collector_command(
    parkour_python: Path,
    *,
    mode: str,
    teacher_checkpoint: Path,
    policy_config: Path,
    output_dir: Path,
    inference_device: list[str],
    num_envs: int,
    seed: int,
    round_index: int,
    dagger_row_budget: int = PARKOUR_DAGGER_ROWS_PER_ROUND,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    eval_episodes: int = PARKOUR_EVAL_EPISODES,
    max_episode_steps: int = PARKOUR_EVAL_MAX_EPISODE_STEPS,
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
        "--eval-config" if mode == "latency-eval" else "--policy_config",
        str(policy_config),
        "--output_dir",
        str(output_dir),
        "--inference_batch_size",
        str(inference_batch_size),
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
    ]
    for device in inference_device:
        command.extend(["--inference_device", device])
    if mode == "dagger-collect":
        command.extend(
            [
                "--dagger_round",
                str(round_index),
                "--dagger_shard_rows",
                str(PARKOUR_DAGGER_SHARD_ROWS),
                "--dagger_row_budget",
                str(dagger_row_budget),
            ]
        )
    elif mode.endswith("-eval"):
        command.extend(
            [
                "--eval_episodes",
                str(eval_episodes),
                "--max_episode_steps",
                str(max_episode_steps),
            ]
        )
    return command


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _exit_on_sigterm(signum: int, _frame) -> None:
    raise SystemExit(128 + signum)


def run_parkour_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    startup_timeout_sec: float,
) -> None:
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        with tempfile.TemporaryDirectory(prefix="parkour-startup-") as temporary_dir:
            ready_file = Path(temporary_dir) / "ready"
            command = [*command, "--startup-ready-file", str(ready_file)]
            for attempt in range(2):
                ready_file.unlink(missing_ok=True)
                process = None
                try:
                    previous_signal_mask = signal.pthread_sigmask(
                        signal.SIG_BLOCK, {signal.SIGTERM}
                    )
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=cwd,
                            env=env,
                            start_new_session=True,
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
                    deadline = time.monotonic() + startup_timeout_sec
                    while not ready_file.is_file():
                        return_code = process.poll()
                        if return_code is not None:
                            if return_code:
                                raise subprocess.CalledProcessError(return_code, command)
                            raise RuntimeError(
                                "Parkour process exited before signaling startup readiness"
                            )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, startup_timeout_sec)
                        time.sleep(min(1.0, remaining))
                    return_code = process.wait()
                    if return_code:
                        raise subprocess.CalledProcessError(return_code, command)
                    return
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    if attempt == 0:
                        print(
                            f"Parkour startup timed out after {startup_timeout_sec:g}s; "
                            "retrying once",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    raise
                except BaseException:
                    if process is not None:
                        _kill_process_group(process)
                    raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


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


def bootstrap_prepare_command(raw_dir: Path, dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PREPARE_SCRIPT),
        str(raw_dir),
        str(dataset_dir),
        "--force",
        "--video-shards",
    ]


def training_command(
    *,
    bootstrap_checkpoint: Path,
    replay_dataset: Path | None,
    dataset_dirs: list[Path],
    training_output: Path,
    round_index: int,
    experiment_name: str,
    actor_checkpoint: Path,
    max_steps: int = TOTAL_UPDATES,
    stop_every_steps: int = UPDATES_PER_ROUND,
) -> list[str]:
    command = [
        str(GROOT_ROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=2",
        str(FINETUNE_SCRIPT),
        "--base-model-path",
        str(bootstrap_checkpoint),
        "--dataset-path",
        os.pathsep.join(str(path) for path in dataset_dirs),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(MODALITY_CONFIG),
        "--num-gpus",
        "2",
        "--output-dir",
        str(training_output),
        "--experiment-name",
        experiment_name,
        "--max-state-dim",
        "53",
        "--action-horizon",
        str(PARKOUR_ACTION_HORIZON),
        "--num-inference-timesteps",
        "4",
        "--allow-padding",
        "--action-objective",
        "parkour",
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
        str(GRADIENT_ACCUMULATION_STEPS),
        "--episode-sampling-rate",
        "1.0",
        "--shard-size",
        str(PARKOUR_DAGGER_SHARD_ROWS),
        "--dataloader-num-workers",
        "8",
        "--save-steps",
        str(UPDATES_PER_ROUND),
        "--stop-every-steps",
        str(stop_every_steps),
        "--save-total-limit",
        "2",
        "--use-wandb",
    ]
    if replay_dataset is not None:
        command.extend(["--replay-bootstrap-dataset", str(replay_dataset)])
    if round_index > 0:
        command.extend(
            [
                "--resume-checkpoint-path",
                str(checkpoint_dir(training_output / experiment_name, round_index)),
            ]
        )
    return command


def bootstrap_training_command(
    *,
    base_model: Path,
    dataset_dir: Path,
    training_output: Path,
    experiment_name: str,
    actor_checkpoint: Path,
    max_steps: int = BOOTSTRAP_UPDATES,
) -> list[str]:
    return training_command(
        bootstrap_checkpoint=base_model,
        replay_dataset=None,
        dataset_dirs=[dataset_dir],
        training_output=training_output,
        round_index=0,
        experiment_name=experiment_name,
        actor_checkpoint=actor_checkpoint,
        max_steps=max_steps,
        stop_every_steps=max_steps,
    )


def read_eval_metrics(eval_path: Path) -> dict[str, float]:
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    task_metrics = result["task_metrics"]
    metric_values = {
        "reward": {
            "mean": result["mean_return"],
            "std": result["std_return"],
        },
        "episode_length": {
            "mean": result["mean_length"],
            "std": series_std(result["lengths"]),
        },
        **{
            name: task_metrics[name]
            for name in ("normalized_waypoint_progress", "edge_violation")
        },
    }
    return {
        f"dagger_eval/{metric}_{stat}": metric_values[metric][stat]
        for metric in EVAL_METRICS
        for stat in ("mean", "std")
    }


def run_bootstrap(args: argparse.Namespace) -> Path:
    bootstrap_updates = args.bootstrap_updates
    bootstrap_root = args.output_root / "bootstrap"
    raw_dir = (
        bootstrap_root / "raw" if args.bootstrap_raw is None else args.bootstrap_raw
    )
    if args.bootstrap_raw is None:
        parkour_env = os.environ | {
            "PYTHONPATH": str(REPO_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }
        for mode in ("teacher-eval", "oracle-eval"):
            run_parkour_command(
                collector_command(
                    args.parkour_python,
                    mode=mode,
                    teacher_checkpoint=args.teacher_checkpoint,
                    policy_config=args.policy_config,
                    output_dir=bootstrap_root / mode,
                    inference_device=args.inference_device,
                    num_envs=50,
                    seed=2,
                    round_index=0,
                    inference_batch_size=args.inference_batch_size,
                ),
                cwd=PARKOUR_ROOT,
                env=parkour_env,
                startup_timeout_sec=args.parkour_startup_timeout_sec,
            )
        teacher_progress = json.loads(
            (bootstrap_root / "teacher-eval/teacher-eval.json").read_text(
                encoding="utf-8"
            )
        )["normalized_waypoint_progress"]["mean"]
        oracle_progress = json.loads(
            (bootstrap_root / "oracle-eval/oracle-eval.json").read_text(
                encoding="utf-8"
            )
        )["normalized_waypoint_progress"]["mean"]
        if oracle_progress < teacher_progress * 0.5:
            raise RuntimeError(
                "Parkour oracle normalized waypoint progress is below 50% of teacher"
            )
        run_parkour_command(
            collector_command(
                args.parkour_python,
                mode="collect",
                teacher_checkpoint=args.teacher_checkpoint,
                policy_config=args.policy_config,
                output_dir=raw_dir,
                inference_device=args.inference_device,
                num_envs=50,
                seed=0,
                round_index=0,
                inference_batch_size=args.inference_batch_size,
            ),
            cwd=PARKOUR_ROOT,
            env=parkour_env,
            startup_timeout_sec=args.parkour_startup_timeout_sec,
        )
    subprocess.run(
        bootstrap_prepare_command(raw_dir, args.bootstrap_dataset),
        cwd=REPO_ROOT,
        env=os.environ | {"PYTHONPATH": str(REPO_ROOT)},
        check=True,
    )
    experiment_name = f"{args.wandb_run_id}_sft"
    training_output = bootstrap_root / "training"
    subprocess.run(
        bootstrap_training_command(
            base_model=args.base_model,
            dataset_dir=args.bootstrap_dataset,
            training_output=training_output,
            experiment_name=experiment_name,
            actor_checkpoint=args.teacher_checkpoint,
            max_steps=bootstrap_updates,
        ),
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPATH": f"{GROOT_ROOT}{os.pathsep}{REPO_ROOT}",
            "WANDB_RUN_ID": experiment_name,
            "WANDB_NAME": experiment_name,
            "WANDB_RESUME": "allow",
        },
        check=True,
    )
    return bootstrap_checkpoint_dir(
        training_output, experiment_name, bootstrap_updates
    )


def log_eval_metrics(eval_path: Path, *, step: int, run_id: str) -> None:
    run = wandb.init(
        project="finetune-gr00t-n1d7",
        id=run_id,
        resume="allow",
    )
    run.define_metric("dagger_eval/*", step_metric="train/global_step")
    run.log({"train/global_step": step, **read_eval_metrics(eval_path)})
    run.finish()
    wandb.teardown()


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    policy_config: Path,
    output_dir: Path,
    step: int,
    environment: dict[str, str],
) -> None:
    eval_config = output_dir / "eval_config.yaml"
    write_eval_config(
        policy_config,
        eval_config,
        model_path=Path(yaml.safe_load(policy_config.read_text(encoding="utf-8"))["policy"]["model_path"]),
        teacher_checkpoint=args.teacher_checkpoint,
        latency_profile_path=args.latency_profile_path,
        output_dir=output_dir,
        inference_device=args.inference_device,
    )
    run_parkour_command(
        collector_command(
            args.parkour_python,
            mode="latency-eval",
            teacher_checkpoint=args.teacher_checkpoint,
            policy_config=eval_config,
            output_dir=output_dir,
            inference_device=args.inference_device,
            num_envs=50,
            seed=2,
            round_index=step // UPDATES_PER_ROUND,
            inference_batch_size=args.inference_batch_size,
        ),
        cwd=PARKOUR_ROOT,
        env=environment,
        startup_timeout_sec=args.parkour_startup_timeout_sec,
    )
    log_eval_metrics(output_dir / "queue_eval_latency_profile_sample.json", step=step, run_id=args.wandb_run_id)


def run_pipeline(args: argparse.Namespace) -> None:
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    args.base_model = args.base_model.resolve()
    args.bootstrap_dataset = args.bootstrap_dataset.resolve()
    args.output_root = args.output_root.resolve()
    args.policy_config = args.policy_config.resolve()
    args.parkour_python = args.parkour_python.resolve()
    bootstrap_checkpoint = args.bootstrap_checkpoint
    if args.run_bootstrap:
        bootstrap_checkpoint = run_bootstrap(args)
    bootstrap_checkpoint = bootstrap_checkpoint.resolve()
    training_output = args.output_root / "training"
    training_run_output = training_output / args.wandb_run_id
    parkour_env = os.environ | {
        "PYTHONPATH": str(REPO_ROOT),
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    prepare_env = os.environ | {"PYTHONPATH": str(REPO_ROOT)}
    training_env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPATH": f"{PARKOUR_ROOT}{os.pathsep}{REPO_ROOT}",
        "WANDB_RUN_ID": args.wandb_run_id,
        "WANDB_NAME": args.wandb_run_id,
        "WANDB_RESUME": "allow",
    }
    first_round_marker = (
        checkpoint_dir(training_run_output, 1) / CHECKPOINT_COMPLETE_MARKER
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
            environment=parkour_env,
        )
    for round_index in range(args.start_round, args.end_round):
        round_dir = args.output_root / f"round_{round_index:02d}"
        raw_dir = round_dir / "raw"
        dataset_dir = round_dir / "lerobot"
        policy_config = round_dir / "policy.yaml"
        current_model = (
            bootstrap_checkpoint
            if round_index == 0
            else checkpoint_dir(training_run_output, round_index)
        )
        write_policy_config(
            args.policy_config,
            policy_config,
            model_path=current_model,
            inference_device=args.inference_device,
        )
        run_parkour_command(
            collector_command(
                args.parkour_python,
                mode="dagger-collect",
                teacher_checkpoint=args.teacher_checkpoint,
                policy_config=policy_config,
                output_dir=raw_dir,
                inference_device=args.inference_device,
                num_envs=DAGGER_ENVS,
                seed=1,
                round_index=round_index,
                inference_batch_size=args.inference_batch_size,
            ),
            cwd=PARKOUR_ROOT,
            env=parkour_env,
            startup_timeout_sec=args.parkour_startup_timeout_sec,
        )
        subprocess.run(
            prepare_command(
                raw_dir,
                dataset_dir,
                round_index=round_index,
                normalization_source=args.bootstrap_dataset,
            ),
            cwd=REPO_ROOT,
            env=prepare_env,
            check=True,
        )
        trained_model = checkpoint_dir(training_run_output, round_index + 1)
        complete_marker = trained_model / CHECKPOINT_COMPLETE_MARKER
        datasets = training_dataset_dirs(args.output_root, round_index)
        training_datasets_path = round_dir / "training_datasets.json"
        training_datasets_manifest = {
            "round_index": round_index,
            "datasets": [str(path) for path in datasets],
            "replay_dataset": str(args.bootstrap_dataset),
            "replay_ratio": 0.5,
        }
        if complete_marker.is_file():
            recorded_training_datasets = json.loads(
                training_datasets_path.read_text(encoding="utf-8")
            )
            if recorded_training_datasets != training_datasets_manifest:
                raise RuntimeError(
                    f"{complete_marker} was produced with a different DAgger dataset aggregate"
                )
        else:
            subprocess.run(
                training_command(
                    bootstrap_checkpoint=bootstrap_checkpoint,
                    replay_dataset=args.bootstrap_dataset,
                    dataset_dirs=datasets,
                    training_output=training_output,
                    round_index=round_index,
                    experiment_name=args.wandb_run_id,
                    actor_checkpoint=args.teacher_checkpoint,
                ),
                cwd=REPO_ROOT,
                env=training_env,
                check=True,
            )
            training_datasets_path.write_text(
                json.dumps(training_datasets_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_marker = trained_model / f"{CHECKPOINT_COMPLETE_MARKER}.tmp"
            temporary_marker.write_text("complete\n", encoding="utf-8")
            temporary_marker.replace(complete_marker)

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
            environment=parkour_env,
        )
        shutil.copy2(dataset_dir / "manifest.json", round_dir / "dataset_manifest.json")
        shutil.rmtree(raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-dataset", type=Path, default=DEFAULT_BOOTSTRAP_DATASET)
    parser.add_argument("--bootstrap-raw", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--parkour-python", type=Path, default=DEFAULT_PARKOUR_PYTHON)
    parser.add_argument(
        "--parkour-startup-timeout-sec",
        type=float,
        default=DEFAULT_PARKOUR_STARTUP_TIMEOUT_SEC,
    )
    parser.add_argument("--inference-device", action="append")
    parser.add_argument(
        "--wandb-run-id",
        default="extreme_parkour_go2_vla_dagger_action_yaw_h40_official_flow_aux_5k",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, default=DAGGER_ROUNDS)
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-updates", type=int, default=BOOTSTRAP_UPDATES)
    parser.add_argument(
        "--latency-profile-path",
        type=Path,
        required=True,
        help="Temporal latency profile used by every periodic closed-loop evaluation.",
    )
    args = parser.parse_args()
    if args.inference_device is None:
        args.inference_device = ["cuda:1"]
    if args.bootstrap_checkpoint is None and not args.run_bootstrap:
        args.bootstrap_checkpoint = bootstrap_checkpoint_dir(
            args.output_root / "bootstrap" / "training",
            f"{args.wandb_run_id}_sft",
            args.bootstrap_updates,
        )
    run_pipeline(args)
    model_path = (
        bootstrap_checkpoint_dir(
            args.output_root / "bootstrap" / "training",
            f"{args.wandb_run_id}_sft",
            args.bootstrap_updates,
        )
        if args.end_round == 0
        else checkpoint_dir(
            args.output_root / "training" / args.wandb_run_id,
            args.end_round,
        )
    )
    print(f"MODEL_PATH={model_path}", flush=True)


if __name__ == "__main__":
    main()
