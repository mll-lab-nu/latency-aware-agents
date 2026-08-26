"""Generate five HumanoidBench 10M Sample Factory baseline jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.envs.humanoidbench import HUMANOIDBENCH_ENV_SPECS
from scripts.experiment_runner.job_manifest import (
    MANIFEST_V2_COLUMNS,
    SCHEMA_VERSION,
    build_job_id,
    write_manifest,
)
from scripts.scaffold_humanoidbench import build_env_config, build_probe_report


BASE_TRAIN_CONFIG = REPO_ROOT / (
    "configs/examples/gymnasium/h1hand_balance_hard/small_model_train.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/humanoidbench_sf_official_10m"
DEFAULT_TASKS = tuple(HUMANOIDBENCH_ENV_SPECS)
TRAIN_ENV_STEPS = 10_000_000
NUM_WORKERS = 32
NUM_ENVS_PER_WORKER = 1
WORKER_NUM_SPLITS = 1
ROLLOUT = 256
BATCH_SIZE = 512
NUM_BATCHES_PER_EPOCH = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 0.5
PPO_CLIP_RATIO = 0.2
PPO_CLIP_VALUE = 1e9
KL_LOSS_COEFF = 0.0
REWARD_CLIP = 1e9
ADAM_EPS = 1e-5
ADAPTIVE_STDDEV = False
WANDB_PROJECT = "latency-sensitive-bench"
WANDB_GROUP = "humanoidbench-sf-official-10m"
WANDB_JOB_TYPE = "sample_factory"
WANDB_TAGS = ["humanoidbench", "official-recipe", "raw-gaussian", "10m"]
SEED = 0
ENV_FPS = 50.0


def _task_env_config(task_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = build_probe_report(task_name)
    env_config = build_env_config(task_name, report)
    env_config = deepcopy(env_config)
    env_config["make_kwargs"] = {
        **env_config["make_kwargs"],
        "render_mode": None,
    }
    return env_config, report


def build_train_config(
    *,
    task_name: str,
    env_config: dict[str, Any],
    train_dir: Path,
    train_output_dir: Path,
    run_name: str,
    train_for_env_steps: int = TRAIN_ENV_STEPS,
) -> dict[str, Any]:
    config = load_config(BASE_TRAIN_CONFIG)
    config["experiment"]["name"] = run_name
    config["experiment"]["seed"] = SEED

    config["backend"]["device"] = "gpu"
    config["backend"]["train_dir"] = str(train_dir)
    config["backend"]["run_mode"] = "train"
    config["backend"]["restart_behavior"] = "resume"
    config["backend"]["actor_worker_gpus"] = []

    config["env"] = env_config
    config["env"]["task_name"] = task_name

    training = config["training"]
    training["train_for_env_steps"] = train_for_env_steps
    training["num_workers"] = NUM_WORKERS
    training["num_envs_per_worker"] = NUM_ENVS_PER_WORKER
    training["worker_num_splits"] = WORKER_NUM_SPLITS
    training["rollout"] = ROLLOUT
    training["batch_size"] = BATCH_SIZE
    training["num_batches_per_epoch"] = NUM_BATCHES_PER_EPOCH
    training["num_epochs"] = NUM_EPOCHS
    training["learning_rate"] = LEARNING_RATE
    training["ppo_clip_ratio"] = PPO_CLIP_RATIO
    training["ppo_clip_value"] = PPO_CLIP_VALUE
    training["max_grad_norm"] = MAX_GRAD_NORM
    training["kl_loss_coeff"] = KL_LOSS_COEFF
    training["reward_clip"] = REWARD_CLIP
    training["adam_eps"] = ADAM_EPS
    training["nonlinearity"] = "tanh"
    training["policy_initialization"] = "orthogonal"
    training["actor_critic_share_weights"] = False
    training["normalize_input"] = False
    training["normalize_returns"] = False
    training["value_bootstrap"] = True
    training["shuffle_minibatches"] = True
    training["adaptive_stddev"] = ADAPTIVE_STDDEV
    training["async_rl"] = False
    training["batched_sampling"] = False
    training["save_every_sec"] = 120
    training["save_best_every_sec"] = 60
    training["save_best_after"] = 100_000

    config["logging"]["output_dir"] = str(train_output_dir)
    config["logging"]["video"]["enabled"] = False
    config["logging"]["wandb_project"] = WANDB_PROJECT
    config["logging"]["wandb_group"] = WANDB_GROUP
    config["logging"]["wandb_job_type"] = WANDB_JOB_TYPE
    config["logging"]["wandb_tags"] = WANDB_TAGS
    return config


def build_eval_config(
    *,
    train_config: dict[str, Any],
    eval_output_dir: Path,
    horizon: int,
) -> dict[str, Any]:
    config = deepcopy(train_config)
    config["backend"]["run_mode"] = "eval"
    config["backend"]["extra_args"] = ["--load_checkpoint_kind", "best"]
    config["policy"] = {
        "type": "gymnasium_sf",
        "checkpoint_kind": "best",
        "device": "cuda",
        "deterministic": True,
    }
    config["evaluation"] = {
        "eval_episodes": 10,
        "eval_parallel_envs": 1,
        "eval_max_steps": horizon,
        "eval_deterministic": True,
        "eval_latency_values": [0],
    }
    config["env"]["make_kwargs"]["render_mode"] = "rgb_array"
    config["logging"]["output_dir"] = str(eval_output_dir)
    config["logging"]["video"]["enabled"] = True
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False
    return config


def _manifest_row(
    *,
    task_name: str,
    run_name: str,
    train_path: Path,
    eval_path: Path,
    train_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> dict[str, str]:
    train_dir = Path(train_config["backend"]["train_dir"])
    train_output_dir = Path(train_config["logging"]["output_dir"])
    eval_output_dir = Path(eval_config["logging"]["output_dir"])
    job_id = build_job_id(
        experiment_family="humanoidbench_sf_official_10m",
        env_name=task_name,
        sweep_name="baseline",
        latency_label="0",
        frame_stack=1,
        obs_fps=ENV_FPS,
        observation_stride=1,
        seed=SEED,
    )
    train_command = (
        f"{shlex.quote(sys.executable)} -m training.train --config "
        f"{shlex.quote(str(train_path))}"
    )
    eval_command = (
        f"{shlex.quote(sys.executable)} -m latency_bench.run --config "
        f"{shlex.quote(str(eval_path))}"
    )
    row = {column: "" for column in MANIFEST_V2_COLUMNS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "env": task_name,
            "experiment_family": "humanoidbench_sf_official_10m",
            "sweep_name": "baseline",
            "latency_label": "0",
            "seed": str(SEED),
            "frame_stack": "1",
            "obs_fps": str(int(ENV_FPS)),
            "observation_stride": "1",
            "train_config": str(train_path),
            "eval_kind": "sample_factory_zero_latency",
            "eval_config": str(eval_path),
            "train_command": train_command,
            "eval_command": eval_command,
            "run_name": run_name,
            "train_dir": str(train_dir),
            "checkpoint_dir": str(train_dir / run_name / "checkpoint_p0"),
            "train_output_dir": str(train_output_dir),
            "eval_output_dir": str(eval_output_dir),
            "train_artifact_dir": str(train_dir / run_name),
            "eval_artifact_dir": str(eval_output_dir),
            "latency_kind": "zero",
            "latency_fixed_frames": "0",
        }
    )
    return row


def generate_baseline(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    train_for_env_steps: int = TRAIN_ENV_STEPS,
) -> dict[str, Any]:
    output_root = output_root.expanduser()
    generated_root = output_root / "generated"
    train_dir = output_root / "checkpoints"
    rows: list[dict[str, str]] = []
    jobs: list[dict[str, Any]] = []

    for task_name in DEFAULT_TASKS:
        env_config, report = _task_env_config(task_name)
        task_root = output_root / task_name
        run_name = f"humanoidbench_sf_official_{task_name}_10m"
        train_config = build_train_config(
            task_name=task_name,
            env_config=env_config,
            train_dir=train_dir,
            train_output_dir=task_root / "train",
            run_name=run_name,
            train_for_env_steps=train_for_env_steps,
        )
        horizon = report["resolved_spec"]["max_episode_steps"]
        eval_config = build_eval_config(
            train_config=train_config,
            eval_output_dir=task_root / "eval_best_l0",
            horizon=horizon,
        )
        train_path = generated_root / task_name / "train.yaml"
        eval_path = generated_root / task_name / "eval.yaml"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")
        eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")
        rows.append(
            _manifest_row(
                task_name=task_name,
                run_name=run_name,
                train_path=train_path,
                eval_path=eval_path,
                train_config=train_config,
                eval_config=eval_config,
            )
        )
        jobs.append(
            {
                "task_name": task_name,
                "horizon": horizon,
                "train_config": str(train_path),
                "eval_config": str(eval_path),
                "checkpoint_dir": rows[-1]["checkpoint_dir"],
                "train_output_dir": str(task_root / "train"),
                "eval_output_dir": str(task_root / "eval_best_l0"),
            }
        )

    generated_root.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_root / "manifest.tsv"
    write_manifest(manifest_path, rows)
    summary = {
        "experiment_family": "humanoidbench_sf_official_10m",
        "tasks": list(DEFAULT_TASKS),
        "seed": SEED,
        "train_for_env_steps": train_for_env_steps,
        "num_workers": NUM_WORKERS,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "worker_num_splits": WORKER_NUM_SPLITS,
        "rollout": ROLLOUT,
        "batch_size": BATCH_SIZE,
        "num_batches_per_epoch": NUM_BATCHES_PER_EPOCH,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "ppo_clip_ratio": PPO_CLIP_RATIO,
        "ppo_clip_value": PPO_CLIP_VALUE,
        "max_grad_norm": MAX_GRAD_NORM,
        "kl_loss_coeff": KL_LOSS_COEFF,
        "reward_clip": REWARD_CLIP,
        "adam_eps": ADAM_EPS,
        "nonlinearity": "tanh",
        "actor_critic_share_weights": False,
        "normalize_input": False,
        "normalize_returns": False,
        "value_bootstrap": True,
        "shuffle_minibatches": True,
        "adaptive_stddev": ADAPTIVE_STDDEV,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_job_type": WANDB_JOB_TYPE,
        "wandb_tags": WANDB_TAGS,
        "backend_device": "gpu",
        "actor_worker_gpus": [],
        "env_render_mode": None,
        "manifest": str(manifest_path),
        "runner_command": (
            f"{shlex.quote(sys.executable)} -m scripts.experiment_runner.run_manifest_jobs "
            f"--manifest {shlex.quote(str(manifest_path))} --gpus 0,1"
        ),
        "jobs": jobs,
    }
    summary_path = generated_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-for-env-steps", type=int, default=TRAIN_ENV_STEPS)
    args = parser.parse_args(argv)
    summary = generate_baseline(
        output_root=args.output_root,
        train_for_env_steps=args.train_for_env_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
