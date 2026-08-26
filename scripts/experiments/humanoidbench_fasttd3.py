"""Generate repeatable FastTD3 HumanoidBench train/eval jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/humanoidbench_fasttd3"
DEFAULT_TASKS = tuple(HUMANOIDBENCH_ENV_SPECS)
TASK_TRAIN_ENV_STEPS = {
    "h1hand_balance_hard": 10_000_000,
    "h1hand_hurdle": 12_160_000,
    "h1hand_run": 9_600_000,
    "h1hand_basketball": 10_000_000,
    "h1strong_highbar_hard": 10_000_000,
}
NUM_WORKERS = 128
NUM_ENVS_PER_WORKER = 1
WORKER_NUM_SPLITS = 1
ROLLOUT = 1
COLLECTION_BATCH_SIZE = 128
REPLAY_CAPACITY = 6_553_600
REPLAY_BATCH_SIZE = 32_768
TRANSITIONS_PER_UPDATE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
COMPILE = True
EVAL_INTERVAL_STEPS = 640_000
SEED = 1
ENV_FPS = 50.0
WANDB_PROJECT = "latency-sensitive-bench"
WANDB_GROUP = "humanoidbench-fasttd3"
WANDB_JOB_TYPE = "sample_factory"
WANDB_TAGS = ["humanoidbench", "fasttd3"]


def _task_env_config(task_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = build_probe_report(task_name)
    env_config = deepcopy(build_env_config(task_name, report))
    env_config["make_kwargs"]["render_mode"] = None
    return env_config, report


def _value_support(task_name: str) -> tuple[float, float]:
    return (-2000.0, 2000.0) if task_name == "h1hand_basketball" else (-250.0, 250.0)


def _train_steps(task_name: str, override: int | None) -> int:
    return TASK_TRAIN_ENV_STEPS[task_name] if override is None else override


def build_train_config(
    *,
    task_name: str,
    env_config: dict[str, Any],
    train_dir: Path,
    train_output_dir: Path,
    run_name: str,
    train_for_env_steps: int,
) -> dict[str, Any]:
    config = load_config(BASE_TRAIN_CONFIG)
    config["experiment"]["name"] = run_name
    config["experiment"]["seed"] = SEED

    config["backend"]["algo"] = "FAST_TD3"
    config["backend"]["device"] = "gpu"
    config["backend"]["train_dir"] = str(train_dir)
    config["backend"]["run_mode"] = "train"
    config["backend"]["restart_behavior"] = "resume"
    config["backend"]["actor_worker_gpus"] = []

    config["env"] = env_config
    config["env"]["task_name"] = task_name

    training = config["training"]
    training.update(
        {
            "train_for_env_steps": train_for_env_steps,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "worker_num_splits": WORKER_NUM_SPLITS,
            "rollout": ROLLOUT,
            "batch_size": COLLECTION_BATCH_SIZE,
            "fasttd3_replay_capacity": REPLAY_CAPACITY,
            "fasttd3_replay_batch_size": REPLAY_BATCH_SIZE,
            "fasttd3_transitions_per_update": TRANSITIONS_PER_UPDATE,
            "fasttd3_compile": COMPILE,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "recurrence": 1,
            "num_epochs": 1,
            "num_batches_per_epoch": 1,
            "async_rl": False,
            "batched_sampling": False,
            "use_rnn": False,
            "actor_critic_share_weights": True,
            "adaptive_stddev": False,
            "normalize_input": False,
            "normalize_returns": False,
            "reward_scale": 1.0,
            "reward_clip": 1e9,
            "fasttd3_v_min": _value_support(task_name)[0],
            "fasttd3_v_max": _value_support(task_name)[1],
        }
    )

    config["evaluation"]["eval_interval_steps"] = EVAL_INTERVAL_STEPS
    config["evaluation"]["eval_suites"] = {"fixed": [[0]]}
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
        experiment_family="humanoidbench_fasttd3",
        env_name=task_name,
        sweep_name="default",
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
            "experiment_family": "humanoidbench_fasttd3",
            "sweep_name": "default",
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


def generate_fasttd3(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tasks: Sequence[str] = DEFAULT_TASKS,
    train_for_env_steps: int | None = None,
) -> dict[str, Any]:
    output_root = output_root.expanduser()
    generated_root = output_root / "generated"
    train_dir = output_root / "checkpoints"
    rows: list[dict[str, str]] = []
    jobs: list[dict[str, Any]] = []

    for task_name in tasks:
        env_config, report = _task_env_config(task_name)
        task_root = output_root / task_name
        run_name = f"humanoidbench_fasttd3_{task_name}"
        train_config = build_train_config(
            task_name=task_name,
            env_config=env_config,
            train_dir=train_dir,
            train_output_dir=task_root / "train",
            run_name=run_name,
            train_for_env_steps=_train_steps(task_name, train_for_env_steps),
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
                "train_for_env_steps": train_config["training"]["train_for_env_steps"],
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
        "experiment_family": "humanoidbench_fasttd3",
        "tasks": list(tasks),
        "train_for_env_steps": train_for_env_steps,
        "task_train_for_env_steps": {
            task_name: _train_steps(task_name, train_for_env_steps) for task_name in tasks
        },
        "num_workers": NUM_WORKERS,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "rollout": ROLLOUT,
        "collection_batch_size": COLLECTION_BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "replay_batch_size": REPLAY_BATCH_SIZE,
        "transitions_per_update": TRANSITIONS_PER_UPDATE,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "compile": COMPILE,
        "eval_interval_steps": EVAL_INTERVAL_STEPS,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_job_type": WANDB_JOB_TYPE,
        "wandb_tags": WANDB_TAGS,
        "manifest": str(manifest_path),
        "runner_command": (
            f"{shlex.quote(sys.executable)} -m scripts.experiment_runner.run_manifest_jobs "
            f"--manifest {shlex.quote(str(manifest_path))} --gpus 0,1"
        ),
        "jobs": jobs,
    }
    (generated_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", dest="tasks", action="append", choices=DEFAULT_TASKS)
    parser.add_argument("--train-for-env-steps", type=int)
    args = parser.parse_args(argv)
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    summary = generate_fasttd3(
        output_root=args.output_root,
        tasks=tasks,
        train_for_env_steps=args.train_for_env_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
