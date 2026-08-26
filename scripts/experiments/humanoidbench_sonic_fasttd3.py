"""Generate FastTD3 + SONIC G1 10M experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.envs.humanoidbench_sonic import G1_SONIC_ENV_SPECS
from latency_bench.envs.humanoidbench_sonic_assets import (
    SONIC_ASSETS_REPO_ID,
    SONIC_ASSETS_REVISION,
    prepare_g1_sonic_assets,
    prepare_g1_sonic_reference_assets,
)
from latency_bench.envs.humanoidbench_sonic_reference import write_reference_tokens
from scripts.experiment_runner.job_manifest import (
    MANIFEST_V2_COLUMNS,
    SCHEMA_VERSION,
    build_job_id,
    write_manifest,
)

BASE_TRAIN_CONFIG = REPO_ROOT / (
    "configs/examples/gymnasium/h1hand_balance_hard/small_model_train.yaml"
)
SONIC_REGISTRATION_IMPORT = "latency_bench.envs.humanoidbench_sonic"
SONIC_TOKEN_DIM = 64
SONIC_HAND_ACTION_LABELS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
SONIC_ACTION_DIM = SONIC_TOKEN_DIM + len(SONIC_HAND_ACTION_LABELS)
SONIC_RESIDUAL_SCALE = 1.0 / 16.0
SONIC_RESIDUAL_VARIANT = "sonic64_residual"
SONIC_RESIDUAL_TASK = "g1_sonic_hurdle"

NUM_WORKERS = 128
NUM_ENVS_PER_WORKER = 1
COLLECTION_BATCH_SIZE = 128
SEED = 1
ENV_FPS = 50.0
WANDB_PROJECT = "latency-sensitive-bench"
WANDB_JOB_TYPE = "sample_factory"

# The 10M comparison uses one G1 scene/controller for both policies.  The
# raw policy emits body targets plus hand controls; the SONIC policy emits
# tokens plus hand controls and decodes only its first 64 actor outputs.
G1_10M_EXPERIMENT_FAMILY = "G1-SONIC-10M"
G1_10M_OUTPUT_ROOT = REPO_ROOT / "results/humanoidbench_g1_sonic_fasttd3_10m"
G1_10M_WANDB_GROUP = "G1-SONIC-10M"
G1_10M_WANDB_TAGS = ["humanoidbench", "g1", "fasttd3", "g1-sonic-10m"]
G1_10M_TRAIN_ENV_STEPS = 10_000_000
G1_10M_VARIANTS = ("raw43", "sonic78", SONIC_RESIDUAL_VARIANT)
G1_10M_TASKS = (
    "g1_sonic_hurdle",
    "g1_sonic_run",
    "g1_sonic_balance_hard",
    "g1_sonic_basketball",
)
G1_TASK_PROMPTS = {
    "g1_sonic_balance_hard": "Keep the G1 humanoid balanced on the rolling support board.",
    "g1_sonic_hurdle": "Move the G1 humanoid forward over the hurdles without falling.",
    "g1_sonic_run": "Run forward while remaining upright.",
    "g1_sonic_basketball": "Catch the basketball and throw it through the hoop without falling.",
}
G1_BODY_ACTION_LABELS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_RAW_ACTION_LABELS = G1_BODY_ACTION_LABELS + SONIC_HAND_ACTION_LABELS
G1_SONIC_ACTION_LABELS = [
    f"sonic_token_{index}" for index in range(SONIC_TOKEN_DIM)
] + SONIC_HAND_ACTION_LABELS
G1_SONIC_RESIDUAL_ACTION_LABELS = [
    f"sonic_residual_{index}" for index in range(SONIC_TOKEN_DIM)
]


def _keyframe_joint_targets(scene_path: Path, joint_names: list[str]) -> list[float]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    positions = model.key_qpos[0, model.jnt_qposadr[joint_ids]]
    ranges = model.jnt_range[joint_ids]
    return (
        2.0 * (positions - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1.0
    ).tolist()


def build_g1_env_config(
    *,
    task_name: str,
    variant: str,
    scene_path: Path,
    decoder_path: Path,
    base_token_path: Path | None = None,
    render_mode: str | None = None,
) -> dict[str, Any]:
    spec = G1_SONIC_ENV_SPECS[task_name]
    sonic = variant == "sonic78"
    residual = variant == SONIC_RESIDUAL_VARIANT
    if residual:
        action_dim = SONIC_TOKEN_DIM
        action_labels = G1_SONIC_RESIDUAL_ACTION_LABELS
        noop_action = [0.0] * SONIC_TOKEN_DIM
        control_mode = "sonic_token_residual"
    elif sonic:
        action_dim = SONIC_ACTION_DIM
        action_labels = G1_SONIC_ACTION_LABELS
        noop_action = [0.0] * SONIC_TOKEN_DIM + _keyframe_joint_targets(
            scene_path, SONIC_HAND_ACTION_LABELS
        )
        control_mode = "sonic_token"
    else:
        action_dim = len(G1_RAW_ACTION_LABELS)
        action_labels = G1_RAW_ACTION_LABELS
        noop_action = _keyframe_joint_targets(scene_path, G1_RAW_ACTION_LABELS)
        control_mode = "raw_body_target"
    make_kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "width": 256,
        "height": 256,
        "scene_path": str(scene_path),
        "control_mode": control_mode,
    }
    if sonic or residual:
        make_kwargs.update(
            {
                "sonic_decoder_path": str(decoder_path),
                "sonic_decoder_mode": "inference_worker"
                if render_mode is None
                else "environment",
            }
        )
    if residual:
        make_kwargs.update(
            {
                "sonic_base_token_path": base_token_path.as_posix(),
                "sonic_residual_scale": SONIC_RESIDUAL_SCALE,
            }
        )
    return {
        "name": "gymnasium",
        "task_name": task_name,
        "env_id": spec["env_id"],
        "registration_imports": [SONIC_REGISTRATION_IMPORT],
        "make_kwargs": make_kwargs,
        "env_fps": ENV_FPS,
        "obs_fps": ENV_FPS,
        "frame_stack": 1,
        "action_space": {
            "type": "box",
            "labels": action_labels,
            "low": [-1.0] * action_dim,
            "high": [1.0] * action_dim,
            "dtype": "float32",
        },
        "noop_action": noop_action,
        "base_prompt": G1_TASK_PROMPTS[task_name],
    }


def build_g1_train_config(
    *,
    task_name: str,
    variant: str,
    env_config: dict[str, Any],
    train_dir: Path,
    train_output_dir: Path,
    run_name: str,
    wandb_name: str,
    train_for_env_steps: int,
) -> dict[str, Any]:
    config = load_config(BASE_TRAIN_CONFIG)
    config["experiment"]["name"] = run_name
    config["experiment"]["seed"] = SEED
    config["backend"].update(
        {
            "algo": "FAST_TD3",
            "device": "gpu",
            "train_dir": str(train_dir),
            "run_mode": "train",
            "restart_behavior": "restart",
            "actor_worker_gpus": [],
        }
    )
    config["env"] = deepcopy(env_config)
    training = config["training"]
    training.update(
        {
            "train_for_env_steps": train_for_env_steps,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "worker_num_splits": 1,
            "rollout": 1,
            "batch_size": COLLECTION_BATCH_SIZE,
            "fasttd3_replay_capacity": 6_553_600,
            "fasttd3_replay_batch_size": 32_768,
            "fasttd3_transitions_per_update": 64,
            "fasttd3_compile": True,
            "learning_rate": 3e-4,
            "gamma": 0.99,
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
            "fasttd3_v_min": -2000.0 if task_name == "g1_sonic_basketball" else -250.0,
            "fasttd3_v_max": 2000.0 if task_name == "g1_sonic_basketball" else 250.0,
        }
    )
    if variant in ("sonic78", SONIC_RESIDUAL_VARIANT):
        training["fasttd3_sonic_decoder_path"] = env_config["make_kwargs"][
            "sonic_decoder_path"
        ]
    if variant == SONIC_RESIDUAL_VARIANT:
        training["fasttd3_sonic_residual_scale"] = SONIC_RESIDUAL_SCALE
    config["evaluation"]["eval_interval_steps"] = None
    config["logging"].update(
        {
            "output_dir": str(train_output_dir),
            "wandb_project": WANDB_PROJECT,
            "wandb_group": G1_10M_WANDB_GROUP,
            "wandb_name": wandb_name,
            "wandb_job_type": WANDB_JOB_TYPE,
            "wandb_tags": G1_10M_WANDB_TAGS + [task_name, variant],
        }
    )
    config["logging"]["video"]["enabled"] = False
    return config


def build_g1_eval_config(
    *,
    train_config: dict[str, Any],
    variant: str,
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
    if variant in ("sonic78", SONIC_RESIDUAL_VARIANT):
        config["training"].pop("fasttd3_sonic_decoder_path")
        config["env"]["make_kwargs"]["sonic_decoder_mode"] = "environment"
    config["env"]["make_kwargs"]["render_mode"] = "rgb_array"
    config["logging"].update(
        {
            "output_dir": str(eval_output_dir),
            "wandb_project": WANDB_PROJECT,
            "wandb_group": G1_10M_WANDB_GROUP,
            "wandb_job_type": WANDB_JOB_TYPE,
            "wandb_tags": G1_10M_WANDB_TAGS,
        }
    )
    config["logging"]["video"]["enabled"] = True
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False
    return config


def _g1_manifest_row(
    *,
    task_name: str,
    variant: str,
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
        experiment_family=G1_10M_EXPERIMENT_FAMILY,
        env_name=f"{task_name}_{variant}",
        sweep_name=variant,
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
            "experiment_family": G1_10M_EXPERIMENT_FAMILY,
            "sweep_name": variant,
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
            "wandb_group": G1_10M_WANDB_GROUP,
            "wandb_name": train_config["logging"]["wandb_name"],
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


def generate_g1_10m(
    *,
    output_root: Path = G1_10M_OUTPUT_ROOT,
    tasks: Sequence[str] = G1_10M_TASKS,
    variants: Sequence[str] = G1_10M_VARIANTS,
    train_for_env_steps: int = G1_10M_TRAIN_ENV_STEPS,
) -> dict[str, Any]:
    """Generate the same-scene G1 baselines and Hurdle residual job."""

    output_root = output_root.expanduser()
    asset_root, decoder_path = prepare_g1_sonic_assets(output_root)
    base_token_path = None
    if SONIC_RESIDUAL_TASK in tasks and SONIC_RESIDUAL_VARIANT in variants:
        planner_path, encoder_path = prepare_g1_sonic_reference_assets(output_root)
        base_token_path = output_root / "reference" / "hurdle_run_tokens.npy"
        base_token_path.parent.mkdir(parents=True, exist_ok=True)
        write_reference_tokens(
            planner_path,
            encoder_path,
            base_token_path,
            G1_SONIC_ENV_SPECS[SONIC_RESIDUAL_TASK]["horizon"],
        )
    scene_paths = {
        task_name: asset_root
        / G1_SONIC_ENV_SPECS[task_name]["scene_filename"].removeprefix("sonic_g1/")
        for task_name in tasks
    }
    generated_root = output_root / "generated"
    train_dir = output_root / "checkpoints"
    rows: list[dict[str, str]] = []
    jobs: list[dict[str, Any]] = []
    for task_name in tasks:
        spec = G1_SONIC_ENV_SPECS[task_name]
        for variant in variants:
            if variant == SONIC_RESIDUAL_VARIANT and task_name != SONIC_RESIDUAL_TASK:
                continue
            run_name = (
                f"g1_sonic_{task_name.removeprefix('g1_sonic_')}_{variant}_s1"
            )
            env_config = build_g1_env_config(
                task_name=task_name,
                variant=variant,
                scene_path=scene_paths[task_name],
                decoder_path=decoder_path,
                base_token_path=base_token_path,
            )
            task_root = output_root / task_name / variant
            train_config = build_g1_train_config(
                task_name=task_name,
                variant=variant,
                env_config=env_config,
                train_dir=train_dir,
                train_output_dir=task_root / "train",
                run_name=run_name,
                wandb_name=f"{output_root.name}_{run_name}",
                train_for_env_steps=train_for_env_steps,
            )
            eval_config = build_g1_eval_config(
                train_config=train_config,
                variant=variant,
                eval_output_dir=task_root / "eval_best_l0",
                horizon=spec["horizon"],
            )
            config_root = generated_root / task_name / variant
            config_root.mkdir(parents=True, exist_ok=True)
            train_path = config_root / "train.yaml"
            eval_path = config_root / "eval.yaml"
            train_path.write_text(
                yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8"
            )
            eval_path.write_text(
                yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8"
            )
            row = _g1_manifest_row(
                task_name=task_name,
                variant=variant,
                run_name=run_name,
                train_path=train_path,
                eval_path=eval_path,
                train_config=train_config,
                eval_config=eval_config,
            )
            rows.append(row)
            jobs.append(
                {
                    "task_name": task_name,
                    "variant": variant,
                    "train_for_env_steps": train_for_env_steps,
                    "horizon": spec["horizon"],
                    "train_config": str(train_path),
                    "eval_config": str(eval_path),
                    "checkpoint_dir": row["checkpoint_dir"],
                    "train_output_dir": row["train_output_dir"],
                    "eval_output_dir": row["eval_output_dir"],
                }
            )
    generated_root.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_root / "manifest.tsv"
    write_manifest(manifest_path, rows)
    summary = {
        "experiment_family": G1_10M_EXPERIMENT_FAMILY,
        "tasks": list(tasks),
        "variants": list(variants),
        "train_for_env_steps": train_for_env_steps,
        "num_workers": NUM_WORKERS,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "collection_batch_size": COLLECTION_BATCH_SIZE,
        "replay_capacity": 6_553_600,
        "replay_batch_size": 32_768,
        "transitions_per_update": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "fasttd3_compile": True,
        "periodic_evaluation": False,
        "final_eval_episodes": 10,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": G1_10M_WANDB_GROUP,
        "wandb_tags": G1_10M_WANDB_TAGS,
        "sonic_decoder_path": str(decoder_path),
        "sonic_reference_token_path": (
            None if base_token_path is None else str(base_token_path)
        ),
        "sonic_assets_repo_id": SONIC_ASSETS_REPO_ID,
        "sonic_assets_revision": SONIC_ASSETS_REVISION,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=G1_10M_OUTPUT_ROOT)
    parser.add_argument("--train-for-env-steps", type=int)
    parser.add_argument("--task", action="append", choices=G1_10M_TASKS)
    parser.add_argument("--variant", action="append", choices=G1_10M_VARIANTS)
    args = parser.parse_args(argv)
    summary = generate_g1_10m(
        output_root=args.output_root,
        tasks=tuple(args.task) if args.task else G1_10M_TASKS,
        variants=tuple(args.variant) if args.variant else G1_10M_VARIANTS,
        train_for_env_steps=(
            G1_10M_TRAIN_ENV_STEPS
            if args.train_for_env_steps is None
            else args.train_for_env_steps
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
