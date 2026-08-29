"""Generate the SONIC 29D physical-residual FastTD3 H025 experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.envs.humanoidbench_sonic import (
    G1_CONTROL_MODE_SONIC_COMPOSED,
    G1_SONIC_ENV_SPECS,
)
from latency_bench.envs.humanoidbench_sonic_contract import G1_BODY_JOINT_NAMES
from latency_bench.envs.humanoidbench_sonic_assets import (
    SONIC_ASSETS_REPO_ID,
    SONIC_ASSETS_REVISION,
    prepare_g1_sonic_assets,
    prepare_g1_sonic_encoder,
    prepare_g1_sonic_planner,
    prepare_g1_sonic_skill_references,
)
from latency_bench.envs.humanoidbench_sonic_reference import (
    SONIC_REFERENCE_LOOKAHEAD,
    canonicalize_reference_heading,
    generate_planner_qpos,
    load_bones_seed_reference_qpos,
    resample_qpos_to_50hz,
    straighten_reference_heading,
)
from scripts.experiment_runner.job_manifest import (
    MANIFEST_V2_COLUMNS,
    SCHEMA_VERSION,
    build_job_id,
    write_manifest,
)

BASE_TRAIN_CONFIG = REPO_ROOT / (
    "configs/examples/gymnasium/g1_sonic_hurdle/fasttd3_train.yaml"
)
SONIC_REGISTRATION_IMPORT = "latency_bench.envs.humanoidbench_sonic"
SONIC_PHYSICAL_RESIDUAL_DIM = len(G1_BODY_JOINT_NAMES)
SONIC_VARIANT = "sonic29d_fullres_anchor_h025_1m"
HURDLE_TASK = "g1_sonic_hurdle"
SINGLE_HURDLE_HORIZON = 400
ACTIVE_HURDLES = 1
RUN_REFERENCE_HORIZON = 1500
RUN_TARGET_VELOCITY = 3.0
HURDLE_HEIGHT = 0.25
TRAIN_FOR_ENV_STEPS = 1_000_000
EVAL_INTERVAL_STEPS = 100_000

NUM_WORKERS = 128
NUM_ENVS_PER_WORKER = 1
COLLECTION_BATCH_SIZE = 128
SEED = 1
ENV_FPS = 50.0
WANDB_PROJECT = "latency-sensitive-bench"
WANDB_JOB_TYPE = "sample_factory"

G1_EXPERIMENT_FAMILY = "G1-SONIC-29D-STRAIGHT-REFERENCE-ACTOR-ANCHOR-H025-1M"
G1_OUTPUT_ROOT = REPO_ROOT / (
    "results/humanoidbench_g1_sonic29d_fullres_anchor_h025_1m"
)
G1_SONIC_ASSET_CACHE = REPO_ROOT / "results/humanoidbench_g1_sonic_assets"
G1_WANDB_GROUP = "G1-SONIC-29D-STRAIGHT-REFERENCE-ACTOR-ANCHOR-H025-1M"
G1_WANDB_TAGS = [
    "humanoidbench",
    "g1",
    "hurdle",
    "fasttd3",
    "sonic-physical-residual",
    SONIC_VARIANT,
]


def _write_reference_bundle(
    planner_path: Path,
    skill_paths: dict[str, Path],
    scene_path: Path,
    output_path: Path,
) -> None:
    planner = ort.InferenceSession(
        str(planner_path), providers=["CPUExecutionProvider"]
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    body_qpos_addresses = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in G1_BODY_JOINT_NAMES]
    )
    foot_site_ids = [model.site(name).id for name in ("left_foot", "right_foot")]
    run_qpos = straighten_reference_heading(
        resample_qpos_to_50hz(
            generate_planner_qpos(
                planner,
                RUN_REFERENCE_HORIZON,
                target_velocity=RUN_TARGET_VELOCITY,
            ),
            RUN_REFERENCE_HORIZON + SONIC_REFERENCE_LOOKAHEAD + 1,
        )
    )
    jump_qpos, jump_num_tokens = load_bones_seed_reference_qpos(
        skill_paths["obstacle_jump"]
    )
    references = (
        ("run", run_qpos, RUN_REFERENCE_HORIZON + 1),
        (
            "jump",
            canonicalize_reference_heading(jump_qpos),
            jump_num_tokens,
        ),
    )
    arrays: dict[str, np.ndarray] = {}
    for bundle_name, qpos, num_tokens in references:
        reference_qpos = qpos[:num_tokens]
        arrays[f"{bundle_name}_reference_qpos"] = reference_qpos
        arrays[f"{bundle_name}_encoder_qpos"] = qpos
        if bundle_name == "run":
            foot_contacts = []
            for frame in reference_qpos:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.qpos[:7] = frame[:7]
                data.qpos[body_qpos_addresses] = frame[7:]
                mujoco.mj_forward(model, data)
                foot_contacts.append(data.site_xpos[foot_site_ids, 2] <= 0.02)
            arrays["run_reference_foot_contacts"] = np.asarray(foot_contacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)


def build_g1_env_config(
    *,
    scene_path: Path,
    decoder_path: Path,
    encoder_path: Path,
    reference_path: Path,
    hurdle_height: float,
    render_mode: str | None = None,
) -> dict[str, Any]:
    action_labels = [f"{name}_residual" for name in G1_BODY_JOINT_NAMES]
    action_low = [-1.0] * SONIC_PHYSICAL_RESIDUAL_DIM
    action_high = [1.0] * SONIC_PHYSICAL_RESIDUAL_DIM
    noop_action = [0.0] * SONIC_PHYSICAL_RESIDUAL_DIM
    control_mode = G1_CONTROL_MODE_SONIC_COMPOSED

    make_kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "width": 256,
        "height": 256,
        "scene_path": str(scene_path),
        "control_mode": control_mode,
        "hurdle_height": hurdle_height,
        "hurdle_count": ACTIVE_HURDLES,
        "hurdle_horizon": SINGLE_HURDLE_HORIZON,
    }
    make_kwargs.update(
        {
            "sonic_decoder_path": str(decoder_path),
            "sonic_encoder_path": str(encoder_path),
            "sonic_decoder_mode": (
                "inference_worker" if render_mode is None else "environment"
            ),
            "sonic_reference_bundle_path": str(reference_path),
        }
    )

    spec = G1_SONIC_ENV_SPECS[HURDLE_TASK]
    return {
        "name": "gymnasium",
        "task_name": HURDLE_TASK,
        "env_id": spec["env_id"],
        "registration_imports": [SONIC_REGISTRATION_IMPORT],
        "make_kwargs": make_kwargs,
        "env_fps": ENV_FPS,
        "obs_fps": ENV_FPS,
        "frame_stack": 1,
        "action_space": {
            "type": "box",
            "labels": action_labels,
            "low": action_low,
            "high": action_high,
            "dtype": "float32",
        },
        "noop_action": noop_action,
        "base_prompt": "Move the G1 humanoid over the hurdle without contact or falling.",
    }


def build_g1_train_config(
    *,
    env_config: dict[str, Any],
    train_dir: Path,
    train_output_dir: Path,
    run_name: str,
    wandb_name: str,
    train_for_env_steps: int,
    restart_behavior: str,
    gamma: float = 0.999,
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
            "restart_behavior": restart_behavior,
            "actor_worker_gpus": [],
        }
    )
    config["env"] = deepcopy(env_config)
    config["training"].update(
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
            "gamma": gamma,
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
            "fasttd3_v_min": -300.0,
            "fasttd3_v_max": 300.0,
            "fasttd3_actor_action_l2": 1.0,
        }
    )
    config["training"]["fasttd3_sonic_decoder_path"] = env_config["make_kwargs"][
        "sonic_decoder_path"
    ]
    config["evaluation"].update(
        {
            "eval_interval_steps": EVAL_INTERVAL_STEPS,
            "eval_episodes": 50,
            "eval_max_steps": SINGLE_HURDLE_HORIZON,
            "eval_suites": {"fixed": [[0]]},
        }
    )
    config["logging"].update(
        {
            "output_dir": str(train_output_dir),
            "wandb_project": WANDB_PROJECT,
            "wandb_group": G1_WANDB_GROUP,
            "wandb_name": wandb_name,
            "wandb_job_type": WANDB_JOB_TYPE,
            "wandb_tags": G1_WANDB_TAGS,
        }
    )
    config["logging"]["video"]["enabled"] = False
    return config


def build_g1_eval_config(
    *,
    train_config: dict[str, Any],
    eval_output_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(train_config)
    config["backend"]["run_mode"] = "eval"
    config["backend"]["extra_args"] = ["--load_checkpoint_kind", "latest"]
    config["policy"] = {
        "type": "gymnasium_sf",
        "checkpoint_kind": "latest",
        "device": "cuda",
        "deterministic": True,
    }
    config["evaluation"] = {
        "eval_episodes": 50,
        "eval_parallel_envs": 1,
        "eval_max_steps": SINGLE_HURDLE_HORIZON,
        "eval_deterministic": True,
        "eval_latency_values": [0],
    }
    config["training"].pop("fasttd3_sonic_decoder_path")
    config["env"]["make_kwargs"]["sonic_decoder_mode"] = "environment"
    config["env"]["make_kwargs"]["render_mode"] = "rgb_array"
    config["logging"].update(
        {
            "output_dir": str(eval_output_dir),
            "wandb_job_type": WANDB_JOB_TYPE,
            "wandb_tags": G1_WANDB_TAGS,
        }
    )
    config["logging"]["video"]["enabled"] = True
    config["logging"]["save_step_records"] = True
    config["logging"]["save_action_records"] = True
    config["logging"]["save_latency_records"] = False
    return config


def _manifest_row(
    *,
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
        experiment_family=G1_EXPERIMENT_FAMILY,
        env_name=f"{HURDLE_TASK}_{SONIC_VARIANT}",
        sweep_name=SONIC_VARIANT,
        latency_label="0",
        frame_stack=1,
        obs_fps=ENV_FPS,
        observation_stride=1,
        seed=SEED,
    )
    row = {column: "" for column in MANIFEST_V2_COLUMNS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "env": HURDLE_TASK,
            "experiment_family": G1_EXPERIMENT_FAMILY,
            "sweep_name": SONIC_VARIANT,
            "latency_label": "0",
            "seed": str(SEED),
            "frame_stack": "1",
            "obs_fps": str(int(ENV_FPS)),
            "observation_stride": "1",
            "train_config": str(train_path),
            "eval_kind": "sample_factory_zero_latency",
            "eval_config": str(eval_path),
            "train_command": (
                f"{shlex.quote(sys.executable)} -m training.train --config "
                f"{shlex.quote(str(train_path))}"
            ),
            "eval_command": (
                f"{shlex.quote(sys.executable)} -m latency_bench.run --config "
                f"{shlex.quote(str(eval_path))}"
            ),
            "run_name": run_name,
            "wandb_group": G1_WANDB_GROUP,
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


def generate_g1_experiment(
    *,
    output_root: Path = G1_OUTPUT_ROOT,
    gamma: float = 0.999,
) -> dict[str, Any]:
    output_root = output_root.expanduser()
    asset_root, decoder_path = prepare_g1_sonic_assets(G1_SONIC_ASSET_CACHE)
    encoder_path = prepare_g1_sonic_encoder(G1_SONIC_ASSET_CACHE)
    planner_path = prepare_g1_sonic_planner(G1_SONIC_ASSET_CACHE)
    skill_paths = prepare_g1_sonic_skill_references(G1_SONIC_ASSET_CACHE)
    scene_path = asset_root / G1_SONIC_ENV_SPECS[HURDLE_TASK][
        "scene_filename"
    ].removeprefix("sonic_g1/")
    reference_path = output_root / "reference" / "hurdle_run_jump.npz"
    _write_reference_bundle(
        planner_path, skill_paths, scene_path, reference_path
    )

    generated_root = output_root / "generated"
    train_dir = output_root / "checkpoints"
    run_name = f"g1_sonic_hurdle_{SONIC_VARIANT}_s{SEED}"
    env_config = build_g1_env_config(
        scene_path=scene_path,
        decoder_path=decoder_path,
        encoder_path=encoder_path,
        reference_path=reference_path,
        hurdle_height=HURDLE_HEIGHT,
    )
    task_root = output_root / HURDLE_TASK / SONIC_VARIANT
    train_config = build_g1_train_config(
        env_config=env_config,
        train_dir=train_dir,
        train_output_dir=task_root / "train",
        run_name=run_name,
        wandb_name=f"{output_root.name}_{run_name}",
        train_for_env_steps=TRAIN_FOR_ENV_STEPS,
        restart_behavior="restart",
        gamma=gamma,
    )
    eval_config = build_g1_eval_config(
        train_config=train_config,
        eval_output_dir=task_root / "eval_latest_l0",
    )
    config_root = generated_root / SONIC_VARIANT
    config_root.mkdir(parents=True, exist_ok=True)
    train_path = config_root / "train.yaml"
    eval_path = config_root / "eval.yaml"
    train_path.write_text(
        yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8"
    )
    eval_path.write_text(
        yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8"
    )
    row = _manifest_row(
        run_name=run_name,
        train_path=train_path,
        eval_path=eval_path,
        train_config=train_config,
        eval_config=eval_config,
    )
    job = {
        "task_name": HURDLE_TASK,
        "variant": SONIC_VARIANT,
        "hurdle_height": HURDLE_HEIGHT,
        "train_for_env_steps": TRAIN_FOR_ENV_STEPS,
        "train_config": str(train_path),
        "eval_config": str(eval_path),
        "checkpoint_dir": row["checkpoint_dir"],
        "train_output_dir": row["train_output_dir"],
        "eval_output_dir": row["eval_output_dir"],
        "train_command": row["train_command"],
        "eval_command": row["eval_command"],
    }

    generated_root.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_root / "manifest.tsv"
    write_manifest(manifest_path, [row])
    summary = {
        "experiment_family": G1_EXPERIMENT_FAMILY,
        "task": HURDLE_TASK,
        "variant": SONIC_VARIANT,
        "hurdle_height": HURDLE_HEIGHT,
        "train_for_env_steps": TRAIN_FOR_ENV_STEPS,
        "eval_interval_steps": EVAL_INTERVAL_STEPS,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "collection_batch_size": COLLECTION_BATCH_SIZE,
        "replay_capacity": 6_553_600,
        "replay_batch_size": 32_768,
        "transitions_per_update": 64,
        "learning_rate": 3e-4,
        "gamma": gamma,
        "fasttd3_compile": True,
        "final_eval_episodes": 50,
        "sonic_decoder_path": str(decoder_path),
        "sonic_planner_path": str(planner_path),
        "sonic_reference_bundle_path": str(reference_path),
        "sonic_assets_repo_id": SONIC_ASSETS_REPO_ID,
        "sonic_assets_revision": SONIC_ASSETS_REVISION,
        "manifest": str(manifest_path),
        "jobs": [job],
    }
    (generated_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=G1_OUTPUT_ROOT)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args(argv)
    summary = generate_g1_experiment(output_root=args.output_root, gamma=args.gamma)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
