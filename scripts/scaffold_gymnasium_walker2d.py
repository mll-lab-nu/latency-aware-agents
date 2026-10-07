from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from latency_bench.envs.tasks.walker2d import GYMNASIUM_WALKER2D_ENV_ID


ACTION_LABELS = [
    "right_thigh_torque",
    "right_leg_torque",
    "right_foot_torque",
    "left_thigh_torque",
    "left_leg_torque",
    "left_foot_torque",
]
NOOP_ACTION = [0.0] * 6


def build_walker2d_probe_report(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
) -> dict[str, Any]:
    env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
    observation, reset_info = env.reset(seed=0)
    rendered_frame = env.render()
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, reward, terminated, truncated, step_info = env.step(zero_action)
    report = {
        "source": "runtime_api_probe",
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "mujoco_version": importlib.metadata.version("mujoco"),
        "env_id": base_env_id,
        "resolved_spec": {
            "id": env.spec.id,
            "entry_point": str(env.spec.entry_point),
            "max_episode_steps": env.spec.max_episode_steps,
            "kwargs": env.spec.kwargs,
        },
        "unwrapped_class": f"{type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}",
        "metadata": env.metadata,
        "action_space": {
            "type": type(env.action_space).__name__,
            "shape": list(env.action_space.shape),
            "low": env.action_space.low.tolist(),
            "high": env.action_space.high.tolist(),
            "dtype": str(env.action_space.dtype),
        },
        "observation_space": {
            "type": type(env.observation_space).__name__,
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
        },
        "reset_observation_shape": list(observation.shape),
        "reset_observation_dtype": str(observation.dtype),
        "reset_info_keys": sorted(reset_info),
        "zero_action_step": {
            "action": zero_action.tolist(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info_keys": sorted(step_info),
        },
        "render": {"shape": list(rendered_frame.shape), "dtype": str(rendered_frame.dtype)},
        "unwrapped_dt_seconds": env.unwrapped.dt,
        "unwrapped_frame_skip": env.unwrapped.frame_skip,
        "model_timestep_seconds": env.unwrapped.model.opt.timestep,
        "native_env_fps": 1.0 / env.unwrapped.dt,
        "render_mode": render_mode,
        "base_make_kwargs": base_make_kwargs,
        "action_catalog": {
            "source": "native_walker2d_box",
            "note": "The policy and OpenVLA both use the same six native continuous torques.",
            "labels": ACTION_LABELS,
            "low": env.action_space.low.tolist(),
            "high": env.action_space.high.tolist(),
            "noop_action": NOOP_ACTION,
        },
    }
    env.close()
    return report


def build_walker2d_env_config(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
    env_fps: float,
) -> dict[str, Any]:
    return {
        "name": "gymnasium",
        "task_name": "walker2d_continuous_torque",
        "env_id": GYMNASIUM_WALKER2D_ENV_ID,
        "registration_imports": ["latency_bench.envs.tasks.walker2d"],
        "make_kwargs": {
            "base_env_id": base_env_id,
            "render_mode": render_mode,
            "base_make_kwargs": base_make_kwargs,
        },
        "env_fps": env_fps,
        "obs_fps": env_fps,
        "frame_stack": 1,
        "action_space": {
            "type": "box",
            "labels": ACTION_LABELS,
            "low": [-1.0] * 6,
            "high": [1.0] * 6,
            "dtype": "float32",
        },
        "noop_action": NOOP_ACTION,
        "base_prompt": (
            "Move the Walker2d robot forward while keeping its torso upright. Predict six continuous "
            "torques in [-1, 1] ordered as right thigh, right leg, right foot, left thigh, left leg, "
            "and left foot."
        ),
    }


def write_configs(config_dir: Path, env_config: dict[str, Any]) -> None:
    source_dir = Path("configs/examples/gymnasium/hopper")
    configs = {
        name: yaml.safe_load((source_dir / name).read_text())
        for name in (
            "small_model_train.yaml",
            "small_model_eval.yaml",
            "latency_bench_eval.yaml",
        )
    }
    train = configs["small_model_train.yaml"]
    train["experiment"]["name"] = "gymnasium_walker2d_continuous_zero_latency"
    train["backend"]["train_dir"] = "/mnt/checkpoints/latency-sensitive-bench/small_models/walker2d"
    train["env"] = deepcopy(env_config)
    train["latency"] = {"method": "zero", "add_latency_info": False}
    train["policy"].pop("actions", None)
    train["training"]["train_for_env_steps"] = 10_000_000
    train["logging"]["output_dir"] = "/mnt/data/latency-sensitive-bench/walker2d/train"

    small_eval = configs["small_model_eval.yaml"]
    small_eval["policy"]["checkpoint_path"] = ""
    small_eval["evaluation"]["eval_episodes"] = 10
    small_eval["evaluation"]["eval_latency_values"] = [0, 1, 2, 3, 4]
    small_eval["logging"]["output_dir"] = "/mnt/data/latency-sensitive-bench/walker2d/continuous_small_model_eval"

    bench_eval = configs["latency_bench_eval.yaml"]
    bench_eval["experiment"]["name"] = "gymnasium_walker2d_continuous_latency_bench_eval"
    bench_eval["evaluation"]["eval_episodes"] = 10
    bench_eval["evaluation"]["eval_latency_values"] = [0, 1, 2, 3, 4]
    bench_eval["logging"]["output_dir"] = "/mnt/data/latency-sensitive-bench/walker2d/latency_bench_eval"

    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "small_model_train.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
    for name, config in (
        ("small_model_eval.yaml", small_eval),
        ("latency_bench_eval.yaml", bench_eval),
    ):
        (config_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe Walker2d and generate its latency-sensitive contract.")
    parser.add_argument("--base-env-id", default="Walker2d-v4")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument(
        "--base-make-kwargs-json",
        default=(
            '{"forward_reward_weight":1.0,"ctrl_cost_weight":0.001,"healthy_reward":1.0,'
            '"terminate_when_unhealthy":true,"reset_noise_scale":0.005,'
            '"exclude_current_positions_from_observation":true}'
        ),
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)
    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_walker2d_probe_report(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
    )
    env_config = build_walker2d_env_config(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
        env_fps=report["native_env_fps"],
    )
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.env_yaml_path is not None:
        args.env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        args.env_yaml_path.write_text(yaml.safe_dump({"env": env_config}, sort_keys=False))
    if args.config_dir is not None:
        write_configs(args.config_dir, env_config)
    print(json.dumps({"probe_report": report, "env_config": env_config}, indent=2))


if __name__ == "__main__":
    main()
