from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


ACTION_LABELS = ["thigh_torque", "leg_torque", "foot_torque"]
STATE_LABELS = [
    "torso_height",
    "torso_angle",
    "thigh_angle",
    "leg_angle",
    "foot_angle",
    "torso_x_velocity",
    "torso_z_velocity",
    "torso_angular_velocity",
    "thigh_angular_velocity",
    "leg_angular_velocity",
    "foot_angular_velocity",
]
NOOP_ACTION = [0.0] * 3


def build_hopper_probe_report(*, env_id: str, render_mode: str, base_make_kwargs: dict) -> dict:
    env = gym.make(env_id, render_mode=render_mode, **base_make_kwargs)
    observation, reset_info = env.reset(seed=0)
    rendered_frame = env.render()
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, reward, terminated, truncated, step_info = env.step(zero_action)
    report = {
        "source": "runtime_api_probe",
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "mujoco_version": importlib.metadata.version("mujoco"),
        "env_id": env_id,
        "resolved_spec": {
            "id": env.spec.id,
            "entry_point": str(env.spec.entry_point),
            "max_episode_steps": env.spec.max_episode_steps,
            "kwargs": env.spec.kwargs,
        },
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
        "action_labels": ACTION_LABELS,
        "state_labels": STATE_LABELS,
        "noop_action": NOOP_ACTION,
    }
    env.close()
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe Hopper's RGB+state latency-sensitive contract.")
    parser.add_argument("--env-id", default="Hopper-v4")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path("configs/examples/gymnasium/hopper_rgb_state/small_model_train.yaml").read_text())
    report = build_hopper_probe_report(
        env_id=args.env_id,
        render_mode=args.render_mode,
        base_make_kwargs=config["env"]["make_kwargs"]["base_make_kwargs"],
    )
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.env_yaml_path is not None:
        args.env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        args.env_yaml_path.write_text(yaml.safe_dump({"env": config["env"]}, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
