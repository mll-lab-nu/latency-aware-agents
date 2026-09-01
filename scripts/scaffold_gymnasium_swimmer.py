from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from latency_bench.envs.gymnasium_swimmer import GYMNASIUM_SWIMMER_ENV_ID


ACTION_LABELS = ["first_rotor_torque", "second_rotor_torque"]
NOOP_ACTION = [0.0] * len(ACTION_LABELS)
STATE_LABELS = [
    "front_tip_angle",
    "first_rotor_angle",
    "second_rotor_angle",
    "front_tip_x_velocity",
    "front_tip_y_velocity",
    "front_tip_angular_velocity",
    "first_rotor_angular_velocity",
    "second_rotor_angular_velocity",
]


def build_swimmer_probe_report(*, env_id: str, render_mode: str) -> dict:
    env = gym.make(env_id, render_mode=render_mode)
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
        "action_catalog": {
            "source": "native_swimmer_box",
            "labels": ACTION_LABELS,
            "low": env.action_space.low.tolist(),
            "high": env.action_space.high.tolist(),
            "noop_action": NOOP_ACTION,
        },
        "state_labels": STATE_LABELS,
    }
    env.close()
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe Swimmer's native latency-sensitive contract.")
    parser.add_argument("--env-id", default=GYMNASIUM_SWIMMER_ENV_ID)
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    report = build_swimmer_probe_report(env_id=args.env_id, render_mode=args.render_mode)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.env_yaml_path is not None:
        config = yaml.safe_load(Path("configs/examples/gymnasium/swimmer/small_model_train.yaml").read_text())
        args.env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        args.env_yaml_path.write_text(yaml.safe_dump({"env": config["env"]}, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
