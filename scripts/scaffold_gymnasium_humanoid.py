from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


ACTION_LABELS = [
    "abdomen_y_torque",
    "abdomen_z_torque",
    "abdomen_x_torque",
    "right_hip_x_torque",
    "right_hip_z_torque",
    "right_hip_y_torque",
    "right_knee_torque",
    "left_hip_x_torque",
    "left_hip_z_torque",
    "left_hip_y_torque",
    "left_knee_torque",
    "right_shoulder1_torque",
    "right_shoulder2_torque",
    "right_elbow_torque",
    "left_shoulder1_torque",
    "left_shoulder2_torque",
    "left_elbow_torque",
]
NOOP_ACTION = [0.0] * len(ACTION_LABELS)
JOINT_LABELS = [
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]
BODY_LABELS = [
    "world",
    "torso",
    "lower_waist",
    "pelvis",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_upper_arm",
    "right_lower_arm",
    "left_upper_arm",
    "left_lower_arm",
]
QPOS_LABELS = [
    "torso_z",
    "torso_quaternion_w",
    "torso_quaternion_x",
    "torso_quaternion_y",
    "torso_quaternion_z",
    *(f"{joint}_angle" for joint in JOINT_LABELS),
]
QVEL_LABELS = [
    "torso_x_velocity",
    "torso_y_velocity",
    "torso_z_velocity",
    "torso_angular_velocity_x",
    "torso_angular_velocity_y",
    "torso_angular_velocity_z",
    *(f"{joint}_angular_velocity" for joint in JOINT_LABELS),
]
STATE_LABELS = [
    *QPOS_LABELS,
    *QVEL_LABELS,
    *(f"cinert_{body}_{component}" for body in BODY_LABELS for component in range(10)),
    *(f"cvel_{body}_{component}" for body in BODY_LABELS for component in range(6)),
    *(f"qfrc_actuator_{index}" for index in range(len(QVEL_LABELS))),
    *(f"cfrc_ext_{body}_{component}" for body in BODY_LABELS for component in range(6)),
]


def build_humanoid_probe_report(*, env_id: str, render_mode: str) -> dict:
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
            "source": "native_humanoid_box",
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
    parser = argparse.ArgumentParser(description="Probe Humanoid's native latency-sensitive contract.")
    parser.add_argument("--env-id", default="Humanoid-v4")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    report = build_humanoid_probe_report(env_id=args.env_id, render_mode=args.render_mode)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.env_yaml_path is not None:
        config = yaml.safe_load(Path("configs/examples/gymnasium/humanoid/small_model_train.yaml").read_text())
        args.env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        args.env_yaml_path.write_text(yaml.safe_dump({"env": config["env"]}, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
