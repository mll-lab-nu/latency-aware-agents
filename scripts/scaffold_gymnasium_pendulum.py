from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from latency_bench.envs.tasks.pendulum import GYMNASIUM_PENDULUM_ENV_ID


USER_APPROVED_TORQUE_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
USER_APPROVED_ACTION_LABELS = [
    "torque_clockwise_hard",
    "torque_clockwise_soft",
    "zero_torque",
    "torque_counterclockwise_soft",
    "torque_counterclockwise_hard",
]


def build_pendulum_probe_report(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
) -> dict[str, Any]:
    env = gym.make(
        base_env_id,
        render_mode=render_mode,
        **base_make_kwargs,
    )
    observation, reset_info = env.reset(seed=0)
    rendered_frame = env.render()
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, reward, terminated, truncated, step_info = env.step(zero_action)
    report = {
        "source": "runtime_api_probe",
        "gymnasium_version": importlib.metadata.version("gymnasium"),
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
            "low": env.observation_space.low.tolist(),
            "high": env.observation_space.high.tolist(),
            "dtype": str(env.observation_space.dtype),
        },
        "reset_observation_shape": list(observation.shape),
        "reset_observation_dtype": str(observation.dtype),
        "reset_observation": observation.tolist(),
        "reset_info_keys": sorted(reset_info),
        "zero_action_step": {
            "action": zero_action.tolist(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info_keys": sorted(step_info),
        },
        "render": {
            "shape": list(rendered_frame.shape),
            "dtype": str(rendered_frame.dtype),
        },
        "unwrapped_dt_seconds": env.unwrapped.dt,
        "native_env_fps": 1.0 / env.unwrapped.dt,
        "max_torque": env.unwrapped.max_torque,
        "max_speed": env.unwrapped.max_speed,
        "gravity": env.unwrapped.g,
        "mass": env.unwrapped.m,
        "length": env.unwrapped.l,
        "render_mode": render_mode,
        "base_make_kwargs": base_make_kwargs,
        "action_catalog": {
            "source": "user_approved",
            "note": "Gymnasium exposes a continuous Box action; the five torque values are a task design decision, not API-derived semantics.",
            "labels": USER_APPROVED_ACTION_LABELS,
            "torque_values": USER_APPROVED_TORQUE_VALUES,
            "noop_action": "zero_torque",
        },
    }
    env.close()
    return report


def build_pendulum_env_config(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
    env_fps: float,
) -> dict[str, Any]:
    return {
        "env": {
            "name": "gymnasium",
            "task_name": "pendulum_discrete",
            "env_id": GYMNASIUM_PENDULUM_ENV_ID,
            "registration_imports": ["latency_bench.envs.tasks.pendulum"],
            "make_kwargs": {
                "base_env_id": base_env_id,
                "render_mode": render_mode,
                "torque_values": USER_APPROVED_TORQUE_VALUES,
                "base_make_kwargs": base_make_kwargs,
            },
            "env_fps": env_fps,
            "obs_fps": env_fps,
            "frame_stack": 1,
            "noop_action": "zero_torque",
            "action_map": dict(zip(USER_APPROVED_ACTION_LABELS, range(5))),
            "action_order": USER_APPROVED_ACTION_LABELS,
            "base_prompt": (
                "Swing the pendulum upright and keep it balanced by applying torque. "
                "Choose exactly one action from: torque_clockwise_hard, torque_clockwise_soft, "
                "zero_torque, torque_counterclockwise_soft, torque_counterclockwise_hard."
            ),
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the pinned Pendulum API and emit the user-approved finite-torque task contract."
    )
    parser.add_argument("--base-env-id", default="Pendulum-v1")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--base-make-kwargs-json", default='{"g": 10.0}')
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_pendulum_probe_report(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
    )
    config = build_pendulum_env_config(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
        env_fps=report["native_env_fps"],
    )
    if args.report_path is not None:
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.env_yaml_path is not None:
        args.env_yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"probe_report": report, "env_config": config}, indent=2))


if __name__ == "__main__":
    main()
