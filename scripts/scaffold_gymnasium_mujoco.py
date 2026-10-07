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

from latency_bench.envs.tasks.inverted_pendulum import (
    GYMNASIUM_MUJOCO_INVERTED_PENDULUM_ENV_ID,
)


USER_APPROVED_FORCE_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]
USER_APPROVED_ACTION_LABELS = [
    "push_left_hard",
    "push_left_soft",
    "zero_force",
    "push_right_soft",
    "push_right_hard",
]


def build_inverted_pendulum_probe_report(
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
        "mujoco_version": importlib.metadata.version("mujoco"),
        "env_id": base_env_id,
        "resolved_spec": {
            "id": env.spec.id,
            "entry_point": str(env.spec.entry_point),
            "max_episode_steps": env.spec.max_episode_steps,
            "kwargs": env.spec.kwargs,
        },
        "unwrapped_class": (
            f"{type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}"
        ),
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
        "unwrapped_frame_skip": env.unwrapped.frame_skip,
        "model_timestep_seconds": env.unwrapped.model.opt.timestep,
        "native_env_fps": 1.0 / env.unwrapped.dt,
        "render_mode": render_mode,
        "base_make_kwargs": base_make_kwargs,
        "action_catalog": {
            "source": "user_approved",
            "note": "Gymnasium exposes a continuous Box action; the five force values are a task design decision, not API-derived semantics.",
            "labels": USER_APPROVED_ACTION_LABELS,
            "force_values": USER_APPROVED_FORCE_VALUES,
            "noop_action": "zero_force",
        },
    }
    env.close()
    return report


def build_inverted_pendulum_env_config(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
    env_fps: float,
) -> dict[str, Any]:
    return {
        "env": {
            "name": "gymnasium",
            "task_name": "inverted_pendulum_discrete",
            "env_id": GYMNASIUM_MUJOCO_INVERTED_PENDULUM_ENV_ID,
            "registration_imports": ["latency_bench.envs.tasks.inverted_pendulum"],
            "make_kwargs": {
                "base_env_id": base_env_id,
                "render_mode": render_mode,
                "force_values": USER_APPROVED_FORCE_VALUES,
                "base_make_kwargs": base_make_kwargs,
            },
            "env_fps": env_fps,
            "obs_fps": env_fps,
            "frame_stack": 1,
            "noop_action": "zero_force",
            "action_map": dict(zip(USER_APPROVED_ACTION_LABELS, range(5))),
            "action_order": USER_APPROVED_ACTION_LABELS,
            "base_prompt": (
                "Keep the pole upright by applying horizontal cart forces. Choose exactly one action from: "
                "push_left_hard, push_left_soft, zero_force, push_right_soft, push_right_hard."
            ),
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the pinned InvertedPendulum API and emit the user-approved finite-force task contract."
    )
    parser.add_argument("--base-env-id", default="InvertedPendulum-v4")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--base-make-kwargs-json", default="{}")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_inverted_pendulum_probe_report(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
    )
    config = build_inverted_pendulum_env_config(
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
