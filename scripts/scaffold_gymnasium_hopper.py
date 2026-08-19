from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from latency_bench.envs.gymnasium_hopper import GYMNASIUM_HOPPER_ENV_ID


TORQUE_LEVELS = [-1.0, 0.0, 1.0]
LEVEL_LABELS = {-1.0: "n", 0.0: "z", 1.0: "p"}
USER_APPROVED_ACTION_VALUES = [list(values) for values in itertools.product(TORQUE_LEVELS, repeat=3)]
USER_APPROVED_ACTION_LABELS = [
    f"t_{LEVEL_LABELS[thigh]}_l_{LEVEL_LABELS[leg]}_f_{LEVEL_LABELS[foot]}"
    for thigh, leg, foot in USER_APPROVED_ACTION_VALUES
]


def build_hopper_probe_report(
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
        "reset_observation": observation.tolist(),
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
            "source": "user_approved",
            "note": "Gymnasium exposes a three-dimensional continuous Box action; the 27 ternary torque vectors are a task design decision.",
            "labels": USER_APPROVED_ACTION_LABELS,
            "action_values": USER_APPROVED_ACTION_VALUES,
            "noop_action": "t_z_l_z_f_z",
        },
    }
    env.close()
    return report


def build_hopper_env_config(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
    env_fps: float,
) -> dict[str, Any]:
    return {
        "env": {
            "name": "gymnasium",
            "task_name": "hopper_discrete_ternary",
            "env_id": GYMNASIUM_HOPPER_ENV_ID,
            "registration_imports": ["latency_bench.envs.gymnasium_hopper"],
            "make_kwargs": {
                "base_env_id": base_env_id,
                "render_mode": render_mode,
                "action_values": USER_APPROVED_ACTION_VALUES,
                "base_make_kwargs": base_make_kwargs,
            },
            "env_fps": env_fps,
            "obs_fps": env_fps,
            "frame_stack": 1,
            "noop_action": "t_z_l_z_f_z",
            "action_map": dict(zip(USER_APPROVED_ACTION_LABELS, range(27))),
            "action_order": USER_APPROVED_ACTION_LABELS,
            "base_prompt": (
                "Move the hopper forward while keeping it upright. Choose exactly one action from "
                "the 27-action torque catalog; t, l, and f denote thigh, leg, and foot, while n, z, "
                "and p denote -1, 0, and +1 torque."
            ),
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the pinned Hopper API and emit the user-approved ternary torque contract."
    )
    parser.add_argument("--base-env-id", default="Hopper-v4")
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
    args = parser.parse_args(argv)
    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_hopper_probe_report(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
    )
    config = build_hopper_env_config(
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
