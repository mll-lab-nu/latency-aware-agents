"""Probe MyoChallenge Table Tennis P2 and emit its benchmark contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from latency_bench.envs.myochallenge_table_tennis import (
    GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
    MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE,
)


def build_myochallenge_table_tennis_probe_report(
    *,
    base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    base_make_kwargs: dict[str, Any],
    camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
    render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
) -> dict[str, Any]:
    env = gym.make(
        GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
        base_env_id=base_env_id,
        render_mode="rgb_array",
        base_make_kwargs=base_make_kwargs,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )
    try:
        observation, reset_info = env.reset(seed=0)
        rendered_frame = env.render()
        zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _, reward, terminated, truncated, step_info = env.step(zero_action)
        native_env = env.unwrapped
        model = native_env.sim.model
        action_labels = [model.actuator(index).name for index in range(model.nu)]
        entry_point = env.spec.entry_point
        return {
            "source": "runtime_api_probe",
            "myosuite_version": importlib.metadata.version("MyoSuite"),
            "mujoco_version": importlib.metadata.version("mujoco"),
            "env_id": base_env_id,
            "benchmark_env_id": GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
            "resolved_spec": {
                "id": env.spec.id,
                "entry_point": f"{entry_point.__module__}.{entry_point.__qualname__}",
                "max_episode_steps": env.spec.max_episode_steps,
                "kwargs": env.spec.kwargs,
            },
            "unwrapped_class": f"{type(native_env).__module__}.{type(native_env).__name__}",
            "metadata": env.metadata,
            "action_space": {
                "type": type(env.action_space).__name__,
                "shape": list(env.action_space.shape),
                "low_min": float(np.min(env.action_space.low)),
                "low_max": float(np.max(env.action_space.low)),
                "high_min": float(np.min(env.action_space.high)),
                "high_max": float(np.max(env.action_space.high)),
                "dtype": str(env.action_space.dtype),
                "labels": action_labels,
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
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "solved": bool(step_info["solved"]),
                "score": float(step_info["score"]),
                "info_keys": sorted(step_info),
            },
            "render": {
                "camera": camera_name,
                "shape": list(rendered_frame.shape),
                "dtype": str(rendered_frame.dtype),
            },
            "unwrapped_dt_seconds": native_env.dt,
            "unwrapped_frame_skip": native_env.frame_skip,
            "model_timestep_seconds": native_env.sim.model.opt.timestep,
            "native_env_fps": 1.0 / native_env.dt,
            "max_episode_steps": env.spec.max_episode_steps,
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe MyoChallenge 2025 Table Tennis P2.")
    parser.add_argument("--base-env-id", default=MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID)
    parser.add_argument("--base-make-kwargs-json", default="{}")
    parser.add_argument("--camera-name", default=MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA)
    parser.add_argument("--render-width", type=int, default=MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0])
    parser.add_argument("--render-height", type=int, default=MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1])
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    report = build_myochallenge_table_tennis_probe_report(
        base_env_id=args.base_env_id,
        base_make_kwargs=json.loads(args.base_make_kwargs_json),
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
