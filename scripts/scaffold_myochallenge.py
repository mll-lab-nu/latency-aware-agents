"""Probe a registered MyoSuite environment and emit its Gymnasium contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml


def build_myochallenge_probe_report(
    *,
    benchmark_env_id: str,
    base_env_id: str,
    registration_imports: Sequence[str],
    base_make_kwargs: dict[str, Any],
    camera_name: str,
    render_width: int,
    render_height: int,
    noop_action: Sequence[float],
) -> dict[str, Any]:
    """Probe one MyoSuite registration and return runtime facts."""
    for module_name in registration_imports:
        importlib.import_module(module_name)

    env = gym.make(
        benchmark_env_id,
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
        native_action = np.asarray(noop_action, dtype=env.action_space.dtype)
        next_observation, reward, terminated, truncated, step_info = env.step(native_action)
        native_env = env.unwrapped
        model = native_env.sim.model
        action_labels = [model.actuator(index).name for index in range(model.nu)]
        resolved_spec = env.spec
        entry_point = resolved_spec.entry_point
        return {
            "source": "runtime_api_probe",
            "myosuite_version": importlib.metadata.version("MyoSuite"),
            "mujoco_version": importlib.metadata.version("mujoco"),
            "env_id": base_env_id,
            "benchmark_env_id": benchmark_env_id,
            "resolved_spec": {
                "id": resolved_spec.id,
                "entry_point": f"{entry_point.__module__}.{entry_point.__qualname__}",
                "max_episode_steps": resolved_spec.max_episode_steps,
                "kwargs": resolved_spec.kwargs,
            },
            "unwrapped_class": f"{type(native_env).__module__}.{type(native_env).__name__}",
            "metadata": env.metadata,
            "action_space": {
                "type": type(env.action_space).__name__,
                "shape": list(env.action_space.shape),
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
                "dtype": str(env.action_space.dtype),
                "labels": action_labels,
            },
            "noop_action": native_action.tolist(),
            "observation_space": {
                "type": type(env.observation_space).__name__,
                "shape": list(env.observation_space.shape),
                "dtype": str(env.observation_space.dtype),
            },
            "reset": {
                "observation_shape": list(observation.shape),
                "observation_dtype": str(observation.dtype),
                "info_keys": sorted(reset_info),
            },
            "step": {
                "observation_shape": list(next_observation.shape),
                "observation_dtype": str(next_observation.dtype),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info_keys": sorted(step_info),
            },
            "render": {
                "camera": camera_name,
                "shape": list(rendered_frame.shape),
                "dtype": str(rendered_frame.dtype),
            },
            "timing": {
                "dt_seconds": native_env.dt,
                "frame_skip": native_env.frame_skip,
                "model_timestep_seconds": native_env.sim.model.opt.timestep,
                "native_env_fps": 1.0 / native_env.dt,
            },
            "native_episode_horizon": env.get_wrapper_attr("native_episode_horizon"),
        }
    finally:
        env.close()


def build_myochallenge_env_config(
    *,
    report: dict[str, Any],
    task_name: str,
    benchmark_env_id: str,
    base_env_id: str,
    registration_imports: Sequence[str],
    base_make_kwargs: dict[str, Any],
    camera_name: str,
    render_width: int,
    render_height: int,
    base_prompt: str,
    env_fps: float,
    obs_fps: float,
    frame_stack: int,
) -> dict[str, Any]:
    """Build a complete ``env`` mapping from one completed API probe."""
    action_space = report["action_space"]
    return {
        "name": "gymnasium",
        "task_name": task_name,
        "env_id": benchmark_env_id,
        "registration_imports": list(registration_imports),
        "make_kwargs": {
            "base_env_id": base_env_id,
            "render_mode": "rgb_array",
            "base_make_kwargs": base_make_kwargs,
            "camera_name": camera_name,
            "render_width": render_width,
            "render_height": render_height,
        },
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "frame_stack": frame_stack,
        "action_space": {
            "type": "box",
            "labels": action_space["labels"],
            "low": action_space["low"],
            "high": action_space["high"],
            "dtype": action_space["dtype"],
        },
        "noop_action": report["noop_action"],
        "base_prompt": base_prompt,
    }


def write_myochallenge_outputs(
    *,
    report: dict[str, Any],
    env_config: dict[str, Any],
    report_path: Path | None,
    env_yaml_path: Path | None,
) -> None:
    """Write optional probe artifacts in the format used by the runbook."""
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if env_yaml_path is not None:
        env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        env_yaml_path.write_text(yaml.safe_dump({"env": env_config}, sort_keys=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-env-id", required=True)
    parser.add_argument("--base-env-id", required=True)
    parser.add_argument("--registration-import", action="append", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--noop-action-json", required=True)
    parser.add_argument("--env-fps", type=float, required=True)
    parser.add_argument("--obs-fps", type=float, required=True)
    parser.add_argument("--frame-stack", type=int, required=True)
    parser.add_argument("--base-make-kwargs-json", default="{}")
    parser.add_argument("--camera-name", default="default")
    parser.add_argument("--render-width", type=int, default=224)
    parser.add_argument("--render-height", type=int, default=224)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)

    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    noop_action = json.loads(args.noop_action_json)
    report = build_myochallenge_probe_report(
        benchmark_env_id=args.benchmark_env_id,
        base_env_id=args.base_env_id,
        registration_imports=args.registration_import,
        base_make_kwargs=base_make_kwargs,
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
        noop_action=noop_action,
    )
    env_config = build_myochallenge_env_config(
        report=report,
        task_name=args.task_name,
        benchmark_env_id=args.benchmark_env_id,
        base_env_id=args.base_env_id,
        registration_imports=args.registration_import,
        base_make_kwargs=base_make_kwargs,
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
        base_prompt=args.base_prompt,
        env_fps=args.env_fps,
        obs_fps=args.obs_fps,
        frame_stack=args.frame_stack,
    )
    write_myochallenge_outputs(
        report=report,
        env_config=env_config,
        report_path=args.report_path,
        env_yaml_path=args.env_yaml_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
