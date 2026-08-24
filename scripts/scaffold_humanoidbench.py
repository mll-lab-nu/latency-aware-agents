"""Probe H1Hand Balance Hard and emit its frozen Gymnasium contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml


ENV_ID = "LatencyBench/H1HandBalanceHard-v0"
NATIVE_ENV_ID = "h1hand-balance_hard-v0"
TASK_NAME = "h1hand_balance_hard"
REGISTRATION_IMPORTS = ["latency_bench.envs.humanoidbench"]
MAKE_KWARGS = {"render_mode": "rgb_array", "width": 256, "height": 256}
BASE_PROMPT = (
    "Keep the H1Hand humanoid balanced on the rolling support board. "
    "Maintain an upright torso and avoid falling or contacting the floor."
)


def actuator_labels(env: gym.Env) -> list[str]:
    model = env.unwrapped.model
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    ]


def nominal_stance_action(env: gym.Env) -> np.ndarray:
    native_env = env.unwrapped
    keyframe_data = mujoco.MjData(native_env.model)
    mujoco.mj_resetDataKeyframe(native_env.model, keyframe_data, native_env.keyframe)
    mujoco.mj_forward(native_env.model, keyframe_data)
    return (
        2.0
        * (keyframe_data.actuator_length - native_env.action_low)
        / (native_env.action_high - native_env.action_low)
        - 1.0
    ).astype(env.action_space.dtype)


def build_probe_report() -> dict[str, Any]:
    for module_name in REGISTRATION_IMPORTS:
        importlib.import_module(module_name)

    env = gym.make(ENV_ID, **MAKE_KWARGS)
    try:
        native_env = env.unwrapped
        noop_action = nominal_stance_action(env)
        observation, reset_info = env.reset(seed=0)
        next_observation, reward, terminated, truncated, step_info = env.step(noop_action)
        rendered_frame = env.render()
        resolved_spec = env.spec
        entry_point = resolved_spec.entry_point
        return {
            "source": "runtime_api_probe",
            "versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "humanoid-bench",
                    "gymnasium",
                    "mujoco",
                    "dm-control",
                    "jax",
                    "jaxlib",
                    "sample-factory",
                    "numpy",
                    "torch",
                    "imageio",
                )
            },
            "env_id": ENV_ID,
            "native_env_id": NATIVE_ENV_ID,
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
                "labels": actuator_labels(env),
            },
            "noop_action": noop_action.tolist(),
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
                "shape": list(rendered_frame.shape),
                "dtype": str(rendered_frame.dtype),
            },
            "timing": {
                "dt_seconds": native_env.dt,
                "frame_skip": native_env.frame_skip,
                "model_timestep_seconds": native_env.model.opt.timestep,
                "native_env_fps": 1.0 / native_env.dt,
            },
            "make_kwargs": MAKE_KWARGS,
        }
    finally:
        env.close()


def build_env_config(report: dict[str, Any]) -> dict[str, Any]:
    action_space = report["action_space"]
    env_fps = report["timing"]["native_env_fps"]
    return {
        "name": "gymnasium",
        "task_name": TASK_NAME,
        "env_id": ENV_ID,
        "registration_imports": REGISTRATION_IMPORTS,
        "make_kwargs": MAKE_KWARGS,
        "env_fps": env_fps,
        "obs_fps": env_fps,
        "frame_stack": 1,
        "action_space": {
            "type": "box",
            "labels": action_space["labels"],
            "low": action_space["low"],
            "high": action_space["high"],
            "dtype": action_space["dtype"],
        },
        "noop_action": report["noop_action"],
        "base_prompt": BASE_PROMPT,
    }


def write_outputs(
    *,
    report: dict[str, Any],
    env_config: dict[str, Any],
    report_path: Path | None,
    env_yaml_path: Path | None,
) -> None:
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if env_yaml_path is not None:
        env_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        env_yaml_path.write_text(
            yaml.safe_dump({"env": env_config}, sort_keys=False),
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)

    report = build_probe_report()
    env_config = build_env_config(report)
    write_outputs(
        report=report,
        env_config=env_config,
        report_path=args.report_path,
        env_yaml_path=args.env_yaml_path,
    )
    print(json.dumps({"probe_report": report, "env_config": env_config}, indent=2))


if __name__ == "__main__":
    main()
