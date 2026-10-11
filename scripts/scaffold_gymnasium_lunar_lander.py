from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml


ACTION_LABELS = [
    "do_nothing",
    "fire_left_orientation_engine",
    "fire_main_engine",
    "fire_right_orientation_engine",
]


def build_lunar_lander_probe_report(
    *,
    env_id: str,
    make_kwargs: dict[str, Any],
) -> dict[str, Any]:
    env = gym.make(env_id, **make_kwargs)
    observation, reset_info = env.reset(seed=0)
    rendered_frame = env.render()
    _, reward, terminated, truncated, step_info = env.step(0)
    report = {
        "source": "runtime_api_probe",
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "box2d_version": importlib.metadata.version("box2d-py"),
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
            "n": env.action_space.n,
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
        "noop_step": {
            "action": 0,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info_keys": sorted(step_info),
        },
        "render": {
            "shape": list(rendered_frame.shape),
            "dtype": str(rendered_frame.dtype),
        },
        "native_env_fps": env.metadata["render_fps"],
        "make_kwargs": make_kwargs,
        "action_catalog": {
            "source": "gymnasium_api",
            "labels": ACTION_LABELS,
            "values": list(range(4)),
            "noop_action": "do_nothing",
        },
    }
    env.close()
    return report


def build_lunar_lander_env_config(
    *,
    env_id: str,
    make_kwargs: dict[str, Any],
    env_fps: int,
) -> dict[str, Any]:
    return {
        "env": {
            "name": "lunar_lander_discrete",
            "env_id": env_id,
            "registration_imports": [],
            "make_kwargs": make_kwargs,
            "env_fps": env_fps,
            "obs_fps": env_fps,
            "frame_stack": 1,
            "noop_action": "do_nothing",
            "action_map": dict(zip(ACTION_LABELS, range(4))),
            "action_order": ACTION_LABELS,
            "base_prompt": (
                "Land the spacecraft safely between the flags. Choose exactly one action from: "
                "do_nothing, fire_left_orientation_engine, fire_main_engine, "
                "fire_right_orientation_engine."
            ),
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the pinned discrete Lunar Lander API and emit its task contract."
    )
    parser.add_argument("--env-id", default="LunarLander-v2")
    parser.add_argument(
        "--make-kwargs-json",
        default=(
            '{"render_mode":"rgb_array","continuous":false,"gravity":-10.0,'
            '"enable_wind":false,"wind_power":15.0,"turbulence_power":1.5}'
        ),
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    make_kwargs = json.loads(args.make_kwargs_json)
    report = build_lunar_lander_probe_report(
        env_id=args.env_id,
        make_kwargs=make_kwargs,
    )
    config = build_lunar_lander_env_config(
        env_id=args.env_id,
        make_kwargs=make_kwargs,
        env_fps=report["native_env_fps"],
    )
    if args.report_path is not None:
        args.report_path.write_text(
            json.dumps(report, indent=2, default=lambda value: value.item()) + "\n"
        )
    if args.env_yaml_path is not None:
        args.env_yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        json.dumps(
            {"probe_report": report, "env_config": config},
            indent=2,
            default=lambda value: value.item(),
        )
    )


if __name__ == "__main__":
    main()
