from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

from latency_bench.envs.gymnasium_air_raid import GYMNASIUM_AIR_RAID_ENV_ID


AIR_RAID_ENV_FPS = 50
ACTION_LABELS = ["noop", "fire", "right", "left", "rightfire", "leftfire"]


def build_air_raid_probe_report(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
) -> dict[str, Any]:
    env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
    observation, reset_info = env.reset(seed=0)
    rendered_frame = env.render()
    reset_episode_frame = env.unwrapped.ale.getEpisodeFrameNumber()
    _, reward, terminated, truncated, step_info = env.step(0)
    report = {
        "source": "runtime_api_probe",
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "ale_py_version": importlib.metadata.version("ale-py"),
        "env_id": base_env_id,
        "resolved_spec": {
            "id": env.spec.id,
            "entry_point": str(env.spec.entry_point),
            "max_episode_steps": env.spec.max_episode_steps,
            "kwargs": env.spec.kwargs,
        },
        "unwrapped_class": f"{type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}",
        "metadata": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in env.metadata.items()
        },
        "action_space": {
            "type": type(env.action_space).__name__,
            "n": env.action_space.n,
            "dtype": str(env.action_space.dtype),
            "meanings": env.unwrapped.get_action_meanings(),
        },
        "observation_space": {
            "type": type(env.observation_space).__name__,
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
        },
        "reset_observation_shape": list(observation.shape),
        "reset_observation_dtype": str(observation.dtype),
        "reset_info": reset_info,
        "noop_step": {
            "action": 0,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": step_info,
        },
        "render": {"shape": list(rendered_frame.shape), "dtype": str(rendered_frame.dtype)},
        "available_modes": env.unwrapped.ale.getAvailableModes(),
        "available_difficulties": env.unwrapped.ale.getAvailableDifficulties(),
        "reset_episode_frame_number": reset_episode_frame,
        "post_step_episode_frame_number": env.unwrapped.ale.getEpisodeFrameNumber(),
        "native_env_fps": AIR_RAID_ENV_FPS,
        "video_standard": "PAL",
        "clock_source": "AirRaid PAL ROM task contract",
        "render_mode": render_mode,
        "base_make_kwargs": base_make_kwargs,
        "action_catalog": {
            "source": "gymnasium_api",
            "labels": ACTION_LABELS,
            "values": list(range(6)),
            "noop_action": "noop",
        },
    }
    env.close()
    return report


def build_air_raid_env_config(
    *,
    base_env_id: str,
    render_mode: str,
    screen_size: int,
    noop_max: int,
    base_make_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "env": {
            "name": "gymnasium",
            "task_name": "air_raid",
            "env_id": GYMNASIUM_AIR_RAID_ENV_ID,
            "registration_imports": ["latency_bench.envs.gymnasium_air_raid"],
            "make_kwargs": {
                "base_env_id": base_env_id,
                "render_mode": render_mode,
                "screen_size": screen_size,
                "noop_max": noop_max,
                "base_make_kwargs": base_make_kwargs,
            },
            "env_fps": AIR_RAID_ENV_FPS,
            "obs_fps": AIR_RAID_ENV_FPS / 4,
            "frame_stack": 4,
            "noop_action": "noop",
            "action_map": dict(zip(ACTION_LABELS, range(6))),
            "action_order": ACTION_LABELS,
            "base_prompt": (
                "Protect both buildings from flying saucers. Choose exactly one action from: "
                "noop, fire, right, left, rightfire, leftfire."
            ),
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the pinned AirRaid API and emit its CHW policy-observation contract."
    )
    parser.add_argument("--base-env-id", default="ALE/AirRaid-v5")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=0)
    parser.add_argument(
        "--base-make-kwargs-json",
        default=(
            '{"obs_type":"rgb","frameskip":1,"repeat_action_probability":0.0,'
            '"full_action_space":false,"mode":1,"difficulty":0,'
            '"max_num_frames_per_episode":108000}'
        ),
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)
    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_air_raid_probe_report(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        base_make_kwargs=base_make_kwargs,
    )
    config = build_air_raid_env_config(
        base_env_id=args.base_env_id,
        render_mode=args.render_mode,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        base_make_kwargs=base_make_kwargs,
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
