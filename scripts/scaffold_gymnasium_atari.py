from __future__ import annotations

import argparse
import re
from collections.abc import Sequence

import gymnasium as gym
import yaml

from latency_bench.envs.gymnasium_atari import GYMNASIUM_ATARI_ENV_ID


def _task_name_from_ale_env_id(ale_env_id: str) -> str:
    game_name = ale_env_id.rsplit("/", 1)[-1].rsplit("-v", 1)[0]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", game_name).lower()


def build_atari_env_config(
    *,
    ale_env_id: str,
    task_name: str,
    env_fps: float | None,
    obs_fps: float | None,
    screen_size: int,
    noop_max: int,
    base_prompt: str | None,
) -> dict:
    env = gym.make(
        GYMNASIUM_ATARI_ENV_ID,
        ale_env_id=ale_env_id,
        render_mode="rgb_array",
        screen_size=screen_size,
        noop_max=noop_max,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        terminal_on_life_loss=False,
        max_num_frames_per_episode=108000,
        grayscale_obs=True,
        mode=1,
        difficulty=0,
    )
    env.reset(seed=0)
    raw_screen_height = env.unwrapped.ale.getScreenRGB().shape[0]
    action_labels = [meaning.lower() for meaning in env.unwrapped.get_action_meanings()]
    env.close()
    native_env_fps = env_fps if env_fps is not None else {210: 60.0, 250: 50.0}[raw_screen_height]
    decision_obs_fps = obs_fps if obs_fps is not None else native_env_fps / 4.0
    prompt = (
        base_prompt
        if base_prompt is not None
        else (
            f"Play {task_name}. Choose exactly one action from: "
            f"{', '.join(action_labels)}."
        )
    )
    return {
        "env": {
            "name": "gymnasium",
            "task_name": task_name,
            "env_id": GYMNASIUM_ATARI_ENV_ID,
            "registration_imports": ["latency_bench.envs.gymnasium_atari"],
            "make_kwargs": {
                "ale_env_id": ale_env_id,
                "render_mode": "rgb_array",
                "screen_size": screen_size,
                "noop_max": noop_max,
                "frameskip": 1,
                "repeat_action_probability": 0.0,
                "full_action_space": False,
                "terminal_on_life_loss": False,
                "max_num_frames_per_episode": 108000,
                "grayscale_obs": True,
                "mode": 1,
                "difficulty": 0,
            },
            "env_fps": native_env_fps,
            "obs_fps": decision_obs_fps,
            "frame_stack": 1,
            "noop_action": action_labels[0],
            "action_map": dict(zip(action_labels, range(len(action_labels)))),
            "action_order": action_labels,
            "base_prompt": prompt,
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a latency-sensitive Atari env block.")
    parser.add_argument("--ale-env-id", required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--env-fps", type=float)
    parser.add_argument("--obs-fps", type=float)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--base-prompt")
    args = parser.parse_args(argv)
    task_name = (
        args.task_name
        if args.task_name is not None
        else _task_name_from_ale_env_id(args.ale_env_id)
    )
    config = build_atari_env_config(
        ale_env_id=args.ale_env_id,
        task_name=task_name,
        env_fps=args.env_fps,
        obs_fps=args.obs_fps,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        base_prompt=args.base_prompt,
    )
    print(yaml.safe_dump(config, sort_keys=False), end="")


if __name__ == "__main__":
    main()
