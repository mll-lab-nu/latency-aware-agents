"""Render a seeded Asterix reset frame and one pass through the action catalog."""

from pathlib import Path

import gymnasium as gym
from PIL import Image

import latency_bench.envs.tasks.asterix
from latency_bench.core.config import load_config


def main() -> None:
    output_dir = Path("/workspace/asterix-adapt/media")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/examples/gymnasium/asterix/small_model_train.yaml")
    env = gym.make(config["env"]["env_id"], **config["env"]["make_kwargs"])
    env.reset(seed=0)
    Image.fromarray(env.render()).save(output_dir / "reset.png")
    frames = []
    env.reset(seed=7)
    for action_id in [0, 1, 2, 3, 4, 5, 6, 7, 8] * 8:
        _, _, terminated, truncated, _ = env.step(action_id)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    frames[0].save(
        output_dir / "action_catalog_trace.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    env.close()


if __name__ == "__main__":
    main()
