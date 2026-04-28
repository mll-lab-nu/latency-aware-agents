from __future__ import annotations

from typing import List, Optional

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from PIL import Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import flappy_bird_gymnasium  # noqa: F401

from envs.base_wrappers import EnvFrameStack, QueueLatencyWrapper


class RenderToImageWrapper(gym.Wrapper):
    """Converts env render output to a fixed-size (H,W,C) uint8 image observation."""

    def __init__(self, env: gym.Env, image_size: int = 84, grayscale: bool = False):
        super().__init__(env)
        self.image_size = int(image_size)
        self.grayscale = bool(grayscale)
        channels = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.image_size, self.image_size, channels),
            dtype=np.uint8,
        )

    def _convert(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        img = img.convert("L" if self.grayscale else "RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        out = np.asarray(img, dtype=np.uint8)
        if self.grayscale:
            out = out[..., None]
        return out

    def _obs(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("render() returned None — ensure render_mode='rgb_array'")
        return self._convert(frame)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(), float(reward), terminated, truncated, info


def make_demo_env(
    latency: int = 0,
    image_size: int = 84,
    stack_frames: int = 4,
    seed: int = 0,
) -> DummyVecEnv:
    """
    Builds a single Flappy Bird env matching the training setup (image_only, RGB).
    Returns a DummyVecEnv (n_envs=1) compatible with SB3's PPO.predict().

    Obs shape: (1, 84, 84, 12)  — 3 RGB channels × 4 stacked frames.
    """

    def _init():
        env = gym.make(
            "FlappyBird-v0",
            use_lidar=False,
            normalize_obs=False,
            render_mode="rgb_array",
        )
        env = QueueLatencyWrapper(env, frame_skip_values=[latency], seed=seed)
        env = RenderToImageWrapper(env, image_size=image_size, grayscale=False)
        if stack_frames > 1:
            env = EnvFrameStack(env, n_stack=stack_frames)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])
