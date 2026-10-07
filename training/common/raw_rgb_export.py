from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer


class RawRgbFrameStackInfoWrapper(gym.Wrapper):
    """Attach recent raw RGB frames to info for teacher dataset export."""

    def __init__(self, env: gym.Env, *, num_frames: int, raw_frame_fn: Callable[[gym.Env], Any]):
        super().__init__(env)
        self.raw_frame_fn = raw_frame_fn
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=num_frames)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = self.render_game_frame()
        self._raw_rgb_frames.reset(frame)
        info = info.copy()
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frame_stack()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self._raw_rgb_frames.append(self.render_game_frame())
        info = info.copy()
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frame_stack()
        return observation, reward, terminated, truncated, info

    def render_game_frame(self) -> np.ndarray:
        return np.asarray(self.raw_frame_fn(self.env))

    def _raw_rgb_frame_stack(self) -> np.ndarray:
        return self._raw_rgb_frames.stack()
