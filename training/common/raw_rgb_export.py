from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np


ENV_RAW_RGB_FRAME_STACK_INFO_KEY = "env_raw_rgb_frame_stack"


class RawRgbFrameStackInfoWrapper(gym.Wrapper):
    """Attach recent raw RGB frames to info for teacher dataset export."""

    def __init__(self, env: gym.Env, *, num_frames: int, raw_frame_fn: Callable[[gym.Env], Any]):
        super().__init__(env)
        self.num_frames = int(num_frames)
        self.raw_frame_fn = raw_frame_fn
        self._raw_rgb_frames: deque[np.ndarray] = deque(maxlen=self.num_frames)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = self._current_raw_rgb_frame()
        self._raw_rgb_frames.clear()
        for _ in range(self.num_frames):
            self._raw_rgb_frames.append(frame.copy())
        info = dict(info or {})
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frame_stack()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self._raw_rgb_frames.append(self._current_raw_rgb_frame())
        info = dict(info or {})
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frame_stack()
        return observation, reward, terminated, truncated, info

    def _current_raw_rgb_frame(self) -> np.ndarray:
        return np.asarray(self.raw_frame_fn(self.env))

    def _raw_rgb_frame_stack(self) -> np.ndarray:
        return np.stack(tuple(self._raw_rgb_frames), axis=0)
