from __future__ import annotations

from bisect import insort_right
from collections import deque
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class QueueLatencyWrapper(gym.Wrapper):
    """
    Delays action application by `frame_skip_values[0]` frames.
    Action chosen at step t is applied at step t + latency.
    The env executes the most recently matured action (NOOP=0 until first action matures).
    """

    def __init__(self, env: gym.Env, frame_skip_values: List[int], seed: int = 0):
        super().__init__(env)
        assert len(frame_skip_values) > 0
        self.frame_skip_values = [max(0, int(x)) for x in frame_skip_values]
        self.rng = np.random.RandomState(seed)
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple] = []
        self._schedule_seq = 0

    def _sample_frame_skip(self) -> int:
        return int(self.rng.choice(self.frame_skip_values))

    def _schedule_action(self, action: int) -> int:
        latency = int(self.current_frame_skip)
        apply_step = self.env_step_idx + latency
        self._schedule_seq += 1
        insort_right(self.pending_actions, (apply_step, self._schedule_seq, int(action), latency))
        return apply_step

    def _advance_pending(self):
        while self.pending_actions and self.pending_actions[0][0] <= self.env_step_idx:
            _, _, action, _ = self.pending_actions.pop(0)
            self.current_applied_action = int(action)

    def reset(self, **kwargs):
        self.current_applied_action = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0
        self.current_frame_skip = self._sample_frame_skip()
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        info["queue_latency"] = int(self.current_frame_skip)
        return obs, info

    def step(self, action):
        self._schedule_action(int(action))
        self._advance_pending()
        obs, reward, terminated, truncated, info = self.env.step(int(self.current_applied_action))
        if not isinstance(info, dict):
            info = {}
        info["queue_latency"] = int(self.current_frame_skip)
        info["applied_action"] = int(self.current_applied_action)
        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


class EnvFrameStack(gym.Wrapper):
    """HWC frame stacking (used by Flappy and VizDoom)."""

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3
        self.n_stack = max(1, int(n_stack))
        h, w, c = self.observation_space.shape
        self.frames: deque = deque(maxlen=self.n_stack)
        self.observation_space = Box(low=0, high=255,
                                     shape=(h, w, c * self.n_stack), dtype=np.uint8)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        arr = np.asarray(obs, dtype=np.uint8)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(arr.copy())
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs, dtype=np.uint8))
        return self._get_obs(), reward, terminated, truncated, info


class NCHWFrameStack(gym.Wrapper):
    """NCHW frame stacking (used by Demon Attack / CleanRL)."""

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = int(n_stack)
        shape = self.observation_space.shape
        h, w = (shape[0], shape[1]) if len(shape) == 2 else (shape[0], shape[1])
        self.frames: deque = deque(maxlen=self.n_stack)
        self.observation_space = Box(low=0, high=255,
                                     shape=(self.n_stack, h, w), dtype=np.uint8)

    def _fmt(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    def _get_obs(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._fmt(obs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(frame.copy())
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._fmt(obs))
        return self._get_obs(), reward, terminated, truncated, info
