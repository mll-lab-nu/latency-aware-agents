from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


GYMNASIUM_WALKER2D_ENV_ID = "LatencyBench/Walker2dContinuous-v0"


class Walker2dContinuousActionEnv(gym.Wrapper):
    """Task-owned pass-through wrapper for Walker2d's six native torques."""

    def __init__(
        self,
        *,
        base_env_id: str,
        render_mode: str,
        base_make_kwargs: dict[str, Any],
    ) -> None:
        env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
        super().__init__(env)
    def step(self, action):
        native_action = np.asarray(action, dtype=self.env.action_space.dtype)
        applied_action = np.clip(
            native_action,
            self.env.action_space.low,
            self.env.action_space.high,
        )
        observation, reward, terminated, truncated, info = self.env.step(applied_action)
        info = dict(info)
        info["applied_action"] = applied_action.tolist()
        return observation, reward, terminated, truncated, info


def make_latency_sensitive_walker2d_env(
    *,
    base_env_id: str,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
) -> Walker2dContinuousActionEnv:
    return Walker2dContinuousActionEnv(
        base_env_id=base_env_id,
        render_mode=render_mode,
        base_make_kwargs=base_make_kwargs,
    )


gym.register(
    id=GYMNASIUM_WALKER2D_ENV_ID,
    entry_point=make_latency_sensitive_walker2d_env,
)
