from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


GYMNASIUM_HOPPER_ENV_ID = "LatencyBench/HopperDiscrete-v0"


class HopperDiscreteActionEnv(gym.Wrapper):
    """Task-owned finite torque catalog over Hopper's native Box action."""

    def __init__(
        self,
        *,
        base_env_id: str,
        render_mode: str,
        action_values: list[list[float]],
        base_make_kwargs: dict[str, Any],
    ) -> None:
        env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
        super().__init__(env)
        self.action_values = action_values
        self.action_space = Discrete(len(action_values))

    def step(self, action):
        native_action = np.asarray(
            self.action_values[action],
            dtype=self.env.action_space.dtype,
        )
        return self.env.step(native_action)


def make_latency_sensitive_hopper_env(
    *,
    base_env_id: str,
    render_mode: str,
    action_values: list[list[float]],
    base_make_kwargs: dict[str, Any],
) -> HopperDiscreteActionEnv:
    return HopperDiscreteActionEnv(
        base_env_id=base_env_id,
        render_mode=render_mode,
        action_values=action_values,
        base_make_kwargs=base_make_kwargs,
    )


gym.register(
    id=GYMNASIUM_HOPPER_ENV_ID,
    entry_point=make_latency_sensitive_hopper_env,
)
