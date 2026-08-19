from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


GYMNASIUM_PENDULUM_ENV_ID = "LatencyBench/PendulumDiscrete-v0"


class PendulumDiscreteActionEnv(gym.Wrapper):
    """Task-owned finite torque catalog over Pendulum's native Box action."""

    def __init__(
        self,
        *,
        base_env_id: str,
        render_mode: str,
        torque_values: list[float],
        base_make_kwargs: dict[str, Any],
    ) -> None:
        env = gym.make(
            base_env_id,
            render_mode=render_mode,
            **base_make_kwargs,
        )
        super().__init__(env)
        self.torque_values = torque_values
        self.action_space = Discrete(len(torque_values))

    def step(self, action):
        torque = self.torque_values[action]
        native_action = np.asarray([torque], dtype=self.env.action_space.dtype)
        return self.env.step(native_action)


def make_latency_sensitive_pendulum_env(
    *,
    base_env_id: str,
    render_mode: str,
    torque_values: list[float],
    base_make_kwargs: dict[str, Any],
) -> PendulumDiscreteActionEnv:
    return PendulumDiscreteActionEnv(
        base_env_id=base_env_id,
        render_mode=render_mode,
        torque_values=torque_values,
        base_make_kwargs=base_make_kwargs,
    )


gym.register(
    id=GYMNASIUM_PENDULUM_ENV_ID,
    entry_point=make_latency_sensitive_pendulum_env,
)
