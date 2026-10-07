from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


GYMNASIUM_MUJOCO_INVERTED_PENDULUM_ENV_ID = "LatencyBench/InvertedPendulumDiscrete-v0"


class InvertedPendulumDiscreteActionEnv(gym.Wrapper):
    """Task-owned finite force catalog over InvertedPendulum's native Box action."""

    def __init__(
        self,
        *,
        base_env_id: str,
        render_mode: str,
        force_values: list[float],
        base_make_kwargs: dict[str, Any],
    ) -> None:
        env = gym.make(
            base_env_id,
            render_mode=render_mode,
            **base_make_kwargs,
        )
        super().__init__(env)
        self.force_values = force_values
        self.action_space = Discrete(len(force_values))

    def step(self, action):
        force = self.force_values[action]
        native_action = np.asarray([force], dtype=self.env.action_space.dtype)
        return self.env.step(native_action)


def make_latency_sensitive_inverted_pendulum_env(
    *,
    base_env_id: str,
    render_mode: str,
    force_values: list[float],
    base_make_kwargs: dict[str, Any],
) -> InvertedPendulumDiscreteActionEnv:
    return InvertedPendulumDiscreteActionEnv(
        base_env_id=base_env_id,
        render_mode=render_mode,
        force_values=force_values,
        base_make_kwargs=base_make_kwargs,
    )


gym.register(
    id=GYMNASIUM_MUJOCO_INVERTED_PENDULUM_ENV_ID,
    entry_point=make_latency_sensitive_inverted_pendulum_env,
)
