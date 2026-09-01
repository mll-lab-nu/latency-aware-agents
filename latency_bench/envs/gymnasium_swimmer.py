from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.spaces import Box


GYMNASIUM_SWIMMER_ENV_ID = "LatencyBench/SwimmerContinuous-v0"
SWIMMER_XML_PATH = Path(__file__).with_name("assets") / "swimmer_v4_mujoco3.xml"


class SwimmerV4Mujoco3Env(SwimmerEnv):
    """Swimmer-v4 with its removed MuJoCo 2 collision option omitted."""

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        observation_dim = 8 if exclude_current_positions_from_observation else 10
        observation_space = Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            str(SWIMMER_XML_PATH),
            4,
            observation_space=observation_space,
            **kwargs,
        )


class SwimmerContinuousActionEnv(gym.Wrapper):
    """Task-owned wrapper for Swimmer's two native torques."""

    def __init__(
        self,
        *,
        render_mode: str,
        base_make_kwargs: dict[str, Any],
    ) -> None:
        env = SwimmerV4Mujoco3Env(render_mode=render_mode, **base_make_kwargs)
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


def make_latency_sensitive_swimmer_env(
    *,
    render_mode: str,
    base_make_kwargs: dict[str, Any],
) -> SwimmerContinuousActionEnv:
    return SwimmerContinuousActionEnv(
        render_mode=render_mode,
        base_make_kwargs=base_make_kwargs,
    )


gym.register(
    id=GYMNASIUM_SWIMMER_ENV_ID,
    entry_point=make_latency_sensitive_swimmer_env,
    max_episode_steps=1000,
)
