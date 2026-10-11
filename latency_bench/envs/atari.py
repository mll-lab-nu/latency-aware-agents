from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


TRUE_EPISODE_END_INFO_KEY = "is_true_episode_end"


def is_true_atari_episode_end(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any] | None,
) -> bool:
    """Distinguish true Atari game ends from life-loss pseudo terminals."""

    if truncated:
        return True
    if not terminated:
        return False
    if info is None:
        return True
    if TRUE_EPISODE_END_INFO_KEY in info:
        return info[TRUE_EPISODE_END_INFO_KEY]
    if "game_over" in info:
        return info["game_over"]
    if "lives" in info:
        return info["lives"] <= 0
    return True


def register_ale() -> None:
    """Register ALE when a Gymnasium Atari task is constructed."""
    import ale_py

    register_envs = getattr(gym, "register_envs", None)
    if register_envs is not None:
        register_envs(ale_py)


class AtariChannelFirstObservation(gym.ObservationWrapper):
    """Convert Atari observations from HWC to CHW uint8 without changing pixels."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width, channels = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(np.asarray(observation, dtype=np.uint8), (2, 0, 1))
