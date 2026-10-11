from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


GYMNASIUM_ASTERIX_ENV_ID = "LatencyBench/Asterix-v0"


def _register_ale() -> None:
    import ale_py

    register_envs = getattr(gym, "register_envs", None)
    if register_envs is not None:
        register_envs(ale_py)


class AtariChannelFirstObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width, channels = self.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(np.asarray(observation, dtype=np.uint8), (2, 0, 1))


def make_latency_sensitive_asterix_env(
    *,
    base_env_id: str,
    render_mode: str,
    screen_size: int,
    noop_max: int,
    base_make_kwargs: dict[str, Any],
) -> gym.Env:
    _register_ale()
    from stable_baselines3.common.atari_wrappers import NoopResetEnv, WarpFrame

    env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
    if noop_max != 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = WarpFrame(env, width=screen_size, height=screen_size)
    return AtariChannelFirstObservation(env)


gym.register(id=GYMNASIUM_ASTERIX_ENV_ID, entry_point=make_latency_sensitive_asterix_env)
