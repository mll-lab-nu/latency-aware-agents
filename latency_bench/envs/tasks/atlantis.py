from __future__ import annotations

from typing import Any

import gymnasium as gym

from latency_bench.envs.atari import AtariChannelFirstObservation, register_ale


GYMNASIUM_ATLANTIS_ENV_ID = "LatencyBench/Atlantis-v0"


def make_latency_sensitive_atlantis_env(
    *,
    base_env_id: str,
    render_mode: str,
    screen_size: int,
    noop_max: int,
    base_make_kwargs: dict[str, Any],
) -> gym.Env:
    register_ale()
    from stable_baselines3.common.atari_wrappers import NoopResetEnv, WarpFrame

    env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
    if noop_max != 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = WarpFrame(env, width=screen_size, height=screen_size)
    return AtariChannelFirstObservation(env)


gym.register(id=GYMNASIUM_ATLANTIS_ENV_ID, entry_point=make_latency_sensitive_atlantis_env)
