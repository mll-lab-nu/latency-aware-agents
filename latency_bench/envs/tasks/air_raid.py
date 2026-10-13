from __future__ import annotations

from typing import Any

import gymnasium as gym

from latency_bench.envs.atari import AtariChannelFirstObservation, register_ale


GYMNASIUM_AIR_RAID_ENV_ID = "LatencyBench/AirRaid-v0"


def make_latency_sensitive_air_raid_env(
    *,
    base_env_id: str,
    render_mode: str,
    screen_size: int,
    noop_max: int,
    base_make_kwargs: dict[str, Any],
    fire_reset: bool = False,
) -> gym.Env:
    register_ale()
    from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, WarpFrame

    env = gym.make(base_env_id, render_mode=render_mode, **base_make_kwargs)
    if noop_max != 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if fire_reset:
        env = FireResetEnv(env)
    env = WarpFrame(env, width=screen_size, height=screen_size)
    return AtariChannelFirstObservation(env)


gym.register(
    id=GYMNASIUM_AIR_RAID_ENV_ID,
    entry_point=make_latency_sensitive_air_raid_env,
)
