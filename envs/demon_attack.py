from __future__ import annotations

import gymnasium as gym

try:
    import ale_py  # noqa: F401
except ImportError:
    pass

try:
    from cleanrl_utils.atari_wrappers import (
        ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
    )
except ImportError:
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
    )

from envs.base_wrappers import NCHWFrameStack, QueueLatencyWrapper


def make_demo_env(
    latency: int = 0,
    env_id: str = "ALE/DemonAttack-v5",
    native_frame_skip: int = 1,
    noop_max: int = 30,
    screen_size: int = 84,
    stack_frames: int = 4,
    seed: int = 0,
) -> gym.Env:
    """
    Builds a single Demon Attack env matching the CleanRL training setup.
    Returns a plain (non-vectorised) gymnasium env.

    Obs shape: (4, 84, 84)  — NCHW, 4 grayscale stacked frames.
    The CleanRL Agent.get_deterministic_action() expects a (1, 4, 84, 84) uint8 tensor.
    """
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=native_frame_skip)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = gym.wrappers.ResizeObservation(env, (screen_size, screen_size))
    env = gym.wrappers.GrayscaleObservation(env)
    env = QueueLatencyWrapper(env, frame_skip_values=[latency], seed=seed)

    if stack_frames > 1:
        env = NCHWFrameStack(env, n_stack=stack_frames)

    env.action_space.seed(seed)
    return env
