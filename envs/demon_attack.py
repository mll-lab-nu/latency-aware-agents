from __future__ import annotations

import gymnasium as gym

try:
    import ale_py  # noqa: F401
except ImportError:
    pass

from stable_baselines3.common.atari_wrappers import AtariWrapper

from envs.base_wrappers import NCHWFrameStack, QueueLatencyWrapper


def make_demo_env(
    latency: int = 0,
    env_id: str = "ALE/DemonAttack-v5",
    native_frame_skip: int = 1,
    noop_max: int = 30,
    screen_size: int = 84,
    stack_frames: int = 4,
    seed: int = 0,
    terminal_on_life_loss: bool = True,
    clip_reward: bool = False,
) -> gym.Env:
    """
    Builds a single Demon Attack env matching the CleanRL training setup.
    Returns a plain (non-vectorised) gymnasium env.

    Obs shape: (4, 84, 84)  — NCHW, 4 grayscale stacked frames.
    The CleanRL Agent.get_deterministic_action() expects a (1, 4, 84, 84) uint8 tensor.
    """
    del native_frame_skip
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="rgb_array",
    )
    env = AtariWrapper(
        env,
        noop_max=int(noop_max),
        frame_skip=1,
        screen_size=int(screen_size),
        terminal_on_life_loss=bool(terminal_on_life_loss),
        clip_reward=bool(clip_reward),
        action_repeat_probability=0.0,
    )
    if int(latency) > 0:
        env = QueueLatencyWrapper(env, frame_skip_values=[latency], seed=seed)

    if stack_frames > 1:
        env = NCHWFrameStack(env, n_stack=stack_frames)

    env.action_space.seed(seed)
    return env
