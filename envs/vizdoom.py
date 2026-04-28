from __future__ import annotations

import os
import tempfile

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from vizdoom import gymnasium_wrapper  # noqa: F401  registers VizdoomBasic-v1 etc.
except ImportError:
    gymnasium_wrapper = None  # type: ignore

from envs.base_wrappers import EnvFrameStack, QueueLatencyWrapper


class VizdoomScreenObsWrapper(ObservationWrapper):
    """Extracts the 'screen' buffer from VizDoom's dict observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert "screen" in self.observation_space.spaces
        self.observation_space = self.observation_space.spaces["screen"]

    def observation(self, obs):
        return np.asarray(obs["screen"], dtype=np.uint8)


def make_demo_env(
    latency: int = 0,
    env_id: str = "VizdoomBasic-v1",
    vizdoom_frame_skip: int = 1,
    max_buttons_pressed: int = 1,
    screen_size: tuple = (84, 84),
    stack_frames: int = 4,
    seed: int = 0,
) -> DummyVecEnv:
    """
    Builds a single VizDoom Basic env matching the SB3 training setup (image_only, grayscale).
    Returns a DummyVecEnv (n_envs=1).

    Obs shape: (1, 84, 84, 4)  — 4 stacked grayscale frames (HWC with 4 channels).
    """
    if gymnasium_wrapper is None:
        raise ImportError("vizdoom not installed. Run: pip install vizdoom")

    _tmp_root = tempfile.mkdtemp(prefix="vizdoom_demo_")

    def _init():
        worker_dir = os.path.join(_tmp_root, f"seed_{seed}")
        os.makedirs(worker_dir, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(worker_dir)

        env = gym.make(
            env_id,
            frame_skip=int(vizdoom_frame_skip),
            max_buttons_pressed=int(max_buttons_pressed),
            render_mode="rgb_array",
            treat_episode_timeout_as_truncation=True,
            use_multi_binary_action_space=False,
        )

        os.chdir(old_cwd)

        env = QueueLatencyWrapper(env, frame_skip_values=[latency], seed=seed)
        env = VizdoomScreenObsWrapper(env)
        env = ResizeObservation(env, shape=screen_size)
        env = GrayscaleObservation(env, keep_dim=True)

        if stack_frames > 1:
            env = EnvFrameStack(env, n_stack=stack_frames)

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])
