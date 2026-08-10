from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


GYMNASIUM_ATARI_ENV_ID = "LatencyBench/Atari-v0"


class AtariChannelFirstObservation(gym.ObservationWrapper):
    """Expose Atari policy observations as CHW uint8 images."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        if len(shape) == 2:
            channel_first_shape = (1, shape[0], shape[1])
        else:
            channel_first_shape = (shape[2], shape[0], shape[1])
        self.observation_space = Box(
            low=0,
            high=255,
            shape=channel_first_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        array = np.asarray(observation, dtype=np.uint8)
        if array.ndim == 2:
            return array[None, ...]
        return np.transpose(array, (2, 0, 1))


def make_latency_sensitive_atari_env(
    *,
    ale_env_id: str,
    render_mode: str = "rgb_array",
    screen_size: int = 84,
    noop_max: int = 30,
    frameskip: int = 1,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    terminal_on_life_loss: bool = False,
    max_num_frames_per_episode: int = 108000,
    grayscale_obs: bool = True,
    mode: int = 1,
    difficulty: int = 0,
):
    env = gym.make(
        ale_env_id,
        render_mode=render_mode,
        obs_type="rgb",
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_num_frames_per_episode=max_num_frames_per_episode,
        mode=mode,
        difficulty=difficulty,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=1,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale_obs,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    return AtariChannelFirstObservation(env)


gym.register(
    id=GYMNASIUM_ATARI_ENV_ID,
    entry_point=make_latency_sensitive_atari_env,
)
