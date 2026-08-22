"""MyoChallenge Table Tennis P2 Gymnasium integration.

The MyoSuite package is imported from the Gymnasium entry point rather than at
module import time.  This keeps the rest of the benchmark importable when the
optional musculoskeletal simulator is not installed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID = "LatencyBench/MyoChallengeTableTennisP2-v0"
MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID = "myoChallengeTableTennisP2-v0"
MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA = "default"
MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE = (224, 224)
MYOCHALLENGE_TABLE_TENNIS_P2_MAX_EPISODE_STEPS = 300


class MyoChallengeTableTennisP2Env(gym.Wrapper):
    """Expose MyoSuite's native P2 state, action, score, and RGB interfaces."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def __init__(
        self,
        *,
        base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
        render_mode: str = "rgb_array",
        base_make_kwargs: dict[str, Any],
        camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
        render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
        render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
    ) -> None:
        # MyoSuite registers its environments during import.  It is deliberately
        # imported by the factory below, not by this module.
        self._render_mode = render_mode
        env = gym.make(base_env_id, **base_make_kwargs)
        super().__init__(env)
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        self._offscreen_renderer: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # TableTennisEnvV0 randomizes P2 properties before its base reset sees
        # the Gymnasium seed. Seed MyoSuite's task RNG at the wrapper boundary
        # so saved action traces can be replayed in a fresh environment.
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        native_action = np.asarray(action, dtype=self.env.action_space.dtype)
        observation, reward, terminated, truncated, info = self.env.step(native_action)
        info = dict(info)
        info["solved"] = bool(info["solved"])
        info["score"] = float(info["solved"])
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._offscreen_renderer is None:
            import mujoco

            model = self.env.unwrapped.sim.model.ptr
            self._offscreen_renderer = mujoco.Renderer(
                model,
                height=self.render_height,
                width=self.render_width,
            )
        self._offscreen_renderer.update_scene(
            self.env.unwrapped.sim.data.ptr,
            camera=self.camera_name,
        )
        return self._offscreen_renderer.render()

    def close(self) -> None:
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        self.env.close()
        self.env.unwrapped.sim.close()


def make_latency_sensitive_myochallenge_table_tennis_p2_env(
    *,
    base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    render_mode: str = "rgb_array",
    base_make_kwargs: dict[str, Any],
    camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
    render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
) -> MyoChallengeTableTennisP2Env:
    """Construct P2 after lazily importing MyoSuite's environment registry."""
    import myosuite  # noqa: F401

    return MyoChallengeTableTennisP2Env(
        base_env_id=base_env_id,
        render_mode=render_mode,
        base_make_kwargs=base_make_kwargs,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )


gym.register(
    id=GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
    entry_point=make_latency_sensitive_myochallenge_table_tennis_p2_env,
    max_episode_steps=MYOCHALLENGE_TABLE_TENNIS_P2_MAX_EPISODE_STEPS,
    kwargs={
        "base_env_id": MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
        "render_mode": "rgb_array",
        "base_make_kwargs": {},
        "camera_name": MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
        "render_width": MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
        "render_height": MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
    },
)
