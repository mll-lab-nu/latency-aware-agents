"""Shared Gymnasium integration for MyoSuite environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class MyoSuiteEnv(gym.Wrapper):
    """Wrap MyoSuite construction, seed/action boundaries, horizon, and RGB rendering."""

    metadata = {"render_modes": ["rgb_array"]}

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def native_episode_horizon(self) -> int | None:
        return self.env.spec.max_episode_steps

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        native_action = np.asarray(action, dtype=self.env.action_space.dtype)
        return self.env.step(native_action)

    def __init__(
        self,
        *,
        base_env_id: str,
        render_mode: str,
        base_make_kwargs: dict[str, Any],
        camera_name: str,
        render_width: int,
        render_height: int,
    ) -> None:
        self._render_mode = render_mode
        env = gym.make(base_env_id, **base_make_kwargs)
        super().__init__(env)
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        self._offscreen_renderer: Any = None

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


MYOSUITE_ENV_ID = "LatencyBench/MyoSuite-v0"


def make_myo_suite_env(
    *,
    base_env_id: str,
    render_mode: str = "rgb_array",
    base_make_kwargs: dict[str, Any],
    camera_name: str = "default",
    render_width: int = 224,
    render_height: int = 224,
) -> MyoSuiteEnv:
    """Construct any MyoSuite task through the shared benchmark boundary."""
    # MyoSuite is optional; load it only when this registered env is built.
    import myosuite  # noqa: F401

    return MyoSuiteEnv(
        base_env_id=base_env_id,
        render_mode=render_mode,
        base_make_kwargs=base_make_kwargs,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )


gym.register(
    id=MYOSUITE_ENV_ID,
    entry_point=make_myo_suite_env,
    kwargs={
        "render_mode": "rgb_array",
        "base_make_kwargs": {},
        "camera_name": "default",
        "render_width": 224,
        "render_height": 224,
    },
)
