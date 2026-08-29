from __future__ import annotations

from importlib import import_module
from typing import Any

import gymnasium
import numpy as np

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.raw_rgb import RawRgbFrameStackBuffer


class GymnasiumEnvAdapter(EnvAdapter):
    """Pass-through adapter for a standard Gymnasium environment."""

    OBSERVATION_TYPE = "gymnasium_observation_v1"

    def __init__(
        self,
        *,
        env_id: str,
        registration_imports: list[str],
        make_kwargs: dict[str, Any],
        env_fps: float,
        noop_action: Action,
        frame_stack: int,
        export_env_raw_rgb_frames: bool,
        include_state: bool = False,
    ) -> None:
        self.env_id = env_id
        self.registration_imports = registration_imports
        self.make_kwargs = make_kwargs
        self.env_fps = env_fps
        self.noop_action = noop_action
        self.frame_stack = frame_stack
        self.export_env_raw_rgb_frames = export_env_raw_rgb_frames
        self.include_state = include_state

        for module_name in registration_imports:
            import_module(module_name)
        self.gym_env = gymnasium.make(env_id, **make_kwargs)
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self._policy_frame_stack_enabled = frame_stack > 1 and isinstance(
            self.observation_space,
            gymnasium.spaces.Box,
        )
        if self._policy_frame_stack_enabled:
            self.observation_space = gymnasium.spaces.Box(
                low=np.concatenate([self.observation_space.low] * frame_stack, axis=0),
                high=np.concatenate([self.observation_space.high] * frame_stack, axis=0),
                dtype=self.observation_space.dtype,
            )

        self.env_step = 0
        self._last_frame: Any = None
        self._last_info: dict[str, Any] = {}
        self._last_state: Any = None
        self._observed_frames: list[np.ndarray] = []
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=frame_stack)

    @staticmethod
    def _state(observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            observation = observation["obs"]
        return np.asarray(observation).copy()

    def reset(self, seed: int | None = None) -> Observation:
        observation, info = self.gym_env.reset(seed=seed)
        if self.include_state:
            self._last_state = self._state(observation)
        self.env_step = 0
        if not self._policy_frame_stack_enabled:
            self._last_frame = observation
        else:
            self._observed_frames = [np.asarray(observation).copy() for _ in range(self.frame_stack)]
            self._last_frame = np.concatenate(self._observed_frames, axis=0)
        self._last_info = info
        if self.export_env_raw_rgb_frames:
            self._reset_raw_rgb_frame_stack()
        return self._make_observation(info=info)

    def step(self, action: Action) -> StepResult:
        observation, reward, terminated, truncated, info = self.gym_env.step(action.value)
        if self.include_state:
            self._last_state = self._state(observation)
        self.env_step += 1
        if not self._policy_frame_stack_enabled:
            self._last_frame = observation
        else:
            self._observed_frames = [*self._observed_frames[1:], np.asarray(observation).copy()]
            self._last_frame = np.concatenate(self._observed_frames, axis=0)
        self._last_info = info
        if self.export_env_raw_rgb_frames:
            self._append_raw_rgb_frame()

        step_info = dict(info)
        step_info.update(
            {
                "applied_action": action.value,
                "applied_action_name": action.name,
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
            }
        )
        return StepResult(
            observation=self._make_observation(info=info),
            reward=reward,
            done=terminated,
            truncated=truncated,
            info=step_info,
        )

    def observe(self) -> Observation:
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=self._metadata(self._last_info),
        )

    def _metadata(self, info: dict[str, Any]) -> dict[str, Any]:
        metadata = super()._metadata(info)
        if self.include_state:
            metadata["gymnasium_state"] = self._last_state
        return metadata

    def render_game_frame(self) -> Any:
        return self.gym_env.render()

    def close(self) -> None:
        self.gym_env.close()
