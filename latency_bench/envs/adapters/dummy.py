from __future__ import annotations

from typing import Any

import numpy as np

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter


class DummyEnvAdapter(EnvAdapter):
    """Small deterministic env adapter used by factory and executor tests."""

    def __init__(
        self,
        *,
        env_fps: float = 30.0,
        max_steps: int = 5,
        noop_action: Action | None = None,
    ):
        self.env_fps = env_fps
        self.max_steps = max_steps
        self.noop_action = noop_action or Action(value="noop", name="noop", is_noop=True)
        self.env_step = 0
        self.applied_actions: list[Action] = []
        self.gym_env = self
        self.closed = False

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self.applied_actions.clear()
        return self._make_observation(metadata={"seed": seed})

    def step(self, action: Action) -> StepResult:
        self.applied_actions.append(action)
        self.env_step += 1
        return StepResult(
            observation=self._make_observation(metadata={"action": action.name or action.value}),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"env_step": self.env_step},
        )

    def observe(self) -> Observation:
        return self._make_observation()

    def render_frame(self) -> Any:
        return {"state": self.env_step}

    def render_game_frame(self) -> Any:
        return np.full((8, 8, 3), self.env_step, dtype=np.uint8)

    def get_state_info(self) -> dict[str, Any]:
        return {
            "env_step": self.env_step,
            "sim_time_ms": self.env_step * self.frame_ms,
        }

    def close(self) -> None:
        self.closed = True

    def _make_observation(self, metadata: dict[str, Any] | None = None) -> Observation:
        obs_metadata = dict(metadata or {})
        obs_metadata.setdefault("env_step", self.env_step)
        obs_metadata.setdefault("sim_time_ms", self.env_step * self.frame_ms)
        return Observation(
            data={"state": self.env_step},
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=obs_metadata,
        )
