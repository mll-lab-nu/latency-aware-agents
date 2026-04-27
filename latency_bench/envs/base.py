from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from latency_bench.core.types import Action, Observation, StepResult


class EnvAdapter(ABC):
    fps_env: float
    noop_action: Action

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    def render_frame(self) -> Any:
        return None

    def get_state_info(self) -> dict[str, Any]:
        return {}


class DummyEnvAdapter(EnvAdapter):
    def __init__(
        self,
        fps_env: float = 30.0,
        max_steps: int = 100,
        noop_action: Action | None = None,
    ):
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        self.fps_env = float(fps_env)
        self.max_steps = int(max_steps)
        self.noop_action = noop_action or Action(value="noop", name="noop", is_noop=True)
        self.state = 0
        self.env_step = 0
        self.applied_actions: list[Action] = []

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.state = 0
        self.env_step = 0
        self.applied_actions.clear()
        return self._make_observation(metadata={"seed": seed})

    def step(self, action: Action) -> StepResult:
        self.applied_actions.append(action)
        self.state += 1
        self.env_step += 1
        done = self.env_step >= self.max_steps
        obs = self._make_observation()
        info = {
            "state": self.state,
            "env_step": self.env_step,
            "applied_action": action.value,
            "applied_action_name": action.name,
        }
        return StepResult(observation=obs, reward=1.0, done=done, truncated=False, info=info)

    def render_frame(self) -> Any:
        return {"state": self.state}

    def get_state_info(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "env_step": self.env_step,
            "sim_time_ms": self.env_step * self.frame_ms,
        }

    def _make_observation(self, metadata: dict[str, Any] | None = None) -> Observation:
        return Observation(
            data={"state": self.state},
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata or {},
        )

