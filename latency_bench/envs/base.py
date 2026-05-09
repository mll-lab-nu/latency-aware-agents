from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from latency_bench.core.types import Action, Observation, StepResult


class EnvAdapter(ABC):
    env_fps: float
    noop_action: Action

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    def observe(self) -> Observation:
        info = self.get_state_info()
        frame = self.render_frame()
        data = frame if frame is not None else info
        return Observation(
            data=data,
            env_step=int(info.get("env_step", 0)),
            sim_time_ms=float(info.get("sim_time_ms", 0.0)),
            metadata=info,
        )

    def render_frame(self) -> Any:
        return None

    def get_state_info(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None
