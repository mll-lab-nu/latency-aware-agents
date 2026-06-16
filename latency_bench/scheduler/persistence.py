from __future__ import annotations

from abc import ABC, abstractmethod

from latency_bench.core.types import Action


class ActionPersistencePolicy(ABC):
    @abstractmethod
    def on_action_applied(self, action: Action) -> None:
        ...

    @abstractmethod
    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        ...


class HoldPolicy(ActionPersistencePolicy):
    """Hold each applied action for a fixed number of environment frames."""

    def __init__(self, duration_raw_frames: int):
        if duration_raw_frames <= 0:
            raise ValueError("duration_raw_frames must be positive")
        self.duration_raw_frames = duration_raw_frames
        self._frames_remaining = 0

    def reset(self) -> None:
        self._frames_remaining = 0

    def on_action_applied(self, action: Action) -> None:
        self._frames_remaining = self.duration_raw_frames

    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        if self._frames_remaining <= 0:
            return noop_action
        self._frames_remaining -= 1
        return current_action


class OneFrameThenNoopPolicy(HoldPolicy):
    """Apply each action for exactly one environment frame, then return to noop."""

    def __init__(self):
        super().__init__(duration_raw_frames=1)
