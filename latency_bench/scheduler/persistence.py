from __future__ import annotations

from abc import ABC, abstractmethod

from latency_bench.core.types import Action


class ActionPersistencePolicy(ABC):
    """Choose the action between scheduler arrivals."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def on_action_applied(self, action: Action) -> None:
        ...

    @abstractmethod
    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        ...


class HoldPolicy(ActionPersistencePolicy):
    """Retain the current action; the scheduler owns arrival and reset state."""

    def reset(self) -> None:
        pass

    def on_action_applied(self, action: Action) -> None:
        pass

    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        return current_action


class OneFrameThenNoopPolicy(ActionPersistencePolicy):
    """Apply each action for exactly one environment frame, then return to noop."""

    def __init__(self):
        self._pending = False

    def reset(self) -> None:
        self._pending = False

    def on_action_applied(self, action: Action) -> None:
        self._pending = True

    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        if not self._pending:
            return noop_action
        self._pending = False
        return current_action
