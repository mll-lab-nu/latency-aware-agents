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


class HoldPreviousActionPolicy(ActionPersistencePolicy):
    """Keep executing the latest applied action until a new action is applied."""

    def reset(self) -> None:
        return None

    def on_action_applied(self, action: Action) -> None:
        return None

    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        return current_action


class NoopAfterOneShotPolicy(ActionPersistencePolicy):
    """
    If an action is one-shot, apply it for one frame and then fall back to noop.
    Non-one-shot actions may be held.
    """

    def __init__(self):
        self._oneshot_ready_for_current_step = False

    def reset(self) -> None:
        self._oneshot_ready_for_current_step = False

    def on_action_applied(self, action: Action) -> None:
        self._oneshot_ready_for_current_step = bool(action.is_oneshot)

    def get_action_for_step(self, current_action: Action, noop_action: Action) -> Action:
        if not current_action.is_oneshot:
            return current_action
        if self._oneshot_ready_for_current_step:
            self._oneshot_ready_for_current_step = False
            return current_action
        return noop_action
