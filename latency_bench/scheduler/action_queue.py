from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, StaleActionMode
from latency_bench.scheduler.persistence import ActionPersistencePolicy


class ActionScheduler:
    def __init__(
        self,
        clock: EnvClock,
        noop_action: Action,
        persistence_policy: ActionPersistencePolicy,
        stale_action_mode: StaleActionMode = StaleActionMode.LATEST_READY,
        expiration_ms: float | None = None,
    ):
        self.clock = clock
        self.noop_action = noop_action
        self.persistence_policy = persistence_policy
        self.stale_action_mode = StaleActionMode(stale_action_mode)
        self.expiration_ms = clock.frame_ms if expiration_ms is None else float(expiration_ms)
        if self.expiration_ms < 0:
            raise ValueError("expiration_ms must be non-negative")
        self.ready_time_epsilon_ms = max(1e-9, self.clock.frame_ms * 1e-9)
        self._pending_events: list[ActionEvent] = []
        self._applied_events: list[ActionEvent] = []
        self._current_action = noop_action

    def reset(self) -> None:
        self._pending_events.clear()
        self._applied_events.clear()
        self._current_action = self.noop_action
        reset = getattr(self.persistence_policy, "reset", None)
        if reset is not None:
            reset()

    def enqueue(self, event: ActionEvent) -> None:
        self._pending_events.append(event)
        self._pending_events.sort(key=lambda item: (item.ready_time_ms, item.action_id))

    def update(self, current_step: int, current_time_ms: float) -> Action:
        ready = [event for event in self._pending_events if self._is_ready(event, current_time_ms)]
        not_ready = [event for event in self._pending_events if not self._is_ready(event, current_time_ms)]

        selected: ActionEvent | None = None
        if ready:
            if self.stale_action_mode == StaleActionMode.FIFO:
                selected = ready[0]
                self._pending_events = ready[1:] + not_ready
            elif self.stale_action_mode == StaleActionMode.LATEST_READY:
                selected = ready[-1]
                for event in ready[:-1]:
                    self._mark_dropped(event)
                self._pending_events = not_ready
            elif self.stale_action_mode == StaleActionMode.DROP_EXPIRED:
                expired: list[ActionEvent] = []
                fresh: list[ActionEvent] = []
                for event in ready:
                    if current_time_ms - event.ready_time_ms > self.expiration_ms:
                        expired.append(event)
                    else:
                        fresh.append(event)
                for event in expired:
                    self._mark_dropped(event)
                if fresh:
                    selected = fresh[-1]
                    for event in fresh[:-1]:
                        self._mark_dropped(event)
                self._pending_events = not_ready
            else:
                raise ValueError(f"Unsupported stale action mode: {self.stale_action_mode}")

        if selected is not None:
            selected.applied = True
            selected.applied_env_step = int(current_step)
            selected.applied_time_ms = float(current_time_ms)
            self._current_action = selected.action
            self._applied_events.append(selected)
            self.persistence_policy.on_action_applied(selected.action)

        return self.persistence_policy.get_action_for_step(self._current_action, self.noop_action)

    def get_pending_events(self) -> list[ActionEvent]:
        return list(self._pending_events)

    def get_applied_events(self) -> list[ActionEvent]:
        return list(self._applied_events)

    def _is_ready(self, event: ActionEvent, current_time_ms: float) -> bool:
        return event.ready_time_ms <= current_time_ms + self.ready_time_epsilon_ms

    def _mark_dropped(self, event: ActionEvent) -> None:
        event.dropped = True
