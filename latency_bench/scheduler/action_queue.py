from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, OrderingPolicy
from latency_bench.scheduler.persistence import ActionPersistencePolicy


class ActionScheduler:
    def __init__(
        self,
        clock: EnvClock,
        noop_action: Action,
        persistence_policy: ActionPersistencePolicy,
        ordering_policy: OrderingPolicy,
    ):
        self.clock = clock
        self.noop_action = noop_action
        self.persistence_policy = persistence_policy
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.ready_time_epsilon_ms = max(1e-9, self.clock.frame_ms * 1e-9)
        self._pending_events: list[ActionEvent] = []
        self._applied_events: list[ActionEvent] = []
        self._dropped_events_count = 0
        self._current_action = noop_action

    def reset(self) -> None:
        self._pending_events.clear()
        self._applied_events.clear()
        self._dropped_events_count = 0
        self._current_action = self.noop_action
        self.persistence_policy.reset()

    def enqueue(self, event: ActionEvent) -> None:
        self._pending_events.append(event)
        self._pending_events.sort(key=lambda item: item.action_id)

    def update(self, current_step: int, current_time_ms: float) -> Action:
        selected: ActionEvent | None = None
        if self._pending_events:
            if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO:
                head = self._pending_events[0]
                if self._is_ready(head, current_step, current_time_ms):
                    selected = head
                    self._pending_events = self._pending_events[1:]
            else:
                ready = [
                    event
                    for event in self._pending_events
                    if self._is_ready(event, current_step, current_time_ms)
                ]
                not_ready = [
                    event
                    for event in self._pending_events
                    if not self._is_ready(event, current_step, current_time_ms)
                ]
                if not ready:
                    selected = None
                else:
                    selected = ready[-1]
                    for event in ready[:-1]:
                        self._mark_dropped(event)
                    self._pending_events = not_ready
        if selected is not None:
            selected.applied = True
            selected.apply_raw_frame = current_step
            self._current_action = selected.action
            self._applied_events.append(selected)
            self.persistence_policy.on_action_applied(selected.action)

        return self.persistence_policy.get_action_for_step(self._current_action, self.noop_action)

    def get_pending_events(self) -> list[ActionEvent]:
        return list(self._pending_events)

    def get_applied_events(self) -> list[ActionEvent]:
        return list(self._applied_events)

    @property
    def applied_event_count(self) -> int:
        return len(self._applied_events)

    @property
    def latest_applied_event(self) -> ActionEvent | None:
        if not self._applied_events:
            return None
        return self._applied_events[-1]

    def get_applied_events_since(self, index: int) -> list[ActionEvent]:
        return self._applied_events[index:]

    @property
    def dropped_events_count(self) -> int:
        return self._dropped_events_count

    def _is_ready(self, event: ActionEvent, current_step: int, current_time_ms: float) -> bool:
        if event.ready_raw_frame is not None:
            return event.ready_raw_frame <= current_step
        return event.ready_time_ms <= current_time_ms + self.ready_time_epsilon_ms

    def _mark_dropped(self, event: ActionEvent) -> None:
        event.dropped = True
        self._dropped_events_count += 1
