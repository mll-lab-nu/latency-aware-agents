from __future__ import annotations

from copy import deepcopy

from latency_bench.core.clock import EnvClock, FRAME_ROUNDING_EPSILON
from latency_bench.core.types import Action, ActionEvent, OrderingPolicy
from latency_bench.scheduler.persistence import ActionPersistencePolicy


def chunk_context_window(current_step, source_step, min_ready, pending_source_steps):
    """Heads committed before the earliest possible replacement request."""
    replacement = min(
        [current_step + min_ready]
        + [max(current_step + 1, source + min_ready)
           for source in pending_source_steps if source > source_step]
    )
    return current_step - source_step, replacement - source_step


class ActionScheduler:
    def __init__(
        self,
        clock: EnvClock,
        noop_action: Action,
        persistence_policy: ActionPersistencePolicy,
        ordering_policy: OrderingPolicy,
        hold_last_chunk_action: bool = False,
    ):
        self.clock = clock
        self.noop_action = noop_action
        self.persistence_policy = persistence_policy
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.hold_last_chunk_action = hold_last_chunk_action
        self.ready_time_epsilon_ms = self.clock.frame_ms * FRAME_ROUNDING_EPSILON
        self._pending_events: list[ActionEvent] = []
        self._applied_events: list[ActionEvent] = []
        self._dropped_events_count = 0
        self._current_action = noop_action
        self._active_chunk_event: ActionEvent | None = None
        self._chunk_index = 0
        self._underflow_event_ids: set[int] = set()
        self._underflow_count = 0
        self._underflow_frames = 0
        self._stale_prefix_frames = 0
        self._chunk_age_raw_frames = 0
        self._chunk_seen = False

    def reset(self) -> None:
        self._pending_events.clear()
        self._applied_events.clear()
        self._dropped_events_count = 0
        self._current_action = self.noop_action
        self._active_chunk_event = None
        self._chunk_index = 0
        self._underflow_event_ids.clear()
        self._underflow_count = 0
        self._underflow_frames = 0
        self._stale_prefix_frames = 0
        self._chunk_age_raw_frames = 0
        self._chunk_seen = False
        self.persistence_policy.reset()

    def project_actions(self, start_frame: int, count: int) -> list[Action]:
        """Project received actions with the live scheduling rules, without mutation.

        Completed history is not part of the controller state needed for projection.
        Unreturned results live outside this scheduler and are never consulted.
        """
        projected = deepcopy(self, {id(self._applied_events): []})
        return [projected.update(frame, self.clock.step_to_time_ms(frame))
                for frame in range(start_frame, start_frame + count)]

    def enqueue(self, event: ActionEvent) -> None:
        self._pending_events.append(event)
        self._pending_events.sort(key=lambda item: item.action_id)

    def _select_ready_events(
        self,
        current_step: int,
        current_time_ms: float,
    ) -> tuple[ActionEvent | None, list[ActionEvent] | None]:
        """Return the selected event and latest-ready drop candidates.

        FIFO returns ``None`` for candidates because it never drops other ready
        events; latest-ready returns every ready event in queue order.
        """
        if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO:
            head = self._pending_events[0]
            if self._is_ready(head, current_step, current_time_ms):
                return head, None
            return None, None
        ready = [
            event
            for event in self._pending_events
            if self._is_ready(event, current_step, current_time_ms)
        ]
        return (ready[-1] if ready else None), ready

    def update(self, current_step: int, current_time_ms: float) -> Action:
        selected: ActionEvent | None = None
        if self._pending_events:
            selected, ready = self._select_ready_events(current_step, current_time_ms)
            if ready is None:
                if selected is not None:
                    self._pending_events = self._pending_events[1:]
            elif selected is not None:
                for event in ready[:-1]:
                    self._mark_dropped(event)
                self._pending_events = [
                    event
                    for event in self._pending_events
                    if not self._is_ready(event, current_step, current_time_ms)
                ]
        if selected is not None:
            selected.applied = True
            selected.apply_raw_frame = current_step
            self._applied_events.append(selected)
            if selected.action_chunk is None:
                self._active_chunk_event = None
                self._current_action = selected.action
                self.persistence_policy.on_action_applied(selected.action)
            else:
                self._active_chunk_event = selected
                self._chunk_seen = True
                self._chunk_index = current_step - selected.obs_env_step
                self._stale_prefix_frames = self._chunk_index
                self._underflow_event_ids.discard(selected.action_id)

        if self._active_chunk_event is not None:
            return self._next_chunk_action(current_step)
        return self.persistence_policy.get_action_for_step(self._current_action, self.noop_action)

    def scheduled_chunk_event(self, current_step, current_time_ms):
        """Peek at the returned chunk used by the next scheduler update."""
        selected = None
        if self._pending_events:
            selected, _ = self._select_ready_events(current_step, current_time_ms)
        event = selected if selected is not None else self._active_chunk_event
        if event is None or event.action_chunk is None:
            return None, -1
        index = current_step - event.obs_env_step
        if index < len(event.action_chunk):
            return event, index
        if self.hold_last_chunk_action:
            return event, len(event.action_chunk) - 1
        return None, -1

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

    @property
    def underflow_count(self) -> int:
        return self._underflow_count

    @property
    def underflow_frames(self) -> int:
        return self._underflow_frames

    @property
    def underflow_duration_ms(self) -> float:
        return self._underflow_frames * self.clock.frame_ms

    @property
    def stale_prefix_frames(self) -> int:
        return self._stale_prefix_frames

    @property
    def chunk_age_raw_frames(self) -> int:
        return self._chunk_age_raw_frames

    def chunk_metrics(self) -> dict[str, bool | int | float]:
        if not self._chunk_seen:
            return {}
        return {
            "hold_last_chunk_action": self.hold_last_chunk_action,
            "underflow_count": self.underflow_count,
            "underflow_frames": self.underflow_frames,
            "underflow_duration_ms": self.underflow_duration_ms,
            "stale_prefix_frames": self.stale_prefix_frames,
            "chunk_age_raw_frames": self.chunk_age_raw_frames,
            "chunk_age_ms": self.chunk_age_raw_frames * self.clock.frame_ms,
        }

    def _is_ready(self, event: ActionEvent, current_step: int, current_time_ms: float) -> bool:
        if event.ready_raw_frame is not None:
            return event.ready_raw_frame <= current_step
        return event.ready_time_ms <= current_time_ms + self.ready_time_epsilon_ms

    def _mark_dropped(self, event: ActionEvent) -> None:
        event.dropped = True
        self._dropped_events_count += 1

    def _next_chunk_action(self, current_step: int) -> Action:
        event = self._active_chunk_event
        chunk = event.action_chunk
        self._chunk_age_raw_frames = current_step - event.obs_env_step
        event.metadata["stale_prefix_frames"] = self._stale_prefix_frames
        event.metadata["chunk_age_raw_frames"] = self._chunk_age_raw_frames
        event.metadata["chunk_age_ms"] = self._chunk_age_raw_frames * self.clock.frame_ms
        event.metadata["hold_last_chunk_action"] = self.hold_last_chunk_action
        if self._chunk_index < len(chunk):
            action = chunk[self._chunk_index]
            if not isinstance(action, Action):
                action = Action(value=action)
            event.metadata["chunk_index"] = self._chunk_index
            self._chunk_index += 1
            self._current_action = action
            return action

        if event.action_id not in self._underflow_event_ids:
            self._underflow_event_ids.add(event.action_id)
            self._underflow_count += 1
        self._underflow_frames += 1
        event.metadata["underflow_count"] = self._underflow_count
        event.metadata["underflow_frames"] = self._underflow_frames
        event.metadata["underflow_duration_ms"] = self.underflow_duration_ms
        if self.hold_last_chunk_action:
            action = chunk[-1]
            if not isinstance(action, Action):
                action = Action(value=action)
            self._current_action = action
            event.metadata["chunk_index"] = len(chunk) - 1
            return action
        return self.noop_action
