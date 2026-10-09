"""Shared simulated result delivery and inference worker capacity."""

from __future__ import annotations

from dataclasses import dataclass

from latency_bench.core.types import ActionEvent, OrderingPolicy
from latency_bench.scheduler.result_coordinator import ResultCoordinator


class SimulatedResultTimeline:
    """Delivers sampled-latency action events through the shared ResultCoordinator.

    Events wait here until the sim clock reaches their ready frame and then
    "arrive" exactly like realtime results arriving on the wall clock, so both
    executor families share one implementation of the out-of-order semantics
    (a late result older than an already-released one is voided, not applied).
    """

    def __init__(self, *, ordering_policy: OrderingPolicy):
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.coordinator: ResultCoordinator[ActionEvent] = ResultCoordinator(ordering_policy=ordering_policy)
        self._pending: list[tuple[int, int, ActionEvent]] = []

    def reset(self) -> None:
        self.coordinator = ResultCoordinator(ordering_policy=self.ordering_policy)
        self._pending.clear()

    @property
    def pending_event_count(self) -> int:
        return len(self._pending) + len(self.coordinator.result_buffer)

    @property
    def pending_observation_count(self) -> int:
        """Number of submitted observations whose result has not been released."""
        return len(self.coordinator.issue_order)

    def get_pending_events(self) -> list[ActionEvent]:
        """Read in-flight issue times for the committed action context window."""
        return [event for _, _, event in self._pending] + list(self.coordinator.result_buffer.values())

    def submit(self, *, obs_id: int, ready_raw_frame: int, event: ActionEvent) -> None:
        self.mark_submitted(obs_id=obs_id)
        self.attach_event(obs_id=obs_id, ready_raw_frame=ready_raw_frame, event=event)

    def mark_submitted(self, *, obs_id: int) -> None:
        # Must happen at the submission frame even when the event body is not
        # built yet (pooled inference): latest_ready voids every OLDER marked
        # observation when a newer one releases, so an observation whose
        # forward pass is still in flight has to already be on the books to be
        # voidable — deferring the mark would let its stale action apply.
        self.coordinator.mark_submitted(obs_id)

    def attach_event(self, *, obs_id: int, ready_raw_frame: int, event: ActionEvent) -> None:
        self._pending.append((int(ready_raw_frame), int(obs_id), event))

    def release_arrived(self, raw_frame: int | None) -> tuple[list[ActionEvent], list[ActionEvent]]:
        """Return (released, dropped) among events whose ready frame has arrived.

        raw_frame None drains everything, mirroring the realtime executor's
        post-episode drain of still-in-flight results.
        """
        arrived = sorted(
            item for item in self._pending if raw_frame is None or item[0] <= int(raw_frame)
        )
        if not arrived:
            return [], []
        self._pending = [
            item for item in self._pending if raw_frame is not None and item[0] > int(raw_frame)
        ]
        dropped: list[ActionEvent] = []
        for _, obs_id, event in arrived:
            for ignored in self.coordinator.add_completion(obs_id, event):
                _mark_event_dropped(ignored, reason="ignored_late_result")
                dropped.append(ignored)
        released, discarded = self.coordinator.release_ready()
        for event in discarded:
            _mark_event_dropped(event, reason="latest_ready_discarded")
            dropped.append(event)
        return released, dropped


def _mark_event_dropped(event: ActionEvent, *, reason: str) -> None:
    event.dropped = True
    event.metadata["discard_reason"] = reason


@dataclass
class SimulatedWorkerCapacity:
    """Worker occupancy with per-worker identity.

    Workers are individuals, not an anonymous pool: real degradations are
    per-worker states, so the latency model needs to know WHICH worker serves
    each request. Assignment mirrors ProcessInferencePool.submit — the
    lowest-indexed idle worker takes the task — so the simulated request↔worker
    interleaving matches the realtime executor's.
    """

    capacity: int | None
    busy_until_by_worker: dict[int, int]

    def release_ready(self, raw_frame: int) -> None:
        self.busy_until_by_worker = {
            worker: ready_raw_frame
            for worker, ready_raw_frame in self.busy_until_by_worker.items()
            if ready_raw_frame > raw_frame
        }

    def can_submit(self) -> bool:
        return self.capacity is None or len(self.busy_until_by_worker) < self.capacity

    def assign_worker(self) -> int:
        worker = 0
        while worker in self.busy_until_by_worker:
            worker += 1
        return worker

    def submit(self, worker: int, ready_raw_frame: int) -> None:
        self.busy_until_by_worker[int(worker)] = int(ready_raw_frame)

    def reset(self) -> None:
        self.busy_until_by_worker.clear()

    @property
    def in_flight_count(self) -> int:
        return len(self.busy_until_by_worker)

    @property
    def idle_worker_count(self) -> int | None:
        if self.capacity is None:
            return None
        return self.capacity - len(self.busy_until_by_worker)
