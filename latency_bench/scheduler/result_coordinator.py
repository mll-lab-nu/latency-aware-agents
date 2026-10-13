"""Ordering arbitration for out-of-order inference results.

Shared by the realtime executor (results arriving on the wall clock) and the
simulated executors (sampled-latency events arriving on the sim clock) so both
paths run the exact same semantics: once a result is released, every older
in-flight observation is permanently voided and its late result is ignored.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from latency_bench.core.types import OrderingPolicy

ResultT = TypeVar("ResultT")


class ResultCoordinator(Generic[ResultT]):
    def __init__(self, *, ordering_policy: OrderingPolicy):
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.issue_order: list[int] = []
        self.result_buffer: dict[int, ResultT] = {}
        self.ignored_obs_ids: set[int] = set()
        self.ignored_late_result_count = 0
        self.discarded_ready_result_count = 0
        self.fifo_blocked_count = 0

    def mark_submitted(self, obs_id: int) -> None:
        self.issue_order.append(int(obs_id))

    def add_completion(self, obs_id: int, result: ResultT) -> list[ResultT]:
        obs_id = int(obs_id)
        if obs_id in self.ignored_obs_ids:
            self.ignored_obs_ids.remove(obs_id)
            self.ignored_late_result_count += 1
            return [result]
        if obs_id in self.issue_order:
            self.result_buffer[obs_id] = result
        return []

    def release_ready(self) -> tuple[list[ResultT], list[ResultT]]:
        if self.ordering_policy == OrderingPolicy.LATEST_READY:
            return self._release_latest_ready()
        return self._release_fifo()

    def _release_latest_ready(self) -> tuple[list[ResultT], list[ResultT]]:
        if not self.result_buffer:
            return [], []
        selected_obs_id = max(self.result_buffer)
        selected = self.result_buffer.pop(selected_obs_id)
        discarded: list[ResultT] = []
        # Only unfinished requests can produce a late result. Completed
        # discards must not remain in the ignore set for the whole episode.
        for obs_id in self.issue_order:
            if obs_id < selected_obs_id and obs_id not in self.result_buffer:
                self.ignored_obs_ids.add(obs_id)
        for obs_id in sorted(list(self.result_buffer)):
            if obs_id < selected_obs_id:
                discarded.append(self.result_buffer.pop(obs_id))
        self.issue_order = [obs_id for obs_id in self.issue_order if obs_id > selected_obs_id]
        self.discarded_ready_result_count += len(discarded)
        return [selected], discarded

    def _release_fifo(self) -> tuple[list[ResultT], list[ResultT]]:
        released: list[ResultT] = []
        while self.issue_order and self.issue_order[0] in self.result_buffer:
            obs_id = self.issue_order.pop(0)
            released.append(self.result_buffer.pop(obs_id))
        if not released and self.issue_order and self.result_buffer:
            self.fifo_blocked_count += 1
        return released, []
