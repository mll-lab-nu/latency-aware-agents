from __future__ import annotations

from dataclasses import dataclass

from latency_bench.latency.sample import LatencySample

__all__ = ["LatencySample", "SimulatedWorkerCapacity"]


@dataclass
class SimulatedWorkerCapacity:
    capacity: int | None
    busy_until_raw_frames: list[int]

    def release_ready(self, raw_frame: int) -> None:
        if self.capacity is not None:
            self.busy_until_raw_frames = [
                ready_raw_frame
                for ready_raw_frame in self.busy_until_raw_frames
                if ready_raw_frame > raw_frame
            ]

    def can_submit(self) -> bool:
        return self.capacity is None or len(self.busy_until_raw_frames) < self.capacity

    def submit(self, ready_raw_frame: int) -> None:
        if self.capacity is not None:
            self.busy_until_raw_frames.append(int(ready_raw_frame))

    def reset(self) -> None:
        self.busy_until_raw_frames.clear()

    @property
    def in_flight_count(self) -> int:
        return len(self.busy_until_raw_frames)

    @property
    def idle_worker_count(self) -> int | None:
        if self.capacity is None:
            return None
        return self.capacity - len(self.busy_until_raw_frames)
