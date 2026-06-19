from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.latency.sources import LatencySource


@dataclass(frozen=True)
class SimulatedLatencySample:
    latency_ms: float
    latency_raw_frames: int


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


def sample_simulated_latency(
    *,
    latency_source: LatencySource,
    clock: EnvClock,
    context: dict[str, Any],
) -> SimulatedLatencySample:
    next_latency_sample = getattr(latency_source, "next_latency_sample", None)
    if callable(next_latency_sample):
        sample = next_latency_sample(context)
        latency_ms = float(sample["latency_ms"])
        if "latency_raw_frames" in sample:
            latency_raw_frames = int(sample["latency_raw_frames"])
        else:
            latency_raw_frames = clock.latency_ms_to_raw_frames(latency_ms)
        return SimulatedLatencySample(
            latency_ms=latency_ms,
            latency_raw_frames=latency_raw_frames,
        )

    next_raw_frames = getattr(latency_source, "next_latency_raw_frames", None)
    if callable(next_raw_frames):
        latency_raw_frames = next_raw_frames(context)
        return SimulatedLatencySample(
            latency_ms=float(latency_raw_frames) * clock.frame_ms,
            latency_raw_frames=latency_raw_frames,
        )

    latency_ms = float(latency_source.next_latency_ms(context))
    latency_raw_frames = clock.latency_ms_to_raw_frames(latency_ms)
    return SimulatedLatencySample(
        latency_ms=latency_ms,
        latency_raw_frames=latency_raw_frames,
    )
