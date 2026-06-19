from __future__ import annotations

from dataclasses import dataclass

from latency_bench.core.clock import EnvClock


@dataclass(frozen=True)
class LatencySample:
    """One fully-resolved latency draw consumed by the simulated timeline.

    Every field is populated; samplers that only produce a wall-clock latency use
    :meth:`from_latency_ms`, the run-length temporal sampler uses an explicit
    ``latency_raw_frames``, and the queue-conditioned sampler uses
    :meth:`from_action_ready` to carry the separate worker-service component.
    """
    latency_ms: float
    latency_raw_frames: int
    action_ready_raw_frames: int
    worker_service_raw_frames: int
    worker_service_ms: float

    @classmethod
    def from_latency_ms(
        cls,
        latency_ms: float,
        *,
        clock: EnvClock,
        latency_raw_frames: int | None = None,
    ) -> "LatencySample":
        latency_ms = float(latency_ms)
        if latency_raw_frames is None:
            latency_raw_frames = clock.latency_ms_to_raw_frames(latency_ms)
        return cls(
            latency_ms=latency_ms,
            latency_raw_frames=latency_raw_frames,
            action_ready_raw_frames=latency_raw_frames,
            worker_service_raw_frames=latency_raw_frames,
            worker_service_ms=float(latency_raw_frames) * clock.frame_ms,
        )

    @classmethod
    def from_action_ready(
        cls,
        *,
        action_ready_raw_frames: int,
        worker_service_raw_frames: int,
        worker_service_ms: float,
        clock: EnvClock,
        latency_ms: float | None = None,
    ) -> "LatencySample":
        action_ready_raw_frames = int(action_ready_raw_frames)
        if latency_ms is None:
            latency_ms = float(action_ready_raw_frames) * clock.frame_ms
        return cls(
            latency_ms=float(latency_ms),
            latency_raw_frames=action_ready_raw_frames,
            action_ready_raw_frames=action_ready_raw_frames,
            worker_service_raw_frames=int(worker_service_raw_frames),
            worker_service_ms=float(worker_service_ms),
        )
