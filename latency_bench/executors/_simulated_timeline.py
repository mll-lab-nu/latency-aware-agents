from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.latency.sources import LatencySource


@dataclass(frozen=True)
class SimulatedLatencySample:
    latency_ms: float
    latency_raw_frames: int


def sample_simulated_latency(
    *,
    latency_source: LatencySource,
    clock: EnvClock,
    context: dict[str, Any],
) -> SimulatedLatencySample:
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
