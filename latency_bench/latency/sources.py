from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from latency_bench.latency.profiler import LatencyProfiler
from latency_bench.latency.trace import load_latency_trace


class LatencySource(ABC):
    @abstractmethod
    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        ...


class ZeroLatencySource(LatencySource):
    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        return 0.0


class FixedLatencySource(LatencySource):
    def __init__(self, fixed_latency_ms: float):
        if fixed_latency_ms < 0:
            raise ValueError("fixed_latency_ms must be non-negative")
        self.fixed_latency_ms = float(fixed_latency_ms)

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        return self.fixed_latency_ms


class EmpiricalDistributionLatencySource(LatencySource):
    def __init__(self, latency_values_ms: Sequence[float], seed: int | None = None):
        if not latency_values_ms:
            raise ValueError("latency_values_ms cannot be empty")
        self.latency_values_ms = [float(value) for value in latency_values_ms]
        if any(value < 0 for value in self.latency_values_ms):
            raise ValueError("latency_values_ms must be non-negative")
        self.rng = random.Random(seed)

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        return self.rng.choice(self.latency_values_ms)


class TraceReplayLatencySource(LatencySource):
    def __init__(self, trace_path: str | Path, loop_trace: bool = False):
        self.trace_path = Path(trace_path)
        self.loop_trace = bool(loop_trace)
        self.values_ms = load_latency_trace(self.trace_path)
        self.index = 0

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        if self.index >= len(self.values_ms):
            if not self.loop_trace:
                raise StopIteration("Latency trace exhausted")
            self.index = 0
        value = self.values_ms[self.index]
        self.index += 1
        return value


class MeasuredLatencySource(LatencySource):
    def __init__(self, profiler: LatencyProfiler):
        self.profiler = profiler
        self.last_record = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        context = context or {}
        fn = context.get("fn")
        if fn is None:
            raise ValueError("MeasuredLatencySource requires context['fn']")
        args = context.get("args", ())
        kwargs = dict(context.get("kwargs", {}))
        if "obs_id" in context:
            kwargs["obs_id"] = context["obs_id"]
        _, record = self.profiler.run_and_measure(fn, *args, **kwargs)
        self.last_record = record
        return record.latency_ms

