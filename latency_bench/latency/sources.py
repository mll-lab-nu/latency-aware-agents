from __future__ import annotations

import random
from abc import ABC, abstractmethod
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


class NormalDistributionLatencySource(LatencySource):
    def __init__(
        self,
        mean_latency_ms: float,
        std_latency_ms: float,
        seed: int | None = None,
        min_latency_ms: float = 0.0,
        max_latency_ms: float | None = None,
    ):
        mean_latency_ms = float(mean_latency_ms)
        std_latency_ms = float(std_latency_ms)
        min_latency_ms = float(min_latency_ms)
        max_latency_ms = None if max_latency_ms is None else float(max_latency_ms)
        if std_latency_ms < 0:
            raise ValueError("std_latency_ms must be non-negative")
        if min_latency_ms < 0:
            raise ValueError("min_latency_ms must be non-negative")
        if max_latency_ms is not None and max_latency_ms < min_latency_ms:
            raise ValueError("max_latency_ms must be greater than or equal to min_latency_ms")
        self.mean_latency_ms = mean_latency_ms
        self.std_latency_ms = std_latency_ms
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.rng = random.Random(seed)

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        if self.std_latency_ms == 0:
            value = self.mean_latency_ms
        else:
            value = self.rng.normalvariate(self.mean_latency_ms, self.std_latency_ms)
        value = max(value, self.min_latency_ms)
        if self.max_latency_ms is not None:
            value = min(value, self.max_latency_ms)
        return float(value)


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
