from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from latency_bench.latency.profile import load_latency_samples, load_profile_manifest
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


class EmpiricalSampleLatencySource(LatencySource):
    def __init__(self, profile_path: str | Path | None, seed: int | None = None):
        seed = _coerce_seed(seed)
        self.profile_path, self.trace_path, self.profile, self.samples_ms = _load_profile_samples(profile_path)
        self.profile_id = self.profile["profile_id"]
        self.profile_type = self.profile["profile_type"]
        self.latency_scope = self.profile["latency_scope"]
        self.target_device = self.profile["target_device"]
        self.rng = random.Random(seed)
        self.last_sample_index: int | None = None
        self.last_metadata: dict[str, Any] | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        sample_index = self.rng.randrange(len(self.samples_ms))
        self.last_sample_index = sample_index
        self.last_metadata = _profile_latency_metadata(
            self,
            sample_index=sample_index,
        )
        return float(self.samples_ms[sample_index])


class EmpiricalTraceReplayLatencySource(LatencySource):
    def __init__(
        self,
        profile_path: str | Path | None,
        trace_replay_order: str = "sequential",
        loop_trace: bool = True,
        seed: int | None = None,
    ):
        seed = _coerce_seed(seed)
        if trace_replay_order not in {"sequential", "random"}:
            raise ValueError("trace_replay_order must be one of ['random', 'sequential']")
        self.profile_path, self.trace_path, self.profile, self.samples_ms = _load_profile_samples(profile_path)
        self.profile_id = self.profile["profile_id"]
        self.profile_type = self.profile["profile_type"]
        self.latency_scope = self.profile["latency_scope"]
        self.target_device = self.profile["target_device"]
        self.trace_replay_order = trace_replay_order
        self.loop_trace = bool(loop_trace)
        self.order = list(range(len(self.samples_ms)))
        if trace_replay_order == "random":
            random.Random(seed).shuffle(self.order)
        self.index = 0
        self.last_trace_index: int | None = None
        self.last_metadata: dict[str, Any] | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        if self.index >= len(self.order):
            if not self.loop_trace:
                raise StopIteration("Empirical latency trace exhausted")
            self.index = 0
        trace_index = self.order[self.index]
        self.index += 1
        self.last_trace_index = trace_index
        self.last_metadata = _profile_latency_metadata(
            self,
            trace_index=trace_index,
        )
        return float(self.samples_ms[trace_index])


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


def _resolve_profile_manifest_path(profile_path: str | Path | None) -> Path:
    if profile_path is None:
        raise ValueError("profile_path is required")
    resolved = Path(profile_path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "profile.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Profile manifest not found: {resolved}")
    return resolved


def _load_profile_samples(profile_path: str | Path | None) -> tuple[Path, Path, dict[str, Any], list[float]]:
    manifest_path = _resolve_profile_manifest_path(profile_path)
    profile = load_profile_manifest(manifest_path)
    trace_path = Path(profile["trace_path"])
    if not trace_path.is_absolute():
        trace_path = manifest_path.parent / trace_path
    if not trace_path.exists():
        raise FileNotFoundError(f"Latency profile trace file not found: {trace_path}")
    samples_ms = load_latency_samples(trace_path)
    return manifest_path, trace_path, profile, samples_ms


def _coerce_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int or None")
    return seed


def _profile_latency_metadata(source: Any, **extra: int) -> dict[str, Any]:
    metadata = {
        "profile_id": source.profile_id,
        "profile_type": source.profile_type,
        "latency_scope": source.latency_scope,
        "target_device": source.target_device,
        "profile_path": str(source.profile_path),
    }
    metadata.update(extra)
    return metadata
