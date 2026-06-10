from __future__ import annotations

import random
from bisect import bisect_left
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from latency_bench.latency.profile import load_latency_distribution, load_profile_manifest
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
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
        self.fixed_latency_ms = fixed_latency_ms

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
        return value


class UniformDistributionLatencySource(LatencySource):
    def __init__(
        self,
        min_latency_ms: float,
        max_latency_ms: float,
        seed: int | None = None,
    ):
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.rng = random.Random(seed)

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        if self.min_latency_ms == self.max_latency_ms:
            return self.min_latency_ms
        return self.rng.uniform(self.min_latency_ms, self.max_latency_ms)


class MixedNormalDistributionLatencySource(NormalDistributionLatencySource):
    def __init__(
        self,
        mean_latency_ms: float,
        std_latency_ms: float,
        seed: int | None = None,
        min_latency_ms: float = 0.0,
        max_latency_ms: float | None = None,
    ):
        super().__init__(
            mean_latency_ms=mean_latency_ms,
            std_latency_ms=std_latency_ms,
            seed=seed,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
        )
        self._episode_id: int | None = None
        self._episode_latency_ms: float | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        episode_id = int(context["episode_id"])
        if self._episode_id != episode_id:
            self._episode_id = episode_id
            self._episode_latency_ms = super().next_latency_ms(context)
        return self._episode_latency_ms


class MixedUniformDistributionLatencySource(UniformDistributionLatencySource):
    def __init__(
        self,
        min_latency_ms: float,
        max_latency_ms: float,
        seed: int | None = None,
    ):
        super().__init__(
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            seed=seed,
        )
        self._episode_id: int | None = None
        self._episode_latency_ms: float | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        episode_id = int(context["episode_id"])
        if self._episode_id != episode_id:
            self._episode_id = episode_id
            self._episode_latency_ms = super().next_latency_ms(context)
        return self._episode_latency_ms


class TraceReplayLatencySource(LatencySource):
    def __init__(self, trace_path: str | Path, loop_trace: bool = False):
        self.trace_path = Path(trace_path)
        self.loop_trace = loop_trace
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


class ProfileSampleLatencySource(LatencySource):
    def __init__(self, profile_path: str | Path | None, seed: int | None = None):
        self.profile_path = _resolve_profile_manifest_path(profile_path)
        self.profile = load_profile_manifest(self.profile_path)
        self.distribution_path = Path(self.profile["distribution_path"])
        if not self.distribution_path.is_absolute():
            self.distribution_path = self.profile_path.parent / self.distribution_path
        distribution = load_latency_distribution(self.distribution_path)
        self.quantile_levels = [float(value) for value in distribution["quantile_levels"]]
        self.latency_ms = [float(value) for value in distribution["latency_ms"]]
        self.profile_id = self.profile["profile_id"]
        self.profile_type = self.profile["profile_type"]
        self.target_device = self.profile["target_device"]
        self.distribution_type = self.profile["distribution_type"]
        self.rng = random.Random(seed)
        self.last_metadata: dict[str, Any] | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        quantile = self.rng.random()
        latency = self._sample_inverse_cdf(quantile)
        self.last_metadata = _profile_latency_metadata(self)
        return latency

    def _sample_inverse_cdf(self, quantile: float) -> float:
        index = bisect_left(self.quantile_levels, quantile)
        if index == 0:
            return self.latency_ms[0]
        if index >= len(self.quantile_levels):
            return self.latency_ms[-1]
        lower_q = self.quantile_levels[index - 1]
        upper_q = self.quantile_levels[index]
        lower_latency = self.latency_ms[index - 1]
        upper_latency = self.latency_ms[index]
        weight = (quantile - lower_q) / (upper_q - lower_q)
        return lower_latency + weight * (upper_latency - lower_latency)


class MeasuredLatencySource(LatencySource):
    """Compatibility adapter for older code that measured via a latency source.

    Realtime measured execution now measures policy work in the executor's
    policy worker so it can preserve queue wait and effective action delay.
    """

    def __init__(self, profiler: RuntimeLatencyMeasurer):
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


def _profile_latency_metadata(source: Any, **extra: int) -> dict[str, Any]:
    metadata = {
        "profile_id": source.profile_id,
        "profile_type": source.profile_type,
        "target_device": source.target_device,
        "profile_path": str(source.profile_path),
    }
    metadata.update(extra)
    return metadata
