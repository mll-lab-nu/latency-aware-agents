from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from latency_bench.latency.profile import load_profile_manifest
from latency_bench.latency.profile_samplers import build_profile_latency_sampler


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


class ProfileSampleLatencySource(LatencySource):
    def __init__(
        self,
        profile_path: str | Path | None,
        seed: int | None = None,
        sampler: str = "profile_default",
    ):
        self.profile_path = _resolve_profile_manifest_path(profile_path)
        self.profile = load_profile_manifest(self.profile_path)
        self.profile_id = self.profile["profile_id"]
        self.target_device = self.profile["target_device"]
        self.sampler, self.sampler_type = build_profile_latency_sampler(
            profile=self.profile,
            profile_path=self.profile_path,
            sampler=sampler,
            seed=seed,
        )
        self.last_metadata: dict[str, Any] | None = None

    def next_latency_ms(self, context: dict[str, Any] | None = None) -> float:
        return float(self.next_latency_sample(context)["latency_ms"])

    def next_latency_raw_frames(self, context: dict[str, Any] | None = None) -> int:
        return int(self.next_latency_sample(context)["latency_raw_frames"])

    def next_latency_sample(self, context: dict[str, Any] | None = None) -> dict[str, float | int]:
        sample = self.sampler.sample(context).to_dict()
        self.last_metadata = _profile_latency_metadata(self)
        return sample


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
        "target_device": source.target_device,
        "profile_path": str(source.profile_path),
    }
    metadata.update(extra)
    return metadata
