from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_bench.latency.burst_profile import SemiMarkovLatencyBurstSampler
from latency_bench.utils.io import read_json


@dataclass(frozen=True)
class ProfileLatencySample:
    latency_ms: float
    latency_raw_frames: int | None = None
    action_ready_raw_frames: int | None = None
    worker_service_ms: float | None = None
    worker_service_raw_frames: int | None = None

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {"latency_ms": self.latency_ms}
        if self.latency_raw_frames is not None:
            payload["latency_raw_frames"] = self.latency_raw_frames
        if self.action_ready_raw_frames is not None:
            payload["action_ready_raw_frames"] = self.action_ready_raw_frames
        if self.worker_service_ms is not None:
            payload["worker_service_ms"] = self.worker_service_ms
        if self.worker_service_raw_frames is not None:
            payload["worker_service_raw_frames"] = self.worker_service_raw_frames
        return payload


class IidInverseCdfProfileSampler:
    def __init__(self, distribution: dict[str, Any], seed: int | None = None):
        self.distribution = distribution
        self.rng = random.Random(seed)

    def sample(self, context: dict[str, Any]) -> ProfileLatencySample:
        del context
        quantile_levels = self.distribution["quantile_levels"]
        latency_ms = self.distribution["latency_ms"]
        quantile = self.rng.random()
        index = bisect_left(quantile_levels, quantile)
        if index == 0:
            return ProfileLatencySample(latency_ms=float(latency_ms[0]))
        if index >= len(latency_ms):
            return ProfileLatencySample(latency_ms=float(latency_ms[-1]))
        lower_q = float(quantile_levels[index - 1])
        upper_q = float(quantile_levels[index])
        lower_latency = float(latency_ms[index - 1])
        upper_latency = float(latency_ms[index])
        weight = (quantile - lower_q) / (upper_q - lower_q)
        return ProfileLatencySample(latency_ms=lower_latency + weight * (upper_latency - lower_latency))


class SemiMarkovProfileSampler:
    def __init__(self, model: dict[str, Any], seed: int | None = None):
        self.sampler = SemiMarkovLatencyBurstSampler(model, seed=seed)

    def sample(self, context: dict[str, Any]) -> ProfileLatencySample:
        sample = self.sampler.sample(context)
        return ProfileLatencySample(
            latency_ms=float(sample["latency_ms"]),
            latency_raw_frames=int(sample["latency_raw_frames"]),
            action_ready_raw_frames=int(sample["action_ready_raw_frames"]) if "action_ready_raw_frames" in sample else None,
            worker_service_ms=float(sample["worker_service_ms"]) if "worker_service_ms" in sample else None,
            worker_service_raw_frames=int(sample["worker_service_raw_frames"]) if "worker_service_raw_frames" in sample else None,
        )


def build_profile_latency_sampler(
    *,
    profile: dict[str, Any],
    profile_path: Path,
    sampler: str,
    seed: int | None,
):
    sampler_type = _resolve_profile_sampler_type(profile, sampler)
    if sampler_type == "semi_markov_latency_frames":
        model_path = _profile_relative_path(profile_path, profile["burst_model_path"])
        return SemiMarkovProfileSampler(read_json(model_path), seed=seed), sampler_type
    if sampler_type == "iid_inverse_cdf":
        distribution_path = _profile_relative_path(profile_path, profile["distribution_path"])
        return IidInverseCdfProfileSampler(read_json(distribution_path), seed=seed), sampler_type
    raise ValueError(f"Unsupported profile sampler: {sampler_type}")


def _resolve_profile_sampler_type(profile: dict[str, Any], sampler: str) -> str:
    if sampler == "profile_default":
        if "sample_model_type" in profile:
            return str(profile["sample_model_type"])
        return "iid_inverse_cdf"
    return str(sampler)


def _profile_relative_path(profile_path: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return profile_path.parent / path
