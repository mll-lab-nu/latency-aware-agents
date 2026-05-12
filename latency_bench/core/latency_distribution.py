from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import numpy as np

from latency_bench.core.clock import frames_float_to_raw_frames


SUPPORTED_LATENCY_DISTRIBUTIONS = {"normal", "uniform"}
_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class LatencyDistributionSpec:
    type: str
    mean: float | None
    std: float | None
    min: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "min": self.min,
            "max": self.max,
        }
        if self.type == "normal":
            result["mean"] = self.mean
            result["std"] = self.std
        return result


class LatencyDistributionSampler:
    def __init__(self, spec: LatencyDistributionSpec | Mapping[str, Any] | str, *, seed: int | None = None):
        self.spec = parse_latency_distribution(spec)
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

    def sample_float(self) -> float:
        if self.spec.type == "normal":
            return float(self.rng.normal(loc=float(self.spec.mean), scale=float(self.spec.std)))
        if self.spec.type == "uniform":
            return float(self.rng.uniform(float(self.spec.min), float(self.spec.max)))
        raise ValueError(f"Unsupported latency distribution type: {self.spec.type}")

    def sample_frame(self) -> int:
        sampled_value = self.sample_float()
        clamped = min(max(sampled_value, self.spec.min), self.spec.max)
        return frames_float_to_raw_frames(clamped)


def parse_latency_distribution(raw: LatencyDistributionSpec | Mapping[str, Any] | str) -> LatencyDistributionSpec:
    if isinstance(raw, LatencyDistributionSpec):
        distribution_type = raw["type"].strip().lower()
        if distribution_type == "normal":
            return LatencyDistributionSpec(
                type=distribution_type,
                mean=raw.mean,
                std=raw.std,
                min=raw.min,
                max=raw.max,
            )
        if distribution_type == "uniform":
            return LatencyDistributionSpec(
                type=distribution_type,
                mean=None,
                std=None,
                min=raw.min,
                max=raw.max,
            )
        raise ValueError(
            "latency distribution type must be one of "
            f"{sorted(SUPPORTED_LATENCY_DISTRIBUTIONS)}"
        )
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("latency distribution spec must not be empty")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("latency distribution string must be JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("latency distribution spec must be a mapping")

    distribution_type = raw.type.strip().lower()
    if distribution_type not in SUPPORTED_LATENCY_DISTRIBUTIONS:
        raise ValueError(
            "latency distribution type must be one of "
            f"{sorted(SUPPORTED_LATENCY_DISTRIBUTIONS)}"
        )

    if distribution_type == "normal":
        spec = LatencyDistributionSpec(
            type=distribution_type,
            mean=raw["mean"],
            std=raw["std"],
            min=raw["min"],
            max=raw["max"],
        )
    else:
        spec = LatencyDistributionSpec(
            type=distribution_type,
            mean=None,
            std=None,
            min=raw["min"],
            max=raw["max"],
        )
    return spec


def make_latency_distribution_sampler(
    raw: LatencyDistributionSpec | Mapping[str, Any] | str,
    *,
    seed: int | None = None,
) -> LatencyDistributionSampler:
    return LatencyDistributionSampler(raw, seed=seed)


def latency_distribution_label(raw: LatencyDistributionSpec | Mapping[str, Any] | str) -> str:
    spec = parse_latency_distribution(raw)
    if spec.type == "normal":
        assert spec.mean is not None
        assert spec.std is not None
        return (
            f"normal_mean_{_label_number(spec.mean)}"
            f"_std_{_label_number(spec.std)}"
            f"_min_{_label_number(spec.min)}"
            f"_max_{_label_number(spec.max)}"
        )
    if spec.type == "uniform":
        return f"uniform_min_{_label_number(spec.min)}_max_{_label_number(spec.max)}"
    raise ValueError(f"Unsupported latency distribution type: {spec.type}")


def distribution_max_frame_latency(raw: LatencyDistributionSpec | Mapping[str, Any] | str) -> int:
    return frames_float_to_raw_frames(parse_latency_distribution(raw).max)


def distribution_frames_to_ms_latency_config(
    raw: LatencyDistributionSpec | Mapping[str, Any] | str,
    *,
    env_fps: float,
    seed: int | None = None,
) -> dict[str, Any]:
    spec = parse_latency_distribution(raw)
    frame_ms = 1000.0 / env_fps
    if spec.type == "normal":
        config: dict[str, Any] = {
            "type": "normal_distribution",
            "mean_latency_ms": spec.mean * frame_ms,
            "std_latency_ms": spec.std * frame_ms,
            "min_latency_ms": spec.min * frame_ms,
            "max_latency_ms": spec.max * frame_ms,
        }
    elif spec.type == "uniform":
        config = {
            "type": "uniform_distribution",
            "min_latency_ms": spec.min * frame_ms,
            "max_latency_ms": spec.max * frame_ms,
        }
    else:
        raise ValueError(f"Unsupported latency distribution type: {spec.type}")
    if seed is not None:
        config["seed"] = seed
    return config


def derive_latency_seed(
    base_seed: int | None,
    *,
    worker_index: int = 0,
    vector_index: int = 0,
    env_id: int = 0,
    episode_idx: int = 0,
    stream: int = 0,
) -> int:
    values = (
        0 if base_seed is None else int(base_seed),
        int(worker_index),
        int(vector_index),
        int(env_id),
        int(episode_idx),
        int(stream),
    )
    x = _SPLITMIX64_INCREMENT
    for value in values:
        x = _splitmix64((x + int(value) + _SPLITMIX64_INCREMENT) & _UINT64_MASK)
    return int(x & _UINT32_MASK)


def _label_number(value: int | float) -> str:
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid latency distribution label number: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"Invalid latency distribution label number: {value!r}")
    if decimal_value == decimal_value.to_integral_value():
        text = str(int(decimal_value))
    else:
        text = format(decimal_value.normalize(), "f").rstrip("0").rstrip(".")
    if text.startswith("-"):
        text = "neg" + text[1:]
    return text.replace(".", "p")


def _splitmix64(value: int) -> int:
    z = int(value) & _UINT64_MASK
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (z ^ (z >> 31)) & _UINT64_MASK
