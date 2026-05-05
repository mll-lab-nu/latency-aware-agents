from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import numpy as np


SUPPORTED_LATENCY_DISTRIBUTIONS = {"normal", "uniform"}
_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class LatencyDistributionSpec:
    type: str
    mean: float | None
    std: float | None
    min: int
    max: int

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "min": int(self.min),
            "max": int(self.max),
        }
        if self.type == "normal":
            result["mean"] = float(self.mean)
            result["std"] = float(self.std)
        return result


class LatencyDistributionSampler:
    def __init__(self, spec: LatencyDistributionSpec | Mapping[str, Any] | str, *, seed: int | None = None):
        self.spec = parse_latency_distribution(spec)
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

    def sample_float(self) -> float:
        if self.spec.type == "normal":
            assert self.spec.mean is not None
            assert self.spec.std is not None
            if float(self.spec.std) == 0.0:
                return float(self.spec.mean)
            return float(self.rng.normal(loc=float(self.spec.mean), scale=float(self.spec.std)))
        if self.spec.type == "uniform":
            return float(self.rng.uniform(float(self.spec.min), float(self.spec.max)))
        raise ValueError(f"Unsupported latency distribution type: {self.spec.type}")

    def sample_frame(self) -> int:
        return sample_latency_frame(self.spec, self.sample_float())


def parse_latency_distribution(raw: LatencyDistributionSpec | Mapping[str, Any] | str) -> LatencyDistributionSpec:
    if isinstance(raw, LatencyDistributionSpec):
        validate_latency_distribution(raw)
        return raw
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

    distribution_type = str(_require_field(raw, "type")).strip().lower()
    if distribution_type not in SUPPORTED_LATENCY_DISTRIBUTIONS:
        raise ValueError(
            "latency distribution type must be one of "
            f"{sorted(SUPPORTED_LATENCY_DISTRIBUTIONS)}"
        )

    min_frame = _coerce_integer_frame_bound(_require_field(raw, "min"), "latency.train_distribution.min")
    max_frame = _coerce_integer_frame_bound(_require_field(raw, "max"), "latency.train_distribution.max")
    if min_frame > max_frame:
        raise ValueError("latency.train_distribution.min must be <= latency.train_distribution.max")

    if distribution_type == "normal":
        mean = _coerce_finite_float(_require_field(raw, "mean"), "latency.train_distribution.mean")
        std = _coerce_finite_float(_require_field(raw, "std"), "latency.train_distribution.std")
        if std < 0:
            raise ValueError("latency.train_distribution.std must be non-negative")
        spec = LatencyDistributionSpec(
            type=distribution_type,
            mean=mean,
            std=std,
            min=min_frame,
            max=max_frame,
        )
    else:
        spec = LatencyDistributionSpec(
            type=distribution_type,
            mean=None,
            std=None,
            min=min_frame,
            max=max_frame,
        )
    validate_latency_distribution(spec)
    return spec


def validate_latency_distribution(raw: LatencyDistributionSpec | Mapping[str, Any] | str) -> None:
    if not isinstance(raw, LatencyDistributionSpec):
        parse_latency_distribution(raw)
        return
    if raw.type not in SUPPORTED_LATENCY_DISTRIBUTIONS:
        raise ValueError(
            "latency distribution type must be one of "
            f"{sorted(SUPPORTED_LATENCY_DISTRIBUTIONS)}"
        )
    if int(raw.min) < 0 or int(raw.max) < 0:
        raise ValueError("latency distribution frame bounds must be non-negative")
    if int(raw.min) > int(raw.max):
        raise ValueError("latency.train_distribution.min must be <= latency.train_distribution.max")
    if raw.type == "normal":
        if raw.mean is None:
            raise ValueError("Missing required field: latency.train_distribution.mean")
        if raw.std is None:
            raise ValueError("Missing required field: latency.train_distribution.std")
        _coerce_finite_float(raw.mean, "latency.train_distribution.mean")
        std = _coerce_finite_float(raw.std, "latency.train_distribution.std")
        if std < 0:
            raise ValueError("latency.train_distribution.std must be non-negative")


def make_latency_distribution_sampler(
    raw: LatencyDistributionSpec | Mapping[str, Any] | str,
    *,
    seed: int | None = None,
) -> LatencyDistributionSampler:
    return LatencyDistributionSampler(raw, seed=seed)


def sample_latency_frame(raw: LatencyDistributionSpec | Mapping[str, Any] | str, sampled_value: float) -> int:
    spec = parse_latency_distribution(raw)
    return round_half_up_frame(sampled_value, min_frame=spec.min, max_frame=spec.max)


def round_half_up_frame(value: float, *, min_frame: int, max_frame: int) -> int:
    min_frame = _coerce_integer_frame_bound(min_frame, "min_frame")
    max_frame = _coerce_integer_frame_bound(max_frame, "max_frame")
    if min_frame > max_frame:
        raise ValueError("min_frame must be <= max_frame")
    sampled = _coerce_finite_float(value, "latency sample")
    clamped = min(max(sampled, float(min_frame)), float(max_frame))
    return max(0, int(math.floor(clamped + 0.5)))


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
    return int(parse_latency_distribution(raw).max)


def distribution_frames_to_ms_latency_config(
    raw: LatencyDistributionSpec | Mapping[str, Any] | str,
    *,
    fps_env: float,
    seed: int | None = None,
) -> dict[str, Any]:
    spec = parse_latency_distribution(raw)
    fps = _coerce_finite_float(fps_env, "env.fps_env")
    if fps <= 0:
        raise ValueError("env.fps_env must be positive")
    frame_ms = 1000.0 / fps
    if spec.type == "normal":
        assert spec.mean is not None
        assert spec.std is not None
        config: dict[str, Any] = {
            "type": "normal_distribution",
            "mean_latency_ms": float(spec.mean) * frame_ms,
            "std_latency_ms": float(spec.std) * frame_ms,
            "min_latency_ms": float(spec.min) * frame_ms,
            "max_latency_ms": float(spec.max) * frame_ms,
        }
    elif spec.type == "uniform":
        config = {
            "type": "uniform_distribution",
            "min_latency_ms": float(spec.min) * frame_ms,
            "max_latency_ms": float(spec.max) * frame_ms,
        }
    else:
        raise ValueError(f"Unsupported latency distribution type: {spec.type}")
    if seed is not None:
        config["seed"] = int(seed)
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


def _require_field(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field: latency.train_distribution.{key}")
    return raw[key]


def _coerce_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _coerce_integer_frame_bound(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer frame bound")
    number = _coerce_finite_float(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    rounded = int(number)
    if abs(number - rounded) > 1e-9:
        raise ValueError(f"{field_name} must be an integer frame bound")
    return rounded


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
