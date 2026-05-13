from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wasserstein_1d(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    """Compute exact 1D empirical Wasserstein distance for uniform samples."""

    a = sorted(_finite_float_samples(samples_a, "samples_a"))
    b = sorted(_finite_float_samples(samples_b, "samples_b"))
    n_a = len(a)
    n_b = len(b)

    points = sorted(set(a) | set(b))
    prev = points[0]
    index_a = 0
    index_b = 0
    cdf_a = 0.0
    cdf_b = 0.0
    distance = 0.0

    for point in points:
        distance += abs(cdf_a - cdf_b) * (point - prev)
        while index_a < n_a and a[index_a] <= point:
            index_a += 1
        while index_b < n_b and b[index_b] <= point:
            index_b += 1
        cdf_a = index_a / n_a
        cdf_b = index_b / n_b
        prev = point

    return float(distance)


def wasserstein_frame(samples_a_ms: Sequence[float], samples_b_ms: Sequence[float], fps: float) -> float:
    frame_ms = _frame_ms(fps)
    return wasserstein_1d(samples_a_ms, samples_b_ms) / frame_ms


def quantile_errors(
    samples_a_ms: Sequence[float],
    samples_b_ms: Sequence[float],
    fps: float,
    quantiles: Iterable[int | float] = (50, 90, 95, 99),
) -> dict[str, float]:
    frame_ms = _frame_ms(fps)
    a = _finite_float_samples(samples_a_ms, "samples_a_ms")
    b = _finite_float_samples(samples_b_ms, "samples_b_ms")
    result: dict[str, float] = {}
    max_error_frames = 0.0

    for quantile in quantiles:
        q = float(quantile)
        if not math.isfinite(q) or q < 0.0 or q > 100.0:
            raise ValueError("quantiles must be in [0, 100]")
        key = _format_quantile_key(q)
        qa = _linear_percentile(a, q)
        qb = _linear_percentile(b, q)
        error_ms = abs(qa - qb)
        error_frames = error_ms / frame_ms
        result[f"{key}_error_ms"] = float(error_ms)
        result[f"{key}_error_frames"] = float(error_frames)
        max_error_frames = max(max_error_frames, error_frames)

    result["max_quantile_error_frames"] = float(max_error_frames)
    return result


def tail_crossing_probability_errors(
    samples_a_ms: Sequence[float],
    samples_b_ms: Sequence[float],
    fps: float,
    frame_thresholds: Iterable[int | float] = (1, 2, 3),
) -> dict[str, float]:
    frame_ms = _frame_ms(fps)
    a = _finite_float_samples(samples_a_ms, "samples_a_ms")
    b = _finite_float_samples(samples_b_ms, "samples_b_ms")
    result: dict[str, float] = {}
    max_tail_error = 0.0

    for threshold in frame_thresholds:
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or threshold_value < 0.0:
            raise ValueError("frame_thresholds must contain finite non-negative values")
        suffix = _frame_threshold_suffix(threshold_value)
        threshold_ms = threshold_value * frame_ms
        prob_a = sum(sample > threshold_ms for sample in a) / len(a)
        prob_b = sum(sample > threshold_ms for sample in b) / len(b)
        error = abs(prob_a - prob_b)
        result[f"prob_a_gt_{suffix}"] = float(prob_a)
        result[f"prob_b_gt_{suffix}"] = float(prob_b)
        result[f"tail_error_gt_{suffix}"] = float(error)
        max_tail_error = max(max_tail_error, error)

    result["max_tail_error"] = float(max_tail_error)
    return result


def latency_distribution_match_metrics(
    real_samples_ms: Sequence[float],
    sim_samples_ms: Sequence[float],
    fps: float,
) -> dict[str, float]:
    w1_ms = wasserstein_1d(real_samples_ms, sim_samples_ms)
    metrics: dict[str, float] = {
        "w1_ms": float(w1_ms),
        "w1_frames": float(w1_ms / _frame_ms(fps)),
    }
    metrics.update(quantile_errors(real_samples_ms, sim_samples_ms, fps))
    metrics.update(tail_crossing_probability_errors(real_samples_ms, sim_samples_ms, fps))
    return metrics


def absolute_error(sim_value: float, real_value: float) -> float:
    sim = _finite_float(sim_value, "sim_value")
    real = _finite_float(real_value, "real_value")
    return abs(sim - real)


def normalized_error(
    sim_value: float,
    real_value: float,
    *,
    scale: float | None = None,
    fallback_to_realtime_denominator: bool = False,
    eps: float = 1e-8,
) -> dict[str, float | str]:
    eps_value = _finite_float(eps, "eps")
    if eps_value <= 0:
        raise ValueError("eps must be positive")

    abs_value = absolute_error(sim_value, real_value)
    if scale is not None and abs(_finite_float(scale, "scale")) > eps_value:
        denominator = abs(float(scale))
        denominator_mode = "task_scale"
    elif fallback_to_realtime_denominator:
        denominator = abs(_finite_float(real_value, "real_value")) + eps_value
        denominator_mode = "realtime_denominator"
    else:
        raise ValueError("scale is required unless fallback_to_realtime_denominator is true")

    return {
        "abs_error": float(abs_value),
        "norm_error": float(abs_value / denominator),
        "denominator": float(denominator),
        "denominator_mode": denominator_mode,
    }


def return_error_metrics(
    sim_return: float,
    real_return: float,
    *,
    return_scale: float | None = None,
    fallback_to_realtime_denominator: bool = False,
    eps: float = 1e-8,
) -> dict[str, float | str]:
    metrics = normalized_error(
        sim_return,
        real_return,
        scale=return_scale,
        fallback_to_realtime_denominator=fallback_to_realtime_denominator,
        eps=eps,
    )
    return {
        "abs_return_error": metrics["abs_error"],
        "norm_return_error": metrics["norm_error"],
        "return_denominator": metrics["denominator"],
        "return_denominator_mode": metrics["denominator_mode"],
    }


def length_error_metrics(
    sim_length: float,
    real_length: float,
    *,
    length_scale: float | None = None,
    fallback_to_realtime_denominator: bool = False,
    eps: float = 1e-8,
) -> dict[str, float | str]:
    metrics = normalized_error(
        sim_length,
        real_length,
        scale=length_scale,
        fallback_to_realtime_denominator=fallback_to_realtime_denominator,
        eps=eps,
    )
    return {
        "abs_length_error": metrics["abs_error"],
        "norm_length_error": metrics["norm_error"],
        "length_denominator": metrics["denominator"],
        "length_denominator_mode": metrics["denominator_mode"],
    }


def default_latency_equivalence_thresholds() -> dict[str, float]:
    return {
        "w1_frames": 0.1,
        "p50_p90_p95_error_frames": 0.25,
        "p99_error_frames": 0.5,
        "max_tail_error": 0.05,
        "norm_return_error": 0.1,
    }


def _finite_float_samples(samples: Sequence[float], name: str) -> list[float]:
    values = [_finite_float(sample, name) for sample in samples]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _finite_float(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must contain finite values")
    return numeric


def _frame_ms(fps: float) -> float:
    fps_value = _finite_float(fps, "fps")
    if fps_value <= 0:
        raise ValueError("fps must be positive and finite")
    return 1000.0 / fps_value


def _linear_percentile(samples: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), quantile, method="linear"))


def _format_quantile_key(quantile: float) -> str:
    if quantile.is_integer():
        return f"p{int(quantile)}"
    return f"p{str(quantile).replace('.', '_')}"


def _frame_threshold_suffix(threshold: float) -> str:
    threshold_key = str(int(threshold)) if threshold.is_integer() else str(threshold).replace(".", "_")
    unit = "frame" if threshold_key == "1" else "frames"
    return f"{threshold_key}_{unit}"


def _integer_like_values(values: Iterable[Any], name: str) -> list[int]:
    result: list[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain integer-like values")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must contain integer-like values")
        result.append(int(numeric))
    if not result:
        raise ValueError(f"{name} must contain at least one valid value")
    return result
