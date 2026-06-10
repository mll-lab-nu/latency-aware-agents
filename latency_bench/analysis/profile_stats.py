from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def compute_latency_profile_summary(
    samples_ms: Iterable[float],
    env_fps: float,
    tail_frame_thresholds: tuple[int | float, ...] = (1, 2, 3),
) -> dict[str, float | int]:
    """Compute latency profile statistics.

    Percentiles use NumPy's deterministic linear interpolation method.
    ``std_ms`` is the population standard deviation (``ddof=0``).
    """
    samples = np.asarray([float(sample) for sample in samples_ms], dtype=np.float64)
    frame_ms = 1000.0 / env_fps
    p50_ms, p90_ms, p95_ms, p99_ms = np.percentile(samples, [50, 90, 95, 99], method="linear")
    mean_ms = float(np.mean(samples))
    summary: dict[str, float | int] = {
        "n_samples": int(samples.size),
        "mean_ms": mean_ms,
        "std_ms": float(np.std(samples, ddof=0)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "p50_ms": float(p50_ms),
        "p90_ms": float(p90_ms),
        "p95_ms": float(p95_ms),
        "p99_ms": float(p99_ms),
        "frame_ms": frame_ms,
        "mean_effective_frames": mean_ms / frame_ms,
        "p50_frames": float(p50_ms) / frame_ms,
        "p90_frames": float(p90_ms) / frame_ms,
        "p95_frames": float(p95_ms) / frame_ms,
        "p99_frames": float(p99_ms) / frame_ms,
    }

    for threshold in tail_frame_thresholds:
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or threshold_value < 0:
            raise ValueError("tail_frame_thresholds must contain finite non-negative values")
        threshold_key = _format_threshold_key(threshold_value)
        suffix = "frame" if threshold_key == "1" else "frames"
        key = f"prob_latency_gt_{threshold_key}_{suffix}"
        summary[key] = float(np.mean(samples > threshold_value * frame_ms))

    return summary


def _format_threshold_key(threshold: float) -> str:
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "_")
