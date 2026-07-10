from __future__ import annotations

import random
from bisect import bisect_left
from typing import Sequence


def sample_inverse_cdf(
    rng: random.Random,
    quantile_levels: Sequence[float],
    latency_ms: Sequence[float],
    *,
    quantile: float | None = None,
) -> float:
    """Draw one value by inverting an empirical CDF via linear interpolation."""
    quantile = rng.random() if quantile is None else float(quantile)
    index = bisect_left(quantile_levels, quantile)
    if index == 0:
        return float(latency_ms[0])
    if index >= len(latency_ms):
        return float(latency_ms[-1])
    lower_q = float(quantile_levels[index - 1])
    upper_q = float(quantile_levels[index])
    lower_latency = float(latency_ms[index - 1])
    upper_latency = float(latency_ms[index])
    weight = (quantile - lower_q) / (upper_q - lower_q)
    return lower_latency + weight * (upper_latency - lower_latency)
