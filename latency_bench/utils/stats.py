from __future__ import annotations

import math
from collections.abc import Sequence


def percentile_linear(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def autocorrelation(values: Sequence[float], lag: int) -> float | None:
    """Sample autocorrelation at `lag`; None when the lag exceeds the series length."""
    if lag >= len(values):
        return None
    series_mean = mean(values)
    variance = sum((value - series_mean) ** 2 for value in values)
    if variance == 0.0:
        return 0.0
    return sum(
        (values[index] - series_mean) * (values[index - lag] - series_mean)
        for index in range(lag, len(values))
    ) / variance


def std(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    series_mean = mean(values)
    variance = sum((value - series_mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))
