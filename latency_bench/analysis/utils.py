from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_latency_distribution_plot(samples_ms: Iterable[float], output_path: str | Path) -> None:
    values = np.asarray(list(samples_ms), dtype=np.float64)
    lower, upper = np.percentile(values, [0.1, 99.9])
    pad = max((upper - lower) * 0.08, 1.0)
    grid = np.linspace(lower - pad, upper + pad, 520)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if lower == upper:
        ax.axvline(lower, color="#A54762", lw=2.2)
    else:
        ax.plot(grid, _kde(values, grid), color="#A54762", lw=2.2)
    ax.set_xlabel("Observation-to-action latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Observation-to-action latency profile")
    ax.set_xlim(grid[0], grid[-1])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    std = np.std(values, ddof=1)
    bandwidth = 1.6 * 1.06 * std * values.size ** (-1 / 5)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
