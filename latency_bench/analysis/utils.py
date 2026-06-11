from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_latency_distribution_plot(samples_ms: Iterable[float], output_path: str | Path) -> None:
    values = np.asarray([float(sample) for sample in samples_ms], dtype=float)
    lower = float(values.min())
    upper = float(values.max())
    pad = max((upper - lower) * 0.20, 1.0)
    grid = np.linspace(lower - pad, upper + pad, 420)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(values, bins=min(40, max(8, int(np.sqrt(values.size)))), density=True, color="#D88A86", alpha=0.28)
    if values.size > 1 and float(np.ptp(values)) > 0:
        ax.plot(grid, _kde(values, grid), color="#A54762", lw=2.2)
    else:
        ax.axvline(float(values[0]), color="#A54762", lw=2.2)
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Latency distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    std = np.std(values, ddof=1)
    bandwidth = 1.6 * 1.06 * std * values.size ** (-1 / 5)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
