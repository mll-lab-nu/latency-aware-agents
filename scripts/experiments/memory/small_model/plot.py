from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.experiments import plot_colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update(plot_colors.RC_COLORS)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.csv"
STYLE = Path(plot_colors.__file__).with_name("plot_style.mplstyle")
BOOTSTRAP_DRAWS = 10000

plt.style.use(STYLE)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    title: str
    capped_windows: tuple[int, ...]
    uncapped_windows: tuple[int, ...]
    capped_horizon: int
    uncapped_horizon: int


@dataclass(frozen=True)
class RegimeSpec:
    regime_ids: tuple[str, ...]
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class StatisticSpec:
    statistic_id: str
    y_label: str
    aggregate: Callable[..., np.ndarray | float]


ENVS = (
    EnvSpec("flappy", "Flappy Bird", tuple(range(4, 9)), tuple(range(1, 9)), 3600, 36000),
    EnvSpec("demon-attack", "Demon Attack", tuple(range(4, 9)), tuple(range(4, 13)), 3600, 36000),
    EnvSpec("deadly-corridor", "Deadly Corridor", (1, 2, 4, 6, 8, 10, 12, 14, 16), (1, 2, 4, 6, 8, 10, 12, 14, 16), 30000, 30000),
)

REGIMES = (
    RegimeSpec(("fixed_l2", "fixed_l4"), "Fixed delay", plot_colors.PROFILE_COLORS["fixed"], "o"),
    RegimeSpec(("uniform_u1_3", "uniform_u2_6"), "Uniform delay", plot_colors.PROFILE_COLORS["narrow_uniform"], "s"),
)


def middle_three_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    sorted_values = np.sort(values, axis=axis)
    if axis is None:
        return np.mean(sorted_values[1:-1])
    middle_indices = np.arange(1, sorted_values.shape[axis] - 1)
    return np.mean(np.take(sorted_values, middle_indices, axis=axis), axis=axis)


STATISTICS = (
    StatisticSpec("mean", "Mean return", np.mean),
    StatisticSpec("median", "Median evaluation return", np.median),
    StatisticSpec("trimmed_mean", "20% trimmed mean return", middle_three_mean),
    StatisticSpec("iqm", "Interquartile mean return", middle_three_mean),
)


def read_rows(data: Path = DATA) -> list[dict[str, str]]:
    with data.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_means(
    rows: Sequence[dict[str, str]],
    *,
    horizon_variant: str,
    env_id: str,
    context_window: int,
    regime_ids: Sequence[str],
) -> np.ndarray:
    values_by_seed: defaultdict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["horizon_variant"] == horizon_variant
            and row["env_id"] == env_id
            and int(row["context_window"]) == context_window
            and row["latency_regime"] in regime_ids
        ):
            values_by_seed[int(row["seed"])].append(float(row["value"]))
    return np.asarray(
        [np.mean(values_by_seed[seed]) for seed in sorted(values_by_seed)],
        dtype=float,
    )


def statistic_and_ci(
    values: np.ndarray,
    statistic: StatisticSpec,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    bootstrap_statistics = statistic.aggregate(values[indices], axis=1)
    ci_lo, ci_hi = np.percentile(bootstrap_statistics, [2.5, 97.5])
    return float(statistic.aggregate(values)), float(ci_lo), float(ci_hi)


def plot_variant(
    rows: Sequence[dict[str, str]],
    horizon_variant: str,
    statistic: StatisticSpec,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)

    for panel_label, ax, env in zip(("(a)", "(b)", "(c)"), axes, ENVS):
        windows = env.capped_windows if horizon_variant == "capped" else env.uncapped_windows
        horizon = env.capped_horizon if horizon_variant == "capped" else env.uncapped_horizon
        for regime in REGIMES:
            summaries = [
                statistic_and_ci(
                    seed_means(
                        rows,
                        horizon_variant=horizon_variant,
                        env_id=env.env_id,
                        context_window=context_window,
                        regime_ids=regime.regime_ids,
                    ),
                    statistic,
                )
                for context_window in windows
            ]
            centers = np.asarray([summary[0] for summary in summaries])
            ci_lo = np.asarray([summary[1] for summary in summaries])
            ci_hi = np.asarray([summary[2] for summary in summaries])
            ax.fill_between(windows, ci_lo, ci_hi, color=regime.color, alpha=0.14, linewidth=0)
            ax.plot(
                windows,
                centers,
                color=regime.color,
                marker=regime.marker,
                markersize=3.2,
                linewidth=0.8,
            )

        ax.set_title(f"{env.title}\n({horizon:,}-step horizon)")
        ax.set_xticks(windows)
        ax.set_xlabel(r"Context length $C$ (frames)")
        ax.grid(alpha=0.25)
        ax.text(-0.14, 1.04, panel_label, transform=ax.transAxes, fontsize=8)

    axes[0].set_ylabel(statistic.y_label)
    handles = [
        Line2D(
            [0],
            [0],
            color=regime.color,
            marker=regime.marker,
            markersize=3.2,
            linewidth=0.8,
            label=regime.label,
        )
        for regime in REGIMES
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=2, columnspacing=1.6, handlelength=1.8)

    pdf_out = output_dir / f"fig_{horizon_variant}.pdf"
    png_out = output_dir / f"fig_{horizon_variant}.png"
    fig.savefig(pdf_out)
    fig.savefig(png_out, dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_out} and {png_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.data)
    for statistic in STATISTICS:
        output_dir = args.output_dir / statistic.statistic_id
        plot_variant(rows, "capped", statistic, output_dir)
        plot_variant(rows, "uncapped", statistic, output_dir)

    default_statistic = STATISTICS[0]
    plot_variant(rows, "capped", default_statistic, args.output_dir)
    plot_variant(rows, "uncapped", default_statistic, args.output_dir)


if __name__ == "__main__":
    main()
