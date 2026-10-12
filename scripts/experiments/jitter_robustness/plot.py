from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.experiments import plot_colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

plt.rcParams.update(plot_colors.RC_COLORS)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.csv"
STYLE = Path(plot_colors.__file__).with_name("plot_style.mplstyle")

plt.style.use(STYLE)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    title: str
    train_labels: tuple[str, ...]
    eval_labels: tuple[str, ...]


@dataclass(frozen=True)
class CellStats:
    mean_return: float
    std_return: float
    seed_count: int


ENVS = (
    EnvSpec(
        "flappy",
        "Flappy Bird",
        ("1", "2", "3", "N(2,1)", "U(1,3)", "Mix N(2,1)", "Mix U(1,3)"),
        ("1", "2", "3", "N(2,1)", "U(1,3)"),
    ),
    EnvSpec(
        "demon-attack",
        "Demon Attack",
        ("2", "4", "6", "N(4,1)", "U(2,6)", "Mix N(4,1)", "Mix U(2,6)"),
        ("2", "4", "6", "N(4,1)", "U(2,6)"),
    ),
    EnvSpec(
        "deadly-corridor",
        "Deadly Corridor",
        ("2", "4", "6", "N(4,1)", "U(2,6)", "Mix N(4,1)", "Mix U(2,6)"),
        ("2", "4", "6", "N(4,1)", "U(2,6)"),
    ),
)


def read_rows(data: Path = DATA) -> list[dict[str, str]]:
    with data.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_statistics(
    rows: list[dict[str, str]],
    horizon_variant: str,
) -> dict[tuple[str, str, str], CellStats]:
    seed_means: dict[tuple[str, str, str, str], float] = {}
    episode_returns: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        dict
    )
    aggregates: dict[tuple[str, str, str], CellStats] = {}

    for row in rows:
        if row["horizon_variant"] != horizon_variant:
            continue
        cell = (row["env_id"], row["train_label"], row["eval_label"])
        if row["record_type"] == "seed_mean_return":
            seed = (*cell, row["seed"])
            if seed in seed_means:
                raise ValueError(f"duplicate seed mean: {seed}")
            seed_means[seed] = float(row["value"])
        elif row["record_type"] == "episode_return":
            seed = (*cell, row["seed"])
            if row["episode_id"] in episode_returns[seed]:
                raise ValueError(f"duplicate episode: {(*seed, row['episode_id'])}")
            episode_returns[seed][row["episode_id"]] = float(row["value"])
        elif row["record_type"] == "cell_aggregate":
            if cell in aggregates:
                raise ValueError(f"duplicate cell aggregate: {cell}")
            seeds = row["seeds"].split(",")
            if int(row["seed_count"]) != 3 or len(seeds) != 3 or len(set(seeds)) != 3:
                raise ValueError(f"expected three unique aggregate seeds for {cell}")
            aggregates[cell] = CellStats(
                mean_return=float(row["value"]),
                std_return=float(row["std_return"]),
                seed_count=int(row["seed_count"]),
            )
        else:
            raise ValueError(f"unknown record type: {row['record_type']}")

    for seed, values in episode_returns.items():
        if seed in seed_means:
            raise ValueError(f"mixed seed statistical units: {seed}")
        seed_means[seed] = statistics.mean(values.values())

    values_by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (env_id, train_label, eval_label, _seed), value in sorted(seed_means.items()):
        values_by_cell[(env_id, train_label, eval_label)].append(value)
    for cell, values in values_by_cell.items():
        if cell in aggregates:
            raise ValueError(f"mixed cell statistical units: {cell}")
        if len(values) != 3:
            raise ValueError(f"expected three seeds for {cell}, found {len(values)}")
        aggregates[cell] = CellStats(
            mean_return=statistics.mean(values),
            std_return=statistics.stdev(values),
            seed_count=len(values),
        )
    return aggregates


def matrices_for_spec(
    statistics_by_cell: dict[tuple[str, str, str], CellStats],
    spec: EnvSpec,
) -> np.ndarray:
    values = np.empty((len(spec.train_labels), len(spec.eval_labels)))
    for y, train_label in enumerate(spec.train_labels):
        for x, eval_label in enumerate(spec.eval_labels):
            stats = statistics_by_cell[(spec.env_id, train_label, eval_label)]
            values[y, x] = stats.mean_return
    return values


def plot_variant(
    rows: list[dict[str, str]], horizon_variant: str, output_dir: Path = HERE
) -> None:
    statistics_by_cell = cell_statistics(rows, horizon_variant)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.02, wspace=0.005, hspace=0.02)
    cmap = plot_colors.HEATMAP.copy()

    for ax, spec in zip(axes, ENVS):
        log_values = np.log10(matrices_for_spec(statistics_by_cell, spec))
        ax.imshow(
            log_values,
            origin="lower",
            cmap=cmap,
            norm=Normalize(vmin=log_values.min(), vmax=log_values.max()),
            aspect="equal",
            interpolation="nearest",
        )
        ax.grid(False)
        ax.set_title(spec.title)
        ax.set_xticks(np.arange(len(spec.eval_labels)))
        ax.set_yticks(np.arange(len(spec.train_labels)))
        ax.set_xticklabels(spec.eval_labels, rotation=30, ha="right")
        ax.set_yticklabels(spec.train_labels)
        ax.plot(
            [0, len(spec.eval_labels) - 1],
            [0, len(spec.eval_labels) - 1],
            color=plot_colors.EDGE,
            linewidth=0.6,
            linestyle="--",
            alpha=0.5,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(plot_colors.INK)
            spine.set_linewidth(0.6)

    axes[0].set_ylabel("Train latency (ticks)")
    fig.supxlabel("Eval latency (ticks)")

    color_scale = matplotlib.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    cbar = fig.colorbar(color_scale, ax=axes, fraction=0.025, pad=0.01, shrink=0.86)
    cbar.set_label("log10(mean return)")
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor(plot_colors.INK)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(output_dir / f"fig_{horizon_variant}.pdf")
    fig.savefig(output_dir / f"fig_{horizon_variant}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data)
    plot_variant(rows, "capped", args.output_dir)
    plot_variant(rows, "uncapped", args.output_dir)


if __name__ == "__main__":
    main()
