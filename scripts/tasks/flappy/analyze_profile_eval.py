"""Summarize matched Flappy profile evaluations from episode_metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


def read_episodes(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    seeds = [row["metadata"]["episode_seed"] for row in rows]
    if len(rows) != 100 or len(set(seeds)) != 100:
        raise ValueError(f"{path}: expected 100 episodes with distinct seeds")
    return dict(zip(seeds, rows))


def paired_interval(candidate: dict[int, dict], control: dict[int, dict], *, seed: int = 42) -> list[float]:
    if candidate.keys() != control.keys():
        raise ValueError("evaluation episode seeds do not match")
    differences = np.array(
        [candidate[key]["episode_return"] - control[key]["episode_return"] for key in sorted(candidate)]
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(10_000, len(differences)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def summarize(path: Path, rows: dict[int, dict], *, trained: bool) -> dict:
    ordered = [rows[key] for key in sorted(rows)]
    submitted = sum(row["metadata"]["submitted_observation_frames"] for row in ordered)
    dropped = sum(row["metadata"]["dropped_observation_count"] for row in ordered)
    result = {
        "episode_metrics": str(path.resolve()),
        "episode_count": len(ordered),
        "mean_return": mean(row["episode_return"] for row in ordered),
        "median_return": median(row["episode_return"] for row in ordered),
        "mean_game_score": mean(row["game_score"] for row in ordered),
        "median_game_score": median(row["game_score"] for row in ordered),
        "mean_survival_steps": mean(row["survival_steps"] for row in ordered),
        "median_survival_steps": median(row["survival_steps"] for row in ordered),
        "admitted_observation_fraction": submitted / (submitted + dropped),
        "dropped_observation_fraction": dropped / (submitted + dropped),
        "dropped_action_fraction": sum(row["num_dropped_actions"] for row in ordered)
        / sum(row["num_actions"] for row in ordered),
    }
    if trained:
        checkpoints = {row["metadata"]["checkpoint_path"] for row in ordered}
        steps = {row["metadata"]["train_env_steps"] for row in ordered}
        if (len(checkpoints) != 1 or not Path(next(iter(checkpoints))).is_absolute()
                or len(steps) != 1 or next(iter(steps)) is None or next(iter(steps)) <= 0):
            raise ValueError(f"{path}: inconsistent checkpoint path or training step")
        result["checkpoint_path"] = next(iter(checkpoints))
        result["training_step"] = next(iter(steps))
    return result


def analyze(best_path: Path, final_path: Path, noop_path: Path, random_path: Path,
            milestones: list[Path]) -> dict:
    best = read_episodes(best_path)
    final = read_episodes(final_path)
    noop = read_episodes(noop_path)
    random = read_episodes(random_path)
    noop_ci = paired_interval(best, noop)
    random_ci = paired_interval(best, random)
    points = []
    for path in milestones:
        rows = read_episodes(path)
        if rows.keys() != best.keys():
            raise ValueError("milestone episode seeds do not match best evaluation")
        points.append(summarize(path, rows, trained=True))
    if final.keys() != best.keys():
        raise ValueError("final episode seeds do not match best evaluation")
    return {
        "best": summarize(best_path, best, trained=True),
        "final": summarize(final_path, final, trained=True),
        "noop": summarize(noop_path, noop, trained=False),
        "random": summarize(random_path, random, trained=False),
        "best_minus_noop_mean_return_ci95": noop_ci,
        "best_minus_random_mean_return_ci95": random_ci,
        "learned_control": noop_ci[0] > 0 and random_ci[0] > 0
        and median(row["game_score"] for row in best.values()) >= 1,
        "milestones": points,
    }


def plot_curve(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(report["milestones"], key=lambda point: point["training_step"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([point["training_step"] / 1e6 for point in points],
            [point["mean_return"] for point in points], marker="o", label="Milestones")
    ax.scatter([report["final"]["training_step"] / 1e6], [report["final"]["mean_return"]],
               marker="s", label="Final")
    ax.set(xlabel="Training steps (millions)", ylabel="Mean return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--milestone", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.best, args.final, args.noop, args.random, args.milestone)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    plot_curve(report, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
