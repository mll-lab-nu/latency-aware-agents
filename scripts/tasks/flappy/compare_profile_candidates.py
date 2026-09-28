"""Compare Flappy checkpoints on matched 100-episode scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tasks.flappy.analyze_profile_eval import paired_interval, read_episodes, summarize


def compare(baseline20: Path, baseline25: Path, candidate20: Path, candidate25: Path, model: str) -> dict:
    pairs = {
        "20m": (baseline20, candidate20),
        "25m": (baseline25, candidate25),
    }
    report = {"model": model, "checkpoints": {}, "comparisons": {}}
    for step, (baseline_path, candidate_path) in pairs.items():
        baseline = read_episodes(baseline_path)
        candidate = read_episodes(candidate_path)
        base_summary = summarize(baseline_path, baseline, trained=True)
        candidate_summary = summarize(candidate_path, candidate, trained=True)
        ci = paired_interval(candidate, baseline)
        report["checkpoints"][step] = {"baseline": base_summary, "candidate": candidate_summary}
        report["comparisons"][step] = {
            "mean_return_gain": candidate_summary["mean_return"] - base_summary["mean_return"],
            "mean_return_ratio": candidate_summary["mean_return"] / base_summary["mean_return"],
            "paired_gain_ci95": ci,
        }

    candidate20_rows = read_episodes(candidate20)
    candidate25_rows = read_episodes(candidate25)
    report["comparisons"]["candidate_25m_minus_20m"] = {
        "paired_gain_ci95": paired_interval(candidate25_rows, candidate20_rows),
    }
    if model == "gr00t":
        report["meets_target"] = all(
            report["comparisons"][step]["mean_return_ratio"] >= 2
            and report["comparisons"][step]["paired_gain_ci95"][0] > 0
            for step in ("20m", "25m")
        )
    else:
        report["meets_target"] = (
            report["comparisons"]["25m"]["mean_return_ratio"] >= 1.2
            and report["comparisons"]["25m"]["paired_gain_ci95"][0] > 0
            and report["comparisons"]["candidate_25m_minus_20m"]["paired_gain_ci95"][1] >= 0
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gr00t", "oft"), required=True)
    parser.add_argument("--baseline20", type=Path, required=True)
    parser.add_argument("--baseline25", type=Path, required=True)
    parser.add_argument("--candidate20", type=Path, required=True)
    parser.add_argument("--candidate25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline20, args.baseline25, args.candidate20, args.candidate25, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparisons"], indent=2))
    print(f"meets_target={report['meets_target']}")


if __name__ == "__main__":
    main()
