#!/usr/bin/env python3
"""Summarize the formal MIKASA conditioned-H8 evaluation matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


MODELS = ("qwenoft", "qwenpi_v3", "qwengr00t")
DISPLAY_NAMES = {
    "qwenoft": "QwenOFT",
    "qwenpi_v3": "QwenPI v3",
    "qwengr00t": "QwenGR00T",
}
REUSED_ZERO = {
    "qwenoft": {"zero": 25, "profile": 1},
    "qwenpi_v3": {"zero": 34, "profile": 14},
    "qwengr00t": {"zero": 31, "profile": 18},
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _setting_summary(run_dir: Path, setting: str) -> dict:
    result_name = {
        "profile": "eval_profile.json",
        "fixed-2": "eval_fixed_2f.json",
        "zero": "eval_l0.json",
    }[setting]
    result = json.loads((run_dir / f"inference/{result_name}").read_text(encoding="utf-8"))
    evaluation = run_dir / f"evals/final-{setting}"
    actions = _jsonl(evaluation / "actions.jsonl")
    steps = _jsonl(evaluation / "steps.jsonl")
    latencies = np.asarray([row["latency"]["latency_ms"] for row in actions], dtype=np.float64)
    head_counts = Counter(
        min(step["info"]["chunk_age_raw_frames"], 7)
        for step in steps
        if not step["active_action"]["is_noop"]
    )
    episodes = result["episodes"]
    return {
        "success_count": int(result["acceptance"]["success_count"]),
        "episodes": result["acceptance"]["episode_count"],
        "success_rate": result["acceptance"]["success_rate"],
        "return_mean": result["mean_return"],
        "return_std": result["std_return"],
        "episode_length_mean": result["mean_length"],
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_std": float(latencies.std()),
        "submitted_requests": len(actions),
        "dropped_observations": sum(
            episode["metadata"]["dropped_observation_count"] for episode in episodes
        ),
        "underflow_count": sum(
            episode["metadata"]["underflow_count"] for episode in episodes
        ),
        "underflow_frames": sum(
            episode["metadata"]["underflow_frames"] for episode in episodes
        ),
        "executed_head_counts": {
            str(head): head_counts.get(head, 0) for head in range(8)
        },
        "chunk_execution_audit": result["chunk_execution_audit"],
    }


def summarize(root: Path) -> dict:
    zero_root = root.parent / "h8-zero-latency"
    zero_fixed_results = {
        model: _setting_summary(zero_root / model, "fixed-2")
        for model in MODELS
    }
    new_results = {}
    for model in MODELS:
        new_results[model] = {
            "profile-trained_to_profile": _setting_summary(
                root / f"profile-latency/{model}", "profile"
            ),
            "fixed-trained_to_fixed-2": _setting_summary(
                root / f"fixed-2/{model}", "fixed-2"
            ),
            "fixed-trained_to_zero": _setting_summary(
                root / f"fixed-2/{model}", "zero"
            ),
        }
    result = {
        "protocol": {
            "episodes": 50,
            "seeds": [4242424242, 4242424291],
            "max_transitions": 60,
            "control_hz": 20,
            "simulated_worker_capacity": 1,
            "profile_latency_seed": 0,
            "first_action_freeze": False,
            "primary_threshold": "success_once >= 25/50",
        },
        "reused_zero_trained": REUSED_ZERO,
        "zero-trained_to_fixed-2": zero_fixed_results,
        "new_results": new_results,
    }
    (root / "evaluation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for model in MODELS:
        for label, metrics in new_results[model].items():
            rows.append(
                f"| {DISPLAY_NAMES[model]} | {label} | "
                f"{metrics['success_count']}/{metrics['episodes']} | "
                f"{metrics['return_mean']:.4f} ± {metrics['return_std']:.4f} | "
                f"{metrics['episode_length_mean']:.2f} | "
                f"{metrics['latency_ms_mean']:.3f} ± {metrics['latency_ms_std']:.3f} | "
                f"{metrics['dropped_observations']} | "
                f"{metrics['underflow_count']}/{metrics['underflow_frames']} |"
            )
    reused_rows = [
        f"| {DISPLAY_NAMES[model]} | {values['zero']}/50 | "
        f"{zero_fixed_results[model]['success_count']}/50 | {values['profile']}/50 |"
        for model, values in REUSED_ZERO.items()
    ]
    report = "\n".join(
        [
            "# MIKASA conditioned H8 formal evaluation",
            "",
            "Protocol: 50 episodes, seeds 4242424242–4242424291, 60 transitions, "
            "20 Hz, capacity 1, profile seed 0, no first-action freeze.",
            "",
            "## New results",
            "",
            "| Model | Transfer | Success | Return | Length | Latency ms | Dropped obs | Underflow count/frames |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The primary threshold (success_once ≥ 25/50) applies only to "
            "profile-trained_to_profile and fixed-trained_to_fixed-2.",
            "",
            "## Zero-trained references",
            "",
            "The zero and profile columns are reused formal results; fixed-2 is newly evaluated.",
            "",
            "| Model | zero → zero | zero → fixed-2 | zero → profile |",
            "|---|---:|---:|---:|",
            *reused_rows,
            "",
            "Full latency, drop, underflow, executed-head, and chunk-audit fields are in "
            "`evaluation_report.json`.",
            "",
        ]
    )
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))


if __name__ == "__main__":
    main()
