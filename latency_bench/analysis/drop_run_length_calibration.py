"""Calibrate a simulated latency replay against the real-time measure run.

Compares a real measure run dir against a simulated replay run dir (same profile)
on the two acceptance targets for the temporal latency model plus two diagnostics:

1. Consecutive observation-drop run-length distribution (Wasserstein + KS).
2. Overall observation-drop rate.
3. Episode reward gap (mean / std).
4. worker_service autocorrelation (ACF) at lags 1..L.

Both run dirs are expected to contain ``steps.jsonl``, ``actions.jsonl`` and
``episode_metrics.jsonl`` as written by the trajectory logger.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from scipy.stats import ks_2samp, wasserstein_distance

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.latency.burst_profile import _action_records_with_episode_ids, _worker_service_ms
from latency_bench.utils.io import load_jsonl_records, write_json


def compare_run_dirs(real_run_dir: str | Path, sim_run_dir: str | Path, *, max_lag: int = 10) -> dict[str, Any]:
    real = _load_run(real_run_dir)
    sim = _load_run(sim_run_dir)

    real_runs = _drop_run_lengths(real["steps"])
    sim_runs = _drop_run_lengths(sim["steps"])
    real_returns = _episode_returns(real["metrics"])
    sim_returns = _episode_returns(sim["metrics"])
    real_acf = _acf(_worker_service_sequences(real["actions"], real["metrics"], _real_service_frames), max_lag)
    sim_acf = _acf(_worker_service_sequences(sim["actions"], sim["metrics"], _sim_service_frames), max_lag)

    return {
        "drop_run_length": _distribution_comparison(real_runs, sim_runs),
        "drop_rate": _scalar_comparison(_drop_rate(real["steps"]), _drop_rate(sim["steps"])),
        "reward": {
            "real": _moments(real_returns),
            "sim": _moments(sim_returns),
            "mean_gap": _abs_gap(_safe_mean(real_returns), _safe_mean(sim_returns)),
            "std_gap": _abs_gap(_safe_pstdev(real_returns), _safe_pstdev(sim_returns)),
        },
        "worker_service_acf": {
            "real": real_acf,
            "sim": sim_acf,
            "l2_gap": _acf_l2_gap(real_acf, sim_acf),
        },
        "wasserstein_drop_run_length": _wasserstein(real_runs, sim_runs),
        "drop_rate_gap": _abs_gap(_drop_rate(real["steps"]), _drop_rate(sim["steps"])),
        "reward_gap": _abs_gap(_safe_mean(real_returns), _safe_mean(sim_returns)),
        "acf_l2_gap": _acf_l2_gap(real_acf, sim_acf),
    }


def _load_run(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(run_dir)
    return {
        "steps": load_jsonl_records(path / "steps.jsonl"),
        "actions": load_jsonl_records(path / "actions.jsonl"),
        "metrics": load_jsonl_records(path / "episode_metrics.jsonl"),
    }


def _steps_by_episode(step_records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    steps_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in step_records:
        steps_by_episode[int(record["episode_id"])].append(record)
    return steps_by_episode


def _drop_run_lengths(step_records: list[dict[str, Any]]) -> list[int]:
    runs: list[int] = []
    for _episode_id, records in sorted(_steps_by_episode(step_records).items()):
        current = 0
        for record in sorted(records, key=lambda item: int(item["env_step"])):
            info = record["info"]
            if info.get("observation_dropped"):
                current += 1
            elif info.get("observation_submitted"):
                if current:
                    runs.append(current)
                current = 0
        if current:
            runs.append(current)
    return runs


def _drop_rate(step_records: list[dict[str, Any]]) -> float | None:
    submitted = 0
    dropped = 0
    for record in step_records:
        info = record["info"]
        if info.get("observation_dropped"):
            dropped += 1
        elif info.get("observation_submitted"):
            submitted += 1
    total = submitted + dropped
    return dropped / total if total else None


def _episode_returns(metrics_records: list[dict[str, Any]]) -> list[float]:
    return [float(record["episode_return"]) for record in metrics_records]


def _worker_service_sequences(
    action_records: list[dict[str, Any]],
    metrics_records: list[dict[str, Any]],
    service_frames: Callable[[dict[str, Any], float], int],
) -> list[list[int]]:
    frame_ms_by_episode = {
        int(record["episode_id"]): float(record["metadata"]["frame_ms"])
        for record in metrics_records
    }
    sequences: dict[int, list[int]] = defaultdict(list)
    for record, episode_id in _action_records_with_episode_ids(action_records, metrics_records):
        sequences[episode_id].append(service_frames(record, frame_ms_by_episode[episode_id]))
    return [sequences[episode_id] for episode_id in sorted(sequences)]


def _real_service_frames(record: dict[str, Any], frame_ms: float) -> int:
    return ms_to_raw_frames(_worker_service_ms(record), env_fps=1000.0 / frame_ms)


def _sim_service_frames(record: dict[str, Any], frame_ms: float) -> int:
    return int(record["metadata"]["worker_service_raw_frames"])


def _acf(sequences: list[list[int]], max_lag: int) -> list[float | None]:
    values = [float(value) for sequence in sequences for value in sequence]
    if len(values) < 2:
        return [None] * max_lag
    grand_mean = mean(values)
    variance = sum((value - grand_mean) ** 2 for value in values)
    if variance <= 0:
        return [None] * max_lag
    acf: list[float | None] = []
    for lag in range(1, max_lag + 1):
        covariance = 0.0
        count = 0
        for sequence in sequences:
            for index in range(len(sequence) - lag):
                covariance += (sequence[index] - grand_mean) * (sequence[index + lag] - grand_mean)
                count += 1
        acf.append(covariance / variance if count else None)
    return acf


def _acf_l2_gap(real_acf: list[float | None], sim_acf: list[float | None]) -> float | None:
    squared = [
        (real - sim) ** 2
        for real, sim in zip(real_acf, sim_acf)
        if real is not None and sim is not None
    ]
    return sum(squared) ** 0.5 if squared else None


def _distribution_comparison(real: list[int], sim: list[int]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "real": _moments(real),
        "sim": _moments(sim),
        "wasserstein": _wasserstein(real, sim),
    }
    if real and sim:
        statistic, pvalue = ks_2samp(real, sim)
        comparison["ks_statistic"] = float(statistic)
        comparison["ks_pvalue"] = float(pvalue)
    else:
        comparison["ks_statistic"] = None
        comparison["ks_pvalue"] = None
    return comparison


def _scalar_comparison(real: float | None, sim: float | None) -> dict[str, Any]:
    return {"real": real, "sim": sim, "gap": _abs_gap(real, sim)}


def _wasserstein(real: list[int], sim: list[int]) -> float | None:
    if not real or not sim:
        return None
    return float(wasserstein_distance(real, sim))


def _moments(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "p50": float(_percentile(ordered, 50.0)),
        "p90": float(_percentile(ordered, 90.0)),
        "max": float(ordered[-1]),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile / 100.0 * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _abs_gap(real: float | None, sim: float | None) -> float | None:
    if real is None or sim is None:
        return None
    return abs(real - sim)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare a simulated replay run against the real measure run.")
    parser.add_argument("--real-run-dir", type=Path, required=True)
    parser.add_argument("--sim-run-dir", type=Path, required=True)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = compare_run_dirs(args.real_run_dir, args.sim_run_dir, max_lag=args.max_lag)
    if args.out is not None:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
