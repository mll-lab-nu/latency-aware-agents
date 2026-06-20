from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from latency_bench.analysis.error_metrics import latency_distribution_match_metrics, wasserstein_1d
from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.utils.stats import mean, percentile_linear, std


ACF_LAGS = (1, 2, 5, 10, 20)


def roll_latency_sampler(
    *,
    sampler: Any,
    episode_lengths: Sequence[int],
) -> dict[str, list[float] | list[int]]:
    e2e_latency_ms: list[float] = []
    worker_service_ms: list[float] = []
    action_ready_raw_frames: list[int] = []
    worker_service_raw_frames: list[int] = []
    for episode_id, episode_length in enumerate(episode_lengths):
        for _ in range(int(episode_length)):
            sample = sampler.sample({"episode_id": episode_id, "recent_drop_count": 0})
            e2e_latency_ms.append(float(sample.latency_ms))
            worker_service_ms.append(float(sample.worker_service_ms))
            action_ready_raw_frames.append(int(sample.action_ready_raw_frames))
            worker_service_raw_frames.append(int(sample.worker_service_raw_frames))
    return {
        "e2e_latency_ms": e2e_latency_ms,
        "worker_service_ms": worker_service_ms,
        "action_ready_raw_frames": action_ready_raw_frames,
        "worker_service_raw_frames": worker_service_raw_frames,
    }


def temporal_fit_report(
    *,
    real_e2e_latency_ms: Sequence[float],
    real_worker_service_ms: Sequence[float],
    iid_e2e_latency_ms: Sequence[float],
    temporal_e2e_latency_ms: Sequence[float],
    temporal_worker_service_ms: Sequence[float],
    fps: float,
) -> dict[str, Any]:
    threshold = percentile_linear(real_e2e_latency_ms, 90.0)
    return {
        "marginal": {
            "iid": _marginal_comparison(real_e2e_latency_ms, iid_e2e_latency_ms, fps),
            "temporal": _marginal_comparison(real_e2e_latency_ms, temporal_e2e_latency_ms, fps),
        },
        "temporal": {
            "real": _temporal_diagnostics(real_e2e_latency_ms, threshold),
            "iid": _temporal_comparison(real_e2e_latency_ms, iid_e2e_latency_ms, threshold),
            "temporal": _temporal_comparison(real_e2e_latency_ms, temporal_e2e_latency_ms, threshold),
        },
        "frame_level": {
            "iid": _frame_level_comparison(real_e2e_latency_ms, iid_e2e_latency_ms, fps),
            "temporal": _frame_level_comparison(real_e2e_latency_ms, temporal_e2e_latency_ms, fps),
        },
        "joint": {
            "real": _joint_diagnostics(real_e2e_latency_ms, real_worker_service_ms),
            "temporal": _joint_diagnostics(temporal_e2e_latency_ms, temporal_worker_service_ms),
        },
    }


def write_temporal_fit_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def _marginal_comparison(real: Sequence[float], sampled: Sequence[float], fps: float) -> dict[str, Any]:
    metrics = latency_distribution_match_metrics(real, sampled, fps)
    metrics["ks_statistic"] = _ks_statistic(real, sampled)
    metrics["real"] = _series_summary(real)
    metrics["sampled"] = _series_summary(sampled)
    return metrics


def _temporal_comparison(real: Sequence[float], sampled: Sequence[float], threshold: float) -> dict[str, Any]:
    real_diag = _temporal_diagnostics(real, threshold)
    sampled_diag = _temporal_diagnostics(sampled, threshold)
    result: dict[str, Any] = dict(sampled_diag)
    for lag in ACF_LAGS:
        key = f"acf_lag_{lag}"
        result[f"acf_abs_error_lag_{lag}"] = abs(sampled_diag[key] - real_diag[key])
    result["high_tail_run_w1"] = wasserstein_1d(
        real_diag["high_tail_run_lengths"],
        sampled_diag["high_tail_run_lengths"],
    )
    return result


def _temporal_diagnostics(values: Sequence[float], threshold: float) -> dict[str, Any]:
    runs = _true_run_lengths([float(value) >= threshold for value in values])
    result: dict[str, Any] = {
        "high_tail_threshold_ms": float(threshold),
        "high_tail_run_lengths": runs,
        "high_tail_run_count": len(runs),
        "high_tail_run_p50": percentile_linear(runs, 50.0),
        "high_tail_run_p90": percentile_linear(runs, 90.0),
        "high_tail_run_p99": percentile_linear(runs, 99.0),
        "high_tail_run_max": max(runs),
    }
    for lag in ACF_LAGS:
        result[f"acf_lag_{lag}"] = _acf(values, lag)
    return result


def _frame_level_comparison(real: Sequence[float], sampled: Sequence[float], fps: float) -> dict[str, Any]:
    real_frames = [ms_to_raw_frames(value, fps) for value in real]
    sampled_frames = [ms_to_raw_frames(value, fps) for value in sampled]
    return {
        "action_ready_frames": {
            "real": _series_summary(real_frames),
            "sampled": _series_summary(sampled_frames),
            "w1_frames": wasserstein_1d(real_frames, sampled_frames),
        }
    }


def _joint_diagnostics(e2e_latency_ms: Sequence[float], worker_service_ms: Sequence[float]) -> dict[str, float]:
    return {
        "pearson_corr": _pearson(e2e_latency_ms, worker_service_ms),
        "spearman_corr": _pearson(_ranks(e2e_latency_ms), _ranks(worker_service_ms)),
    }


def _series_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": std(values),
        "p50": percentile_linear(values, 50.0),
        "p90": percentile_linear(values, 90.0),
        "p95": percentile_linear(values, 95.0),
        "p99": percentile_linear(values, 99.0),
        "p99_9": percentile_linear(values, 99.9),
        "max": float(max(values)),
    }


def _acf(values: Sequence[float], lag: int) -> float:
    series = [float(value) for value in values]
    if len(series) <= lag:
        return 0.0
    return _pearson(series[:-lag], series[lag:])


def _true_run_lengths(flags: Sequence[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    if not runs:
        return [0]
    return runs


def _ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    left = sorted(float(value) for value in a)
    right = sorted(float(value) for value in b)
    points = sorted(set(left) | set(right))
    index_left = 0
    index_right = 0
    statistic = 0.0
    for point in points:
        while index_left < len(left) and left[index_left] <= point:
            index_left += 1
        while index_right < len(right) and right[index_right] <= point:
            index_right += 1
        statistic = max(statistic, abs(index_left / len(left) - index_right / len(right)))
    return statistic


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][1]] = rank
        cursor = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [float(value) - mean_x for value in xs]
    centered_y = [float(value) - mean_y for value in ys]
    var_x = sum(value * value for value in centered_x)
    var_y = sum(value * value for value in centered_y)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(var_x * var_y)
