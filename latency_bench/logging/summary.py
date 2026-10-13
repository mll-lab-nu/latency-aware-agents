from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latency_bench.core.types import EpisodeMetrics
from latency_bench.utils.io import to_jsonable
from latency_bench.utils.stats import mean as _mean, std as _std

LatencyLabel = int | str


_LATENCY_METADATA_KEYS = (
    "train_latency_steps",
    "eval_latency_steps",
    "configured_latency_ms",
    "checkpoint_kind",
    "suite_name",
    "eval_phase",
    "latency_seed",
    "train_env_steps",
    "checkpoint_path",
    "source_profile_path",
    "checkpoint_global_step",
    "latency_type",
)


def build_queue_eval_summary(
    metrics: Sequence[EpisodeMetrics],
    *,
    experiment_name: str,
    seed: int,
    latency: LatencyLabel,
    metadata: Mapping[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    returns = [float(metric.episode_return) for metric in metrics]
    lengths = [int(metric.survival_steps) for metric in metrics]
    payload: dict[str, Any] = {
        "experiment_name": str(experiment_name),
        "seed": int(seed),
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "latency": _latency_payload_value(latency),
        "suite_name": f"fixed_{latency}" if isinstance(latency, int) else latency,
        "mean_return": _mean(returns),
        "std_return": _std(returns),
        "mean_length": _mean([float(length) for length in lengths]),
        "returns": returns,
        "lengths": lengths,
    }
    task_metrics = _task_metric_summary(metrics)
    task_metrics.update(_task_metric_moment_summary(metrics))
    if task_metrics:
        payload["task_metrics"] = task_metrics
    payload.update(_latency_metadata(metrics, metadata))
    return to_jsonable(payload)


def _task_metric_summary(
    metrics: Sequence[EpisodeMetrics],
) -> dict[str, dict[str, Any]]:
    present = [
        metric.metadata["task_metrics"]
        for metric in metrics
        if "task_metrics" in metric.metadata
    ]
    if not present:
        return {}
    if len(present) != len(metrics):
        raise KeyError("task_metrics")
    summary = {}
    for name in present[0]:
        values = [task_metrics[name] for task_metrics in present]
        summary[name] = {
            "mean": _mean(values),
            "std": _std(values),
            "values": values,
        }
    return summary


def _task_metric_moment_summary(
    metrics: Sequence[EpisodeMetrics],
) -> dict[str, dict[str, Any]]:
    if not any("task_metric_moments" in metric.metadata for metric in metrics):
        return {}
    present = [
        metric.metadata["task_metric_moments"]
        for metric in metrics
    ]
    summary = {}
    for name in present[0]:
        count = sum(item[name]["count"] for item in present)
        total = sum(item[name]["sum"] for item in present)
        total_sq = sum(item[name]["sum_sq"] for item in present)
        mean = total / count
        summary[name] = {
            "mean": mean,
            "std": math.sqrt(max(0.0, total_sq / count - mean * mean)),
            "count": count,
        }
    return summary


def write_queue_eval_summary(
    output_dir: str | Path,
    metrics: Sequence[EpisodeMetrics],
    *,
    experiment_name: str,
    seed: int,
    latency: LatencyLabel,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = build_queue_eval_summary(
        metrics,
        experiment_name=experiment_name,
        seed=seed,
        latency=latency,
        metadata=metadata,
    )

    latency_fragment = _latency_filename_fragment(payload["latency"])
    latency_file = output_path / f"queue_eval_latency_{latency_fragment}.json"
    with latency_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    jsonl_path = output_path / "queue_eval_results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return payload


def _latency_payload_value(latency: LatencyLabel) -> int | str:
    if isinstance(latency, int) and not isinstance(latency, bool):
        return int(latency)
    return str(latency).strip()


def _latency_filename_fragment(latency: int | str) -> str:
    if isinstance(latency, int) and not isinstance(latency, bool):
        return str(int(latency))
    fragment = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(latency)
    )
    return fragment.strip("._-")


def _latency_metadata(
    metrics: Sequence[EpisodeMetrics],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for metric in metrics:
        for key in _LATENCY_METADATA_KEYS:
            value = metric.metadata.get(key)
            if value is not None and key not in selected:
                selected[key] = value
    if metadata is not None:
        for key in _LATENCY_METADATA_KEYS:
            value = metadata.get(key)
            if value is not None:
                selected[key] = value
    return selected
