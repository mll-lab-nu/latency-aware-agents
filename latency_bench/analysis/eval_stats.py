from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from latency_bench.utils.io import load_jsonl_records


EPISODE_METRICS_CANDIDATES = ("episode_metrics.jsonl",)
LATENCY_RECORD_CANDIDATES = ("latencies.jsonl",)
ACTION_RECORD_CANDIDATES = ("actions.jsonl",)

DEFAULT_LENGTH_FIELD = "survival_steps"
METADATA_KEYS = (
    "mode",
    "policy_id",
    "env_id",
    "model_id",
    "gpu_class",
    "workload_id",
    "instance_id",
    "source_run_id",
    "profile_ref",
    "env_fps",
    "obs_fps",
    "frame_ms",
    "latency_type",
)


def resolve_run_file(run_path: str | Path, candidates: Sequence[str]) -> Path:
    path = Path(run_path)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in candidates:
            candidate_path = path / candidate
            if candidate_path.is_file():
                return candidate_path
        raise FileNotFoundError(
            f"Could not find any candidate files in run directory {path}: {', '.join(candidates)}"
        )
    raise FileNotFoundError(f"Run path does not exist: {path}")


def extract_episode_return(record: dict[str, Any], return_field: str = "episode_return_env") -> float:
    return record[return_field]


def extract_episode_length(record: dict[str, Any], length_field: str | None = None) -> float:
    selected_field = DEFAULT_LENGTH_FIELD if length_field is None else length_field
    return record[selected_field]


def summarize_eval_run(
    episode_metrics_path_or_run_dir: str | Path,
    *,
    return_field: str = "episode_return_env",
    length_field: str | None = None,
) -> dict[str, Any]:
    metrics_path = resolve_run_file(episode_metrics_path_or_run_dir, EPISODE_METRICS_CANDIDATES)
    records = load_jsonl_records(metrics_path)

    selected_return_field = return_field
    selected_length_field = DEFAULT_LENGTH_FIELD if length_field is None else length_field
    returns = [extract_episode_return(record, selected_return_field) for record in records]
    lengths = [extract_episode_length(record, selected_length_field) for record in records]

    return_values = np.asarray(returns, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    summary: dict[str, Any] = {
        "n_episodes": int(len(returns)),
        "mean_return": float(np.mean(return_values)),
        "std_return": float(np.std(return_values, ddof=0)),
        "min_return": float(np.min(return_values)),
        "max_return": float(np.max(return_values)),
        "mean_length": float(np.mean(length_values)),
        "std_length": float(np.std(length_values, ddof=0)),
        "min_length": float(np.min(length_values)),
        "max_length": float(np.max(length_values)),
        "return_field": selected_return_field,
        "length_field": selected_length_field,
    }
    summary.update(_summarize_metadata(records))
    return summary


def summarize_eval_sessions(run_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Summarize independent evaluation sessions using whole-session means."""
    runs = [summarize_eval_run(run_path) for run_path in run_paths]
    return_means = np.asarray([run["mean_return"] for run in runs], dtype=np.float64)
    length_means = np.asarray([run["mean_length"] for run in runs], dtype=np.float64)
    return_std = None
    length_std = None
    return_ci95 = None
    length_ci95 = None
    if len(runs) > 1:
        return_std = float(np.std(return_means, ddof=1))
        length_std = float(np.std(length_means, ddof=1))
        t_critical = float(student_t.ppf(0.975, len(runs) - 1))
        return_ci95 = t_critical * return_std / np.sqrt(len(runs))
        length_ci95 = t_critical * length_std / np.sqrt(len(runs))
    return {
        "session_count": len(runs),
        "n_episodes": sum(int(run["n_episodes"]) for run in runs),
        "mean_return": float(np.mean(return_means)),
        "std_return": return_std,
        "return_ci95": return_ci95,
        "mean_length": float(np.mean(length_means)),
        "std_length": length_std,
        "length_ci95": length_ci95,
        "runs": runs,
    }


def independent_mean_difference_ci95(
    *,
    first_mean: float,
    first_std: float,
    first_count: int,
    second_mean: float,
    second_std: float,
    second_count: int,
) -> tuple[float, float]:
    """Welch interval for the difference between independent session means."""
    difference = float(second_mean - first_mean)
    if first_std == 0.0 and second_std == 0.0:
        return difference, 0.0
    first_term = first_std**2 / first_count
    second_term = second_std**2 / second_count
    standard_error = np.sqrt(first_term + second_term)
    degrees_of_freedom = (first_term + second_term) ** 2 / (
        first_term**2 / (first_count - 1) + second_term**2 / (second_count - 1)
    )
    half_width = student_t.ppf(0.975, degrees_of_freedom) * standard_error
    return difference, float(half_width)


def extract_latency_samples_from_records(records: Iterable[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for record in records:
        if "latency_ms" in record and record["latency_ms"] is not None:
            samples.append(record["latency_ms"])
    return samples


def _summarize_metadata(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {key: _collapse_unique_values(_metadata_values(records, key)) for key in METADATA_KEYS}


def _metadata_values(records: Sequence[dict[str, Any]], key: str) -> list[Any]:
    values = (record["metadata"].get(key) for record in records)
    return [value for value in values if value is not None]


def _collapse_unique_values(values: Sequence[Any]) -> Any:
    unique_by_key = {_stable_value_key(value): value for value in values}
    if not unique_by_key:
        return None
    if len(unique_by_key) == 1:
        return next(iter(unique_by_key.values()))
    return ";".join(str(unique_by_key[key]) for key in sorted(unique_by_key))


def _stable_value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)
