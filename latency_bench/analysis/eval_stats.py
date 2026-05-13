from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np


EPISODE_METRICS_CANDIDATES = ("episode_metrics.jsonl", "metrics.jsonl", "episodes.jsonl")
LATENCY_RECORD_CANDIDATES = ("latencies.jsonl", "latency_records.jsonl")
ACTION_RECORD_CANDIDATES = ("actions.jsonl", "action_records.jsonl")

RETURN_FIELD_PRIORITY = ("episode_return_env", "episode_return", "return_raw")
LENGTH_FIELD_PRIORITY = ("episode_length", "survival_steps", "length", "steps", "num_steps")
METADATA_KEYS = (
    "mode",
    "policy_id",
    "env_id",
    "profile_id",
    "profile_type",
    "target_device",
    "replay_host",
    "fps",
    "frame_ms",
    "latency_type",
)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be a JSON object at {jsonl_path}:{line_number}")
            records.append(record)
    return records


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


def extract_episode_return(record: dict[str, Any], return_field: str = "episode_return_env") -> float | None:
    if return_field != "episode_return_env":
        return _optional_float(record.get(return_field), return_field)

    for field in RETURN_FIELD_PRIORITY:
        value = _optional_float(record.get(field), field)
        if value is not None:
            return value
    return None


def extract_episode_length(record: dict[str, Any], length_field: str | None = None) -> int | float | None:
    if length_field is not None:
        return _optional_float(record.get(length_field), length_field)

    for field in LENGTH_FIELD_PRIORITY:
        value = _optional_float(record.get(field), field)
        if value is not None:
            return value
    return None


def summarize_eval_run(
    episode_metrics_path_or_run_dir: str | Path,
    *,
    return_field: str = "episode_return_env",
    length_field: str | None = None,
) -> dict[str, Any]:
    metrics_path = resolve_run_file(episode_metrics_path_or_run_dir, EPISODE_METRICS_CANDIDATES)
    records = load_jsonl_records(metrics_path)
    if not records:
        raise ValueError(f"Episode metrics file is empty: {metrics_path}")

    selected_return_field = _select_return_field(records, return_field)
    selected_length_field = _select_length_field(records, length_field)
    returns = [
        value
        for record in records
        if (value := extract_episode_return(record, selected_return_field)) is not None
    ]
    lengths = [
        value
        for record in records
        if (value := extract_episode_length(record, selected_length_field)) is not None
    ]
    if not returns:
        raise ValueError(f"No valid episode returns found in {metrics_path} for field {selected_return_field!r}")
    if not lengths:
        raise ValueError(f"No valid episode lengths found in {metrics_path} for field {selected_length_field!r}")

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


def extract_latency_samples_from_records(records: Iterable[dict[str, Any]]) -> list[float]:
    samples: list[float] = []
    for record in records:
        if "latency_ms" in record and record["latency_ms"] is not None:
            samples.append(_finite_float(record["latency_ms"], "latency_ms"))
    if not samples:
        raise ValueError("No valid latency samples found")
    return samples


def extract_action_age_frames_from_records(records: Iterable[dict[str, Any]], *, strict: bool = False) -> list[int]:
    values: list[int] = []
    for record in records:
        value = record.get("action_age_frames")
        if value is None:
            continue
        numeric = _finite_float(value, "action_age_frames")
        if not numeric.is_integer():
            raise ValueError("action_age_frames must contain integer-like values")
        values.append(int(numeric))
    if strict and not values:
        raise ValueError("No valid action_age_frames found")
    return values


def _select_return_field(records: Sequence[dict[str, Any]], return_field: str) -> str:
    if return_field != "episode_return_env":
        return return_field
    for field in RETURN_FIELD_PRIORITY:
        if any(_optional_float(record.get(field), field) is not None for record in records):
            return field
    return return_field


def _select_length_field(records: Sequence[dict[str, Any]], length_field: str | None) -> str | None:
    if length_field is not None:
        return length_field
    for field in LENGTH_FIELD_PRIORITY:
        if any(_optional_float(record.get(field), field) is not None for record in records):
            return field
    return None


def _summarize_metadata(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {key: _collapse_unique_values(_metadata_values(records, key)) for key in METADATA_KEYS}


def _metadata_values(records: Sequence[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for record in records:
        value = record.get(key)
        if value is None and isinstance(record.get("metadata"), dict):
            value = record["metadata"].get(key)
        if value is not None:
            values.append(value)
    return values


def _collapse_unique_values(values: Sequence[Any]) -> Any:
    unique_by_key = {_stable_value_key(value): value for value in values}
    if not unique_by_key:
        return None
    if len(unique_by_key) == 1:
        return next(iter(unique_by_key.values()))
    return ";".join(str(unique_by_key[key]) for key in sorted(unique_by_key))


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _finite_float(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must contain finite values")
    return numeric


def _stable_value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)
