from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def save_profile_manifest(profile: Mapping[str, Any], path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(dict(profile), f, indent=2, sort_keys=True)
        f.write("\n")


def load_profile_manifest(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_latency_samples(samples: Iterable[float], path: str | Path) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps({"latency_ms": sample}, sort_keys=True) + "\n")


def load_latency_samples(path: str | Path) -> list[float]:
    trace_path = Path(path)
    samples: list[float] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Latency sample record must be a JSON object at {trace_path}:{line_number}")
            if "latency_ms" not in record:
                raise ValueError(
                    f"Missing latency_ms at {trace_path}:{line_number}"
                )
            samples.append(record["latency_ms"])
    if not samples:
        raise ValueError(f"Latency sample trace is empty: {trace_path}")
    return samples


def build_inverse_cdf_distribution(samples_ms: Iterable[float]) -> dict[str, Any]:
    samples = sorted(samples_ms)
    if not samples:
        raise ValueError("latency_distribution samples must not be empty")
    if len(samples) == 1:
        quantile_levels = [0.0, 1.0]
        latency_ms = [samples[0], samples[0]]
    else:
        denominator = len(samples) - 1
        quantile_levels = [index / denominator for index in range(len(samples))]
        latency_ms = samples
    return {
        "schema_version": 1,
        "distribution_type": "inverse_cdf",
        "n_samples": len(samples),
        "quantile_levels": quantile_levels,
        "latency_ms": latency_ms,
    }


def save_latency_distribution(distribution: Mapping[str, Any], path: str | Path) -> None:
    distribution_path = Path(path)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    with distribution_path.open("w", encoding="utf-8") as f:
        json.dump(dict(distribution), f, indent=2, sort_keys=True)
        f.write("\n")


def load_latency_distribution(path: str | Path) -> dict[str, Any]:
    distribution_path = Path(path)
    with distribution_path.open("r", encoding="utf-8") as f:
        return json.load(f)
