from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from latency_bench.core.types import LatencyRecord


def load_latency_trace(path: str | Path) -> list[float]:
    trace_path = Path(path)
    values: list[float] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "latency_ms" not in record:
                raise ValueError(f"Missing latency_ms at {trace_path}:{line_number}")
            latency_ms = float(record["latency_ms"])
            if latency_ms < 0:
                raise ValueError(f"Negative latency_ms at {trace_path}:{line_number}")
            values.append(latency_ms)
    if not values:
        raise ValueError(f"Latency trace is empty: {trace_path}")
    return values


def save_latency_trace(records: list[LatencyRecord], path: str | Path) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

