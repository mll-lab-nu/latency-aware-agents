from __future__ import annotations

import math
from dataclasses import dataclass, fields
from statistics import mean
from typing import Any

from latency_bench.core.types import ActionEvent


REALTIME_PROFILE_METADATA_KEY = "realtime_profile"

# Stages that together make up the benchmark overhead added on top of raw
# inference latency. Read positions rely on every one of these being measured.
BENCH_OVERHEAD_STAGE_KEYS = (
    "env_observe_ms",
    "env_to_parent_ipc_ms",
    "parent_observation_handle_ms",
    "parent_to_worker_ipc_ms",
    "worker_recv_overhead_ms",
    "worker_to_parent_ipc_ms",
    "parent_release_ms",
    "parent_to_env_ipc_ms",
    "env_action_recv_to_enqueue_ms",
    "env_scheduler_update_ms",
)

# Everything we surface per action: the overhead stages plus context stages
# that are reported but not summed into bench overhead.
REPORTED_STAGE_KEYS = (
    *BENCH_OVERHEAD_STAGE_KEYS,
    "env_step_ms",
    "env_wall_lag_ms",
)


@dataclass
class PipelineProfile:
    """Per-action realtime pipeline timing, all values in milliseconds.

    One profile is created per observation and accumulates timings as that
    observation flows env -> worker -> parent -> env. ``None`` means "not yet
    measured": reading a stage that was never filled is a fail-fast error
    (see :meth:`require`), never a silent zero. Defaults are ``None`` so a
    missing measurement surfaces as a bug instead of masquerading as 0ms.
    """
    # --- reported durations (REPORTED_STAGE_KEYS) ---
    env_observe_ms: float | None = None
    env_to_parent_ipc_ms: float | None = None
    parent_observation_handle_ms: float | None = None
    parent_to_worker_ipc_ms: float | None = None
    worker_recv_overhead_ms: float | None = None
    worker_to_parent_ipc_ms: float | None = None
    parent_release_ms: float | None = None
    parent_to_env_ipc_ms: float | None = None
    env_action_recv_to_enqueue_ms: float | None = None
    env_scheduler_update_ms: float | None = None
    env_step_ms: float | None = None
    env_wall_lag_ms: float | None = None
    # --- scratch timestamps used only to derive the durations above ---
    parent_to_worker_send_start_ms: float | None = None
    parent_to_worker_send_call_ms: float | None = None
    worker_recv_ms: float | None = None
    worker_result_send_start_ms: float | None = None
    parent_observation_recv_ms: float | None = None
    env_observation_send_start_ms: float | None = None
    parent_observation_handle_start_ms: float | None = None
    parent_to_env_send_start_ms: float | None = None
    parent_to_env_send_call_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineProfile:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown PipelineProfile fields: {unknown}")
        return cls(**{key: (None if value is None else float(value)) for key, value in data.items()})

    @classmethod
    def coerce(cls, value: PipelineProfile | dict[str, Any]) -> PipelineProfile:
        return value if isinstance(value, PipelineProfile) else cls.from_dict(value)

    def to_dict(self) -> dict[str, float]:
        # Only measured fields, so merging across IPC round-trips via dict
        # update never clobbers an already-set stage with None.
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    def merge_measured(self, other: PipelineProfile) -> None:
        for f in fields(other):
            value = getattr(other, f.name)
            if value is not None:
                setattr(self, f.name, value)

    def merge_dict(self, data: dict[str, Any]) -> None:
        self.merge_measured(PipelineProfile.from_dict(data))

    def require(self, keys: tuple[str, ...]) -> dict[str, float]:
        missing = [key for key in keys if getattr(self, key) is None]
        if missing:
            raise ValueError(f"PipelineProfile missing measured stages: {missing}")
        return {key: float(getattr(self, key)) for key in keys}

    def bench_overhead_ms(self) -> float:
        return sum(self.require(BENCH_OVERHEAD_STAGE_KEYS).values())


def realtime_pipeline_profile_rows(action_records: list[ActionEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in action_records:
        if not event.applied or event.dropped:
            continue
        raw_profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
        if raw_profile is None:
            continue
        stages = PipelineProfile.coerce(raw_profile).require(REPORTED_STAGE_KEYS)
        row: dict[str, Any] = {
            "action_id": event.action_id,
            "obs_id": event.latency.obs_id,
            "obs_env_step": event.obs_env_step,
            "ready_time_ms": event.ready_time_ms,
            "ready_step_float": event.ready_step_float,
            "apply_raw_frame": event.apply_raw_frame,
            "device": event.latency.device,
            "inference_latency_ms": event.latency.latency_ms,
        }
        row.update(stages)
        row["bench_overhead_ms"] = sum(stages[key] for key in BENCH_OVERHEAD_STAGE_KEYS)
        rows.append(row)
    return rows


def realtime_pipeline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_profiled_actions": 0, "stages": {}}
    return {
        "num_profiled_actions": len(rows),
        "inference_latency_ms": _series_stats([float(row["inference_latency_ms"]) for row in rows]),
        "bench_overhead_ms": _series_stats([float(row["bench_overhead_ms"]) for row in rows]),
        "stages": {
            key: _series_stats([float(row[key]) for row in rows])
            for key in REPORTED_STAGE_KEYS
        },
    }


def _series_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "p50": _percentile(values, 50.0),
        "p90": _percentile(values, 90.0),
        "max": float(max(values)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return float(ordered[index])
