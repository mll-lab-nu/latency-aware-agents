from __future__ import annotations

from latency_bench.core.types import ActionEvent
from latency_bench.logging.realtime_pipeline import PipelineProfile, REALTIME_PROFILE_METADATA_KEY


def e2e_latency_row(event: ActionEvent) -> dict[str, float | int] | None:
    if not event.applied or event.dropped:
        return None
    applied_latency_frames = event.apply_raw_frame - event.obs_env_step
    applied_latency_ms = applied_latency_frames * event.metadata["frame_ms"]
    raw_profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
    row: dict[str, float | int] = {
        "action_id": event.action_id,
        "obs_env_step": event.obs_env_step,
        "apply_raw_frame": event.apply_raw_frame,
        "inference_latency_ms": event.latency.latency_ms,
        "applied_latency_frames": applied_latency_frames,
        "applied_latency_ms": applied_latency_ms,
    }
    if raw_profile is None:
        # Profiling disabled: fall back to the frame-quantized applied delay.
        row["e2e_latency_ms"] = applied_latency_ms
        row["latency_ms"] = applied_latency_ms
    else:
        # Profiling enabled: a present profile must be complete (fail-fast).
        bench_overhead_ms = PipelineProfile.coerce(raw_profile).bench_overhead_ms()
        e2e_latency_ms = float(event.latency.latency_ms) + bench_overhead_ms
        row["bench_overhead_ms"] = bench_overhead_ms
        row["e2e_latency_ms"] = e2e_latency_ms
        row["latency_ms"] = e2e_latency_ms
    return row


def e2e_latency_rows(action_records: list[ActionEvent]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for event in action_records:
        row = e2e_latency_row(event)
        if row is not None:
            rows.append(row)
    return rows
