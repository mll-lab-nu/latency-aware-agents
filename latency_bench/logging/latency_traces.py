from __future__ import annotations

from latency_bench.core.types import ActionEvent


def e2e_latency_rows(action_records: list[ActionEvent]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for event in action_records:
        if not event.applied or event.dropped:
            continue
        e2e_latency_frames = event.apply_raw_frame - event.obs_env_step
        e2e_latency_ms = e2e_latency_frames * event.metadata["frame_ms"]
        rows.append(
            {
                "action_id": event.action_id,
                "obs_env_step": event.obs_env_step,
                "apply_raw_frame": event.apply_raw_frame,
                "inference_latency_ms": event.latency.latency_ms,
                "e2e_latency_frames": e2e_latency_frames,
                "e2e_latency_ms": e2e_latency_ms,
                "latency_ms": e2e_latency_ms,
            }
        )
    return rows
