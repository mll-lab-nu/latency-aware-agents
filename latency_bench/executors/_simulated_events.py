"""Build simulated execution events and their latency records."""

from __future__ import annotations

from typing import Any

from latency_bench.core.types import ActionEvent, LatencyRecord, PolicyOutput
from latency_bench.latency.sample import LatencySample
from latency_bench.logging.metrics import apply_latency_record_raw_fields


def build_simulated_action_event(
    *,
    action_id: int,
    obs_id: int,
    policy_output: PolicyOutput,
    raw_frame: int,
    ready_raw_frame: int,
    ready_time_ms: float,
    latency_sample: LatencySample,
    frame_ms: float,
    latency_type: str,
    profile_metadata: dict[str, Any],
    latency_record_metadata: dict[str, Any] | None = None,
    extra_event_metadata: dict[str, Any] | None = None,
) -> tuple[LatencyRecord, ActionEvent]:
    if latency_record_metadata is None:
        latency_record = LatencyRecord(
            obs_id=obs_id,
            latency_ms=latency_sample.latency_ms,
        )
    else:
        latency_record = LatencyRecord(
            obs_id=obs_id,
            latency_ms=latency_sample.latency_ms,
            metadata=latency_record_metadata,
        )
    apply_latency_record_raw_fields(
        record=latency_record,
        latency_type=latency_type,
        profile_metadata=profile_metadata,
    )
    event_metadata = dict(policy_output.metadata)
    if extra_event_metadata is not None:
        event_metadata.update(extra_event_metadata)
    event_metadata["frame_ms"] = frame_ms
    event_metadata["worker_service_ms"] = latency_sample.worker_service_ms
    event_metadata["worker_service_raw_frames"] = latency_sample.worker_service_raw_frames
    event = ActionEvent(
        action_id=action_id,
        action=policy_output.action,
        obs_env_step=raw_frame,
        ready_time_ms=ready_time_ms,
        ready_step_float=float(ready_raw_frame),
        latency=latency_record,
        latency_raw_frames=latency_sample.action_ready_raw_frames,
        ready_raw_frame=ready_raw_frame,
        action_chunk=policy_output.action_chunk,
        metadata=event_metadata,
    )
    return latency_record, event
