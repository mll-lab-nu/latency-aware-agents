"""IPC message types shared by realtime parent and child processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.types import (
    Action,
    ActionEvent,
    LatencyRecord,
    Observation,
    PolicyOutput,
)
from latency_bench.logging.realtime_pipeline import PipelineProfile, REALTIME_PROFILE_METADATA_KEY


@dataclass
class RealtimeObservationTask:
    """Observation sent from the environment loop to an inference worker."""
    obs_id: int
    observation: Observation
    submit_env_step: int
    submit_time_ms: float
    profile: PipelineProfile | None = None
    shared_obs_id: int | None = None


@dataclass
class RealtimePolicyCompletion:
    """Inference result after the parent reattaches its observation task."""
    task: RealtimeObservationTask
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    worker_id: int
    device: str
    profile: PipelineProfile | None = None


@dataclass
class RealtimePolicyResult:
    """Slim inference-worker result sent back to the parent process."""
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    profile: PipelineProfile | None = None


@dataclass
class RealtimeActionDispatch:
    action_id: int
    action: Action
    obs_env_step: int
    ready_time_ms: float
    ready_step_float: float
    latency_raw_frames: int | None = None
    ready_raw_frame: int | None = None
    apply_raw_frame: int | None = None
    applied: bool = False
    dropped: bool = False
    profile: PipelineProfile | None = None


def _action_event_dispatch_payload(event: ActionEvent) -> dict[str, Any]:
    payload = {
        "action_id": event.action_id,
        "action": event.action,
        "obs_env_step": event.obs_env_step,
        "ready_time_ms": event.ready_time_ms,
        "ready_step_float": event.ready_step_float,
        "latency_raw_frames": event.latency_raw_frames,
        "ready_raw_frame": event.ready_raw_frame,
    }
    profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
    if profile is not None:
        payload["profile"] = profile.to_dict()
    return payload


def _action_event_from_dispatch_payload(payload: dict[str, Any]) -> RealtimeActionDispatch:
    profile = PipelineProfile.from_dict(payload["profile"]) if "profile" in payload else None
    return RealtimeActionDispatch(
        action_id=payload["action_id"],
        action=payload["action"],
        obs_env_step=payload["obs_env_step"],
        ready_time_ms=payload["ready_time_ms"],
        ready_step_float=payload["ready_step_float"],
        latency_raw_frames=payload["latency_raw_frames"],
        ready_raw_frame=payload["ready_raw_frame"],
        profile=profile,
    )


def _action_event_ack_payload(event: RealtimeActionDispatch) -> dict[str, Any]:
    payload = {
        "action_id": event.action_id,
        "applied": event.applied,
        "apply_raw_frame": event.apply_raw_frame,
    }
    if event.profile is not None:
        payload["profile"] = event.profile.to_dict()
    return payload
