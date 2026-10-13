from __future__ import annotations

from typing import Any

from latency_bench.core.types import Action, ActionEvent, StepRecord
from latency_bench.logging.metrics import active_action_step_info
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


STEP_INFO_LOG_OMITTED_KEYS = frozenset({ENV_RAW_RGB_FRAME_STACK_INFO_KEY})


def step_record_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in STEP_INFO_LOG_OMITTED_KEYS
    }


def build_step_record(
    *,
    episode_id: int,
    env_step: int,
    scheduled_time_ms: float,
    active_action: Action,
    reward: float,
    done: bool,
    info: dict[str, Any],
    active_event: ActionEvent | None,
    frame_ms: float,
    latency_type: str,
    pre_active_info: dict[str, Any] | None = None,
) -> StepRecord:
    step_info = step_record_info(info)
    if pre_active_info is not None:
        step_info.update(pre_active_info)
    step_info.update(
        active_action_step_info(
            active_event=active_event,
            scheduled_time_ms=scheduled_time_ms,
            frame_ms=frame_ms,
            latency_type=latency_type,
        )
    )
    return StepRecord(
        episode_id=episode_id,
        env_step=env_step,
        sim_time_ms=scheduled_time_ms,
        active_action=active_action,
        reward=float(reward),
        done=done,
        info=step_info,
    )


def e2e_latency_row(event: ActionEvent) -> dict[str, float | int] | None:
    if not event.applied or event.dropped:
        return None
    applied_latency_frames = event.apply_raw_frame - event.obs_env_step
    applied_latency_ms = applied_latency_frames * event.metadata["frame_ms"]
    row: dict[str, float | int] = {
        "action_id": event.action_id,
        "obs_env_step": event.obs_env_step,
        "apply_raw_frame": event.apply_raw_frame,
        "inference_latency_ms": event.latency.latency_ms,
        "applied_latency_frames": applied_latency_frames,
        "applied_latency_ms": applied_latency_ms,
        "e2e_latency_ms": applied_latency_ms,
        "latency_ms": applied_latency_ms,
    }
    return row


def e2e_latency_rows(action_records: list[ActionEvent]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for event in action_records:
        row = e2e_latency_row(event)
        if row is not None:
            rows.append(row)
    return rows
