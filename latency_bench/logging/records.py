from __future__ import annotations

from typing import Any

from latency_bench.core.types import Action, ActionEvent, StepRecord
from latency_bench.logging.metrics import active_action_step_info
from latency_bench.utils.io import to_jsonable
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


__all__ = ["to_jsonable", "STEP_INFO_LOG_OMITTED_KEYS", "step_record_info", "build_step_record"]

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
