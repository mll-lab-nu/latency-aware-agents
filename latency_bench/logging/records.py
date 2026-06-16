from __future__ import annotations

from typing import Any

from latency_bench.utils.io import to_jsonable
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


__all__ = ["to_jsonable", "STEP_INFO_LOG_OMITTED_KEYS", "step_record_info"]

STEP_INFO_LOG_OMITTED_KEYS = frozenset({ENV_RAW_RGB_FRAME_STACK_INFO_KEY})


def step_record_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in STEP_INFO_LOG_OMITTED_KEYS
    }
