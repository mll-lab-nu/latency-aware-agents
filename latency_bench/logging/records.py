from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


STEP_INFO_LOG_OMITTED_KEYS = frozenset({ENV_RAW_RGB_FRAME_STACK_INFO_KEY})


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def step_record_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in STEP_INFO_LOG_OMITTED_KEYS
    }
