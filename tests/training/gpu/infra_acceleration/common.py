from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def obs_digest(value: Any) -> str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def jsonable_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_value(item) for item in value]
    return value


def step_record(
    *,
    obs: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Mapping[str, Any] | None = None,
    info_keys: Sequence[str] = (),
) -> dict[str, Any]:
    selected_info = {key: jsonable_value(info[key]) for key in info_keys} if info_keys else {}
    return {
        "obs_digest": obs_digest(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": selected_info,
    }


def compare_step_traces(
    baseline: Sequence[Mapping[str, Any]],
    accelerated: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    obs_equal = [row["obs_digest"] for row in baseline] == [row["obs_digest"] for row in accelerated]
    reward_equal = [row["reward"] for row in baseline] == [row["reward"] for row in accelerated]
    terminated_equal = [row["terminated"] for row in baseline] == [row["terminated"] for row in accelerated]
    truncated_equal = [row["truncated"] for row in baseline] == [row["truncated"] for row in accelerated]
    info_equal = [row["info"] for row in baseline] == [row["info"] for row in accelerated]
    behavior_changed = not all((obs_equal, reward_equal, terminated_equal, truncated_equal, info_equal))
    return {
        "obs_equal": obs_equal,
        "reward_equal": reward_equal,
        "terminated_equal": terminated_equal,
        "truncated_equal": truncated_equal,
        "info_equal": info_equal,
        "behavior_changed": behavior_changed,
    }
