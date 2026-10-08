"""Plan temporal image inputs and convert raw frames for StarVLA policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from latency_bench.core.clock import EnvClock
from latency_bench.integrations import starvla_runtime


def _observation_stride_raw_frames(config: Mapping[str, Any]) -> int:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    ).obs_stride_raw_frames


@dataclass(frozen=True)
class ModelInputPlan:
    """The temporal frame selection shared by environment setup and inference."""

    packed: bool
    frame_count: int
    stride: int
    image_mode: str
    stitch_grid: tuple[int, int]
    kv_memory: bool

    @property
    def raw_span(self) -> int:
        return 1 + (self.frame_count - 1) * self.stride


def resolve_model_input_plan(model_cfg: Mapping[str, Any], stride: int) -> ModelInputPlan:
    data = model_cfg["datasets"]["vla_data"]
    # Saved model configs default to one image; packed and temporal inputs opt in.
    packed = data["pack_image_sequence"] if "pack_image_sequence" in data else False
    image_mode = data["image_mode"] if "image_mode" in data else "single"
    # Input-only packed configs may omit the model framework entirely.
    memory = (model_cfg["framework"]["kv_memory"]
              if "framework" in model_cfg and "kv_memory" in model_cfg["framework"] else {})
    kv_memory = memory["enabled"] if "enabled" in memory else False
    count = (data["image_sequence_length"] if packed else
             data["num_obs_frames"] if image_mode != "single" and not kv_memory else 1)
    return ModelInputPlan(packed, count, stride, image_mode,
                          tuple(data["stitch_grid"]) if "stitch_grid" in data else (2, 2),
                          kv_memory)


def apply_starvla_model_input_config(config, *, model_cfg, image_transform="raw_rgb") -> None:
    """Allocate the raw frame window selected by the saved model input plan."""
    plan = resolve_model_input_plan(model_cfg, _observation_stride_raw_frames(config))
    raw_transform = image_transform.strip().lower() in {"", "none", "raw", "raw_rgb"}
    if plan.packed and not raw_transform:
        raise ValueError("WanOFT packed image sequences require image_transform=raw_rgb")
    if raw_transform:
        config["env"]["frame_stack"] = max(config["env"]["frame_stack"], plan.raw_span)


def observation_data_to_hwc_uint8_frames(data: Any) -> list[npt.NDArray[np.uint8]]:
    frame = _extract_observation_array(data)
    if frame.ndim == 4 and frame.shape[-1] == 3:
        return [_as_uint8_image(item) for item in frame]
    if frame.ndim == 4 and frame.shape[1] == 3:
        return [_as_uint8_image(np.transpose(item, (1, 2, 0))) for item in frame]
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return [_as_uint8_image(frame)]
    if frame.ndim == 3 and frame.shape[0] == 3:
        return [_as_uint8_image(np.transpose(frame, (1, 2, 0)))]
    if (
        frame.ndim == 3
        and frame.shape[0] % 3 == 0
        and frame.shape[0] < frame.shape[1]
        and frame.shape[0] < frame.shape[2]
    ):
        return [
            _as_uint8_image(np.transpose(frame[start : start + 3, :, :], (1, 2, 0)))
            for start in range(0, frame.shape[0], 3)
        ]
    if frame.ndim == 3 and frame.shape[-1] % 3 == 0:
        return [
            _as_uint8_image(frame[:, :, start : start + 3])
            for start in range(0, frame.shape[-1], 3)
        ]
    if frame.ndim == 3 and frame.shape[0] % 3 == 0:
        return [
            _as_uint8_image(np.transpose(frame[start : start + 3, :, :], (1, 2, 0)))
            for start in range(0, frame.shape[0], 3)
        ]
    return [_as_uint8_image(frame)]


def _extract_observation_array(data: Any) -> npt.NDArray[Any]:
    if isinstance(data, Mapping):
        return np.asarray(data["observation"])
    return np.asarray(data)


def _as_uint8_image(frame: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(frame, dtype=np.uint8)


_STITCH_FRAMES = None


def get_stitch_frames():
    """Lazily import starVLA's stitch_frames (starVLA path is added at runtime)."""
    global _STITCH_FRAMES
    if _STITCH_FRAMES is None:
        starvla_runtime.ensure_starvla_path()
        from starVLA.training.trainer_utils.trainer_tools import stitch_frames

        _STITCH_FRAMES = stitch_frames
    return _STITCH_FRAMES
