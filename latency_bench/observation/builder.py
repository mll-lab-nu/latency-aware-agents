from __future__ import annotations

from typing import Any

import numpy as np

from latency_bench.core.types import Observation
from latency_bench.envs.base import EnvAdapter
from latency_bench.observation.history import HistoryBuffer


class ObservationBuilder:
    def __init__(self, *, frame_stack: int = 1):
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.frame_stack = int(frame_stack)

    def build(self, env: EnvAdapter, history: HistoryBuffer) -> Observation:
        latest = history.last()
        if latest is None:
            latest = self._build_from_env(env)

        if self.frame_stack == 1:
            return latest

        frames = [self._single_frame_data(observation.data) for observation in history.last_k(self.frame_stack)]
        if not frames:
            frames = [self._single_frame_data(latest.data)]
        if any(frame is None for frame in frames):
            return latest

        arrays = [frame for frame in frames if frame is not None]
        if len(arrays) < self.frame_stack:
            arrays = [arrays[0]] * (self.frame_stack - len(arrays)) + arrays
        try:
            data = self._stack_frames(arrays[-self.frame_stack :])
        except ValueError:
            return latest

        return Observation(
            data=data,
            env_step=latest.env_step,
            sim_time_ms=latest.sim_time_ms,
            metadata=latest.metadata,
        )

    def _build_from_env(self, env: EnvAdapter) -> Observation:
        info = env.get_state_info()
        frame = env.render_frame()
        data = frame if frame is not None else info
        return Observation(
            data=data,
            env_step=int(info.get("env_step", 0)),
            sim_time_ms=float(info.get("sim_time_ms", 0.0)),
            metadata=info,
        )

    def _single_frame_data(self, data: Any) -> np.ndarray | None:
        arr = np.asarray(data)
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim != 3:
            return None
        if _looks_already_stacked(arr):
            return None
        return arr

    def _stack_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames):
            raise ValueError("frame stack requires matching frame shapes")
        axis = 0 if _looks_chw(frames[0]) else -1
        return np.concatenate(frames, axis=axis)


def _looks_chw(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}


def _looks_already_stacked(arr: np.ndarray) -> bool:
    if arr.ndim != 3:
        return False
    if arr.shape[0] > 3 and arr.shape[0] % 3 == 0 and arr.shape[-1] not in {1, 3}:
        return True
    if arr.shape[-1] > 3 and arr.shape[-1] % 3 == 0 and arr.shape[0] not in {1, 3}:
        return True
    return False
