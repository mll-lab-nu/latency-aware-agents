from __future__ import annotations

from typing import Any

import numpy as np

from latency_bench.core.types import Observation
from latency_bench.envs.base import EnvAdapter
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer


class SampleFactoryFrameStackObservationBuilder(ObservationBuilder):
    def __init__(
        self,
        *,
        frame_stack: int,
        base_channels: int,
        expected_layout: str = "chw",
        obs_key: str = "obs",
    ):
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if expected_layout.lower() != "chw":
            raise ValueError("SampleFactoryFrameStackObservationBuilder only outputs CHW observations")

        self.frame_stack = int(frame_stack)
        self.base_channels = int(base_channels)
        self.expected_layout = expected_layout.lower()
        self.obs_key = obs_key

    @property
    def expected_channels(self) -> int:
        return self.base_channels * self.frame_stack

    def build(self, env: EnvAdapter, history: HistoryBuffer) -> Observation:
        latest = history.last()
        if latest is None:
            latest = super().build(env, history)
            frames = [self._single_frame_to_chw(latest.data)]
        else:
            if self._is_stacked_chw(latest.data):
                data = np.asarray(latest.data)
                return Observation(
                    data=data,
                    env_step=latest.env_step,
                    sim_time_ms=latest.sim_time_ms,
                    metadata=latest.metadata,
                )
            frames = [self._single_frame_to_chw(observation.data) for observation in history.last_k(self.frame_stack)]

        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames

        data = np.concatenate(frames[-self.frame_stack :], axis=0)
        return Observation(
            data=data,
            env_step=latest.env_step,
            sim_time_ms=latest.sim_time_ms,
            metadata=latest.metadata,
        )

    def _is_stacked_chw(self, data: Any) -> bool:
        arr = np.asarray(data)
        return arr.ndim == 3 and arr.shape[0] == self.expected_channels

    def _single_frame_to_chw(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise self._shape_error(arr)
        if arr.shape[0] == self.base_channels:
            return arr
        if arr.shape[-1] == self.base_channels:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[0] == self.expected_channels:
            return arr
        raise self._shape_error(arr)

    def _shape_error(self, arr: np.ndarray) -> ValueError:
        return ValueError(
            "SampleFactoryFrameStackObservationBuilder expected a 3D image observation "
            f"with shape CHW where C == base_channels ({self.base_channels}), "
            f"HWC where C == base_channels ({self.base_channels}), or already-stacked "
            f"CHW where C == base_channels * frame_stack ({self.expected_channels}); "
            f"actual shape={arr.shape}, base_channels={self.base_channels}, frame_stack={self.frame_stack}"
        )
