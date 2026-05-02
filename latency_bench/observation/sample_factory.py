from __future__ import annotations

from typing import Any

import numpy as np

from latency_bench.core.types import Observation
from latency_bench.envs.base import EnvAdapter
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from modeling.flappy_ppo.preprocessing import latency_channel_value


class SampleFactoryFrameStackObservationBuilder(ObservationBuilder):
    def __init__(
        self,
        *,
        frame_stack: int,
        base_channels: int,
        expected_layout: str = "chw",
        obs_key: str = "obs",
        obs_mode: str = "image_only",
        max_frame_skip: int = 1,
        constant_frame_skip: int | None = None,
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
        self.obs_mode = str(obs_mode or "image_only").strip().lower()
        if self.obs_mode not in {"image_only", "latency_channel"}:
            raise ValueError("obs_mode must be 'image_only' or 'latency_channel'")
        self.max_frame_skip = max(1, int(max_frame_skip))
        self.constant_frame_skip = None if constant_frame_skip is None else max(0, int(constant_frame_skip))

    @property
    def stacked_rgb_channels(self) -> int:
        return self.base_channels * self.frame_stack

    @property
    def expected_channels(self) -> int:
        return self.stacked_rgb_channels + (1 if self.obs_mode == "latency_channel" else 0)

    def build(self, env: EnvAdapter, history: HistoryBuffer) -> Observation:
        latest = history.last()
        if latest is None:
            latest = super().build(env, history)
            frames = [self._single_frame_to_chw(latest.data)]
        else:
            if self._is_stacked_chw(latest.data):
                data = np.asarray(latest.data)
                if self.obs_mode == "latency_channel" and data.shape[0] == self.stacked_rgb_channels:
                    data = self._append_latency_channel(data, latest.metadata)
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
        if self.obs_mode == "latency_channel":
            data = self._append_latency_channel(data, latest.metadata)
        return Observation(
            data=data,
            env_step=latest.env_step,
            sim_time_ms=latest.sim_time_ms,
            metadata=latest.metadata,
        )

    def _is_stacked_chw(self, data: Any) -> bool:
        arr = np.asarray(data)
        return arr.ndim == 3 and arr.shape[0] in {self.stacked_rgb_channels, self.expected_channels}

    def _single_frame_to_chw(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise self._shape_error(arr)
        if arr.shape[0] == self.base_channels:
            return arr
        if arr.shape[-1] == self.base_channels:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[0] == self.stacked_rgb_channels or arr.shape[0] == self.expected_channels:
            return arr
        raise self._shape_error(arr)

    def _frame_skip_from_metadata(self, metadata: dict[str, Any]) -> int:
        for key in ("frame_skip", "latency_frame_skip", "queue_latency"):
            if key in metadata:
                return max(0, int(metadata[key]))
        if self.constant_frame_skip is not None:
            return self.constant_frame_skip
        raise ValueError(
            "flappy_sf latency_channel observation requires an exact frame_skip/queue_latency; "
            "cannot infer it from this latency_bench config"
        )

    def _append_latency_channel(self, stacked_rgb: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        arr = np.asarray(stacked_rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[0] != self.stacked_rgb_channels:
            raise self._shape_error(arr)
        frame_skip = self._frame_skip_from_metadata(metadata)
        value = latency_channel_value(frame_skip, self.max_frame_skip)
        channel = np.full((1, arr.shape[1], arr.shape[2]), value, dtype=np.uint8)
        return np.concatenate([arr, channel], axis=0)

    def _shape_error(self, arr: np.ndarray) -> ValueError:
        return ValueError(
            "SampleFactoryFrameStackObservationBuilder expected a 3D image observation "
            f"with shape CHW where C == base_channels ({self.base_channels}), "
            f"HWC where C == base_channels ({self.base_channels}), or already-stacked "
            f"CHW where C == RGB stack ({self.stacked_rgb_channels}) or final expected channels "
            f"({self.expected_channels}); "
            f"actual shape={arr.shape}, base_channels={self.base_channels}, frame_stack={self.frame_stack}"
        )
