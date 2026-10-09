from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

from latency_bench.core.types import Action, Observation, StepResult


ENV_RAW_RGB_FRAME_STACK_INFO_KEY = "env_raw_rgb_frame_stack"


class RawRgbFrameStackBuffer:
    """Maintain an oldest-to-newest stack of raw RGB frames."""

    def __init__(self, *, num_frames: int):
        self.num_frames = num_frames
        self._frames: deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self, frame: Any) -> None:
        raw_frame = np.asarray(frame)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(raw_frame.copy())

    def append(self, frame: Any) -> None:
        self._frames.append(np.asarray(frame).copy())

    def stack(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0)


class EnvAdapter(ABC):
    env_fps: float
    noop_action: Action
    OBSERVATION_TYPE: str

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    def _metadata(self, info: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation": self.OBSERVATION_TYPE,
            }
        )
        if self.export_env_raw_rgb_frames:
            metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frame_stack()
        return metadata

    def _make_observation(self, info: dict[str, Any]) -> Observation:
        metadata = self._metadata(info)
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def observe(self) -> Observation:
        info = self.get_state_info()
        frame = self.render_frame()
        return Observation(
            data=frame,
            env_step=info["env_step"],
            sim_time_ms=info["sim_time_ms"],
            metadata=info,
        )

    def render_frame(self) -> Any:
        return self._last_frame

    def _reset_raw_rgb_frame_stack(self) -> None:
        self._raw_rgb_frames.reset(self.render_game_frame())

    def _append_raw_rgb_frame(self) -> None:
        self._raw_rgb_frames.append(self.render_game_frame())

    def _raw_rgb_frame_stack(self) -> np.ndarray:
        return self._raw_rgb_frames.stack()

    def render_game_frame(self) -> Any:
        """Return the raw RGB game image used for human video visualization.

        This is the game frame before model-facing preprocessing such as resize,
        stacking, grayscale conversion, or layout conversion. Video recording
        uses this method instead of policy observations.
        """
        raise NotImplementedError

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "observation": self.OBSERVATION_TYPE,
            }
        )
        return info

    def close(self) -> None:
        return None

def collection_frame(env, observation):
    """Snapshot student RGB in its owner without adding pixels to teacher input."""
    frame = (observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1]
             if ENV_RAW_RGB_FRAME_STACK_INFO_KEY in observation.metadata else env.render_game_frame())
    return np.array(frame, copy=True)
