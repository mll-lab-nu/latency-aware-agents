from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

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
