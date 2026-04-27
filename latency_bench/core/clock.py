from __future__ import annotations

import math


class EnvClock:
    def __init__(self, fps_env: float):
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")
        self.fps_env = float(fps_env)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def step_to_time_ms(self, env_step: int) -> float:
        if env_step < 0:
            raise ValueError("env_step must be non-negative")
        return float(env_step) * self.frame_ms

    def time_ms_to_step_float(self, time_ms: float) -> float:
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative")
        return float(time_ms) / self.frame_ms

    def latency_ms_to_frames_float(self, latency_ms: float) -> float:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        return float(latency_ms) / self.frame_ms


class FrameAccumulator:
    def __init__(self):
        self.residual_frames = 0.0

    def consume(self, frames_float: float) -> int:
        """
        Add fractional frames and return the number of whole frames to advance.
        Keep the remaining fractional frames internally.
        """
        if frames_float < 0:
            raise ValueError("frames_float must be non-negative")
        total = self.residual_frames + float(frames_float)
        whole = int(math.floor(total))
        self.residual_frames = total - whole
        return whole

