from __future__ import annotations

import math


def validate_clock_rates(env_fps: float, obs_fps: float) -> None:
    ratio = env_fps / obs_fps
    rounded_ratio = round(ratio)
    if rounded_ratio < 1 or not math.isclose(ratio, rounded_ratio, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "env_fps must be an integer multiple of obs_fps "
            f"for raw-frame observation scheduling: env_fps={env_fps}, obs_fps={obs_fps}"
        )


def ms_to_raw_frames(latency_ms: float, env_fps: float) -> int:
    latency_ms_value = float(latency_ms)
    if not math.isfinite(latency_ms_value):
        raise ValueError("latency_ms must be finite")
    if latency_ms_value < 0:
        raise ValueError("latency_ms must be non-negative")
    frames = latency_ms_value / 1000.0 * env_fps
    rounded_frames = round(frames)
    if math.isclose(frames, rounded_frames, rel_tol=1e-12, abs_tol=1e-12):
        frames = float(rounded_frames)
    return int(math.ceil(frames))


class EnvClock:
    def __init__(
        self,
        *,
        env_fps: float | None = None,
        obs_fps: float | None = None,
    ):
        if env_fps is None:
            raise ValueError("env_fps is required")
        if obs_fps is None:
            raise ValueError("obs_fps is required")
        validate_clock_rates(env_fps, obs_fps)
        self.env_fps = float(env_fps)
        self.obs_fps = float(obs_fps)
        self.obs_stride_raw_frames = round(float(env_fps) / float(obs_fps))

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

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

    def latency_ms_to_raw_frames(self, latency_ms: float) -> int:
        return ms_to_raw_frames(latency_ms, self.env_fps)


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
