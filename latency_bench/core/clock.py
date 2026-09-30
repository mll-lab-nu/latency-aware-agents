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


FRAME_ROUNDING_EPSILON = 1e-10

def frames_float_to_raw_frames(frames: float) -> int:
    return int(math.ceil(frames - FRAME_ROUNDING_EPSILON))


def ms_to_raw_frames(latency_ms: float, env_fps: float) -> int:
    return frames_float_to_raw_frames(latency_ms * env_fps / 1000.0)

class EnvClock:
    def __init__(
        self,
        *,
        env_fps: float,
        obs_fps: float,
    ):
        validate_clock_rates(env_fps, obs_fps)
        self.env_fps = env_fps
        self.obs_fps = obs_fps
        self.obs_stride_raw_frames = round(env_fps / obs_fps)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def step_to_time_ms(self, env_step: int) -> float:
        return env_step * self.frame_ms

    def time_ms_to_step_float(self, time_ms: float) -> float:
        return time_ms / self.frame_ms

    def latency_ms_to_frames_float(self, latency_ms: float) -> float:
        return latency_ms / self.frame_ms

    def latency_ms_to_raw_frames(self, latency_ms: float) -> int:
        return ms_to_raw_frames(latency_ms, self.env_fps)
