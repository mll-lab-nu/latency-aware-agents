from __future__ import annotations

import math


def validate_clock_rates(env_fps: float, obs_fps: float) -> None:
    env_fps_value = _coerce_positive_finite(env_fps, "env_fps")
    obs_fps_value = _coerce_positive_finite(obs_fps, "obs_fps")
    ratio = env_fps_value / obs_fps_value
    rounded_ratio = round(ratio)
    if rounded_ratio < 1 or not math.isclose(ratio, rounded_ratio, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "env_fps must be an integer multiple of obs_fps "
            f"for raw-frame observation scheduling: env_fps={env_fps_value}, obs_fps={obs_fps_value}"
        )


def compute_obs_stride_raw_frames(env_fps: float, obs_fps: float) -> int:
    validate_clock_rates(env_fps, obs_fps)
    return int(round(float(env_fps) / float(obs_fps)))


def ms_to_raw_frames(latency_ms: float, env_fps: float) -> int:
    latency_ms_value = float(latency_ms)
    if not math.isfinite(latency_ms_value):
        raise ValueError("latency_ms must be finite")
    if latency_ms_value < 0:
        raise ValueError("latency_ms must be non-negative")
    env_fps_value = _coerce_positive_finite(env_fps, "env_fps")
    frames = latency_ms_value / 1000.0 * env_fps_value
    rounded_frames = round(frames)
    if math.isclose(frames, rounded_frames, rel_tol=1e-12, abs_tol=1e-12):
        frames = float(rounded_frames)
    return int(math.ceil(frames))


def _coerce_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class EnvClock:
    def __init__(
        self,
        fps_env: float | None = None,
        *,
        env_fps: float | None = None,
        obs_fps: float | None = None,
    ):
        if fps_env is not None and env_fps is not None:
            if not math.isclose(float(fps_env), float(env_fps), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"fps_env conflicts with env_fps: fps_env={fps_env}, env_fps={env_fps}")
        selected_env_fps = env_fps if env_fps is not None else fps_env
        if selected_env_fps is None:
            raise ValueError("env_fps is required")
        selected_obs_fps = selected_env_fps if obs_fps is None else obs_fps
        validate_clock_rates(selected_env_fps, selected_obs_fps)
        self.env_fps = float(selected_env_fps)
        self.fps_env = self.env_fps
        self.obs_fps = float(selected_obs_fps)
        self.obs_stride_raw_frames = compute_obs_stride_raw_frames(self.env_fps, self.obs_fps)

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
