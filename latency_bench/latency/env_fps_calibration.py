from __future__ import annotations

from dataclasses import asdict, dataclass

from latency_bench.core.clock import ms_to_raw_frames


@dataclass(frozen=True)
class EnvFpsCalibration:
    selected_env_fps: int
    selected_obs_fps: float
    observation_stride: float
    target_latency_frames: int
    ideal_env_fps: float
    max_env_fps: int
    device_count: int
    sample_count: int
    mean_latency_ms: float
    worker_utilization_cap: float
    hit_count: int
    mean_abs_frame_error: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def select_env_fps_from_warmup(
    *,
    latency_samples_ms: list[float],
    target_latency_frames: int,
    base_env_fps: float,
    base_obs_fps: float,
    device_count: int,
    worker_utilization_cap: float,
) -> EnvFpsCalibration:
    sample_count = len(latency_samples_ms)
    mean_latency_ms = sum(latency_samples_ms) / sample_count
    observation_stride = base_env_fps / base_obs_fps
    ideal_env_fps = target_latency_frames * 1000.0 / mean_latency_ms
    max_env_fps = int(device_count * observation_stride * 1000.0 * worker_utilization_cap / mean_latency_ms)

    best: tuple[tuple[float, float, float, int], EnvFpsCalibration] | None = None
    for env_fps in range(1, max_env_fps + 1):
        frame_values = [value * env_fps / 1000.0 for value in latency_samples_ms]
        raw_frames = [ms_to_raw_frames(value, env_fps) for value in latency_samples_ms]
        hit_count = sum(1 for value in raw_frames if value == target_latency_frames)
        mean_abs_frame_error = sum(abs(value - target_latency_frames) for value in frame_values) / sample_count
        score = (
            -hit_count,
            mean_abs_frame_error,
            abs(env_fps - ideal_env_fps),
            env_fps,
        )
        calibration = EnvFpsCalibration(
            selected_env_fps=env_fps,
            selected_obs_fps=env_fps / observation_stride,
            observation_stride=observation_stride,
            target_latency_frames=target_latency_frames,
            ideal_env_fps=ideal_env_fps,
            max_env_fps=max_env_fps,
            device_count=device_count,
            sample_count=sample_count,
            mean_latency_ms=mean_latency_ms,
            worker_utilization_cap=worker_utilization_cap,
            hit_count=hit_count,
            mean_abs_frame_error=mean_abs_frame_error,
        )
        if best is None or score < best[0]:
            best = (score, calibration)
    return best[1]
