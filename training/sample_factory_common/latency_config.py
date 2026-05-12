from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.core.latency_distribution import LatencyDistributionSpec
from latency_bench.core.types import LatencySourceType

from training.sample_factory_common.common import env_instance_seed


@dataclass(frozen=True)
class SampleFactoryLatencyParams:
    fixed_latency_raw_frames: int
    latency_distribution_spec: LatencyDistributionSpec | None


def sample_factory_latency_seed_from_flat_config(
    cfg: Any,
    *,
    fallback_seed: int,
    worker_index: int = 0,
    vector_index: int = 0,
    env_id: int = 0,
) -> int:
    latency_seed = getattr(cfg, "latency_seed", None)
    if latency_seed is None:
        return int(fallback_seed)
    return env_instance_seed(
        base_seed=int(latency_seed),
        worker_index=int(worker_index),
        vector_index=int(vector_index),
        env_id=int(env_id),
    )


def latency_ms_to_raw_frames(latency_ms: Any, *, env_fps: float) -> int:
    return ms_to_raw_frames(float(latency_ms), float(env_fps))


def latency_ms_to_raw_frame_value(latency_ms: Any, *, env_fps: float) -> float:
    return float(latency_ms) * float(env_fps) / 1000.0


def raw_frames_to_latency_ms(raw_frames: Any, *, env_fps: float) -> float:
    return float(raw_frames) * 1000.0 / float(env_fps)


def sample_factory_latency_params_from_flat_config(cfg: Any, *, env_fps: float) -> SampleFactoryLatencyParams:
    latency_type = LatencySourceType(str(getattr(cfg, "latency_type")))
    if latency_type == LatencySourceType.ZERO:
        return SampleFactoryLatencyParams(fixed_latency_raw_frames=0, latency_distribution_spec=None)
    if latency_type == LatencySourceType.FIXED:
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=latency_ms_to_raw_frames(getattr(cfg, "fixed_latency_ms"), env_fps=env_fps),
            latency_distribution_spec=None,
        )
    return SampleFactoryLatencyParams(
        fixed_latency_raw_frames=0,
        latency_distribution_spec=latency_distribution_spec_from_flat_config(cfg, env_fps=env_fps),
    )


def latency_distribution_spec_from_flat_config(cfg: Any, *, env_fps: float) -> LatencyDistributionSpec:
    latency_type = LatencySourceType(str(getattr(cfg, "latency_type")))
    if latency_type == LatencySourceType.NORMAL_DISTRIBUTION:
        return LatencyDistributionSpec(
            type="normal",
            mean=latency_ms_to_raw_frame_value(getattr(cfg, "mean_latency_ms"), env_fps=env_fps),
            std=latency_ms_to_raw_frame_value(getattr(cfg, "std_latency_ms"), env_fps=env_fps),
            min=latency_ms_to_raw_frame_value(getattr(cfg, "min_latency_ms"), env_fps=env_fps),
            max=latency_ms_to_raw_frame_value(getattr(cfg, "max_latency_ms"), env_fps=env_fps),
        )
    return LatencyDistributionSpec(
        type="uniform",
        mean=None,
        std=None,
        min=latency_ms_to_raw_frame_value(getattr(cfg, "min_latency_ms"), env_fps=env_fps),
        max=latency_ms_to_raw_frame_value(getattr(cfg, "max_latency_ms"), env_fps=env_fps),
    )
