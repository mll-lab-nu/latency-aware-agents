from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.core.latency_distribution import LatencyDistributionSpec

from training.common.utils import env_instance_seed, str2bool


_NORMAL_LATENCY_TYPES = {"normal_distribution", "mixed_normal_distribution"}
_UNIFORM_LATENCY_TYPES = {"uniform_distribution", "mixed_uniform_distribution"}
_PER_EPISODE_LATENCY_TYPES = {"mixed_normal_distribution", "mixed_uniform_distribution"}


@dataclass(frozen=True)
class SampleFactoryLatencyParams:
    fixed_latency_raw_frames: int
    latency_distribution_spec: LatencyDistributionSpec | None
    sample_distribution_per_episode: bool = False
    profile_path: str | None = None
    profile_worker_slot: int | None = None
    profile_method: str | None = None
    simulated_worker_capacity: int | None = None


def add_sample_factory_latency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latency-type", dest="latency_type", type=str, default="zero")
    parser.add_argument("--fixed-latency-ms", dest="fixed_latency_ms", type=float)
    parser.add_argument("--mean-latency-ms", dest="mean_latency_ms", type=float)
    parser.add_argument("--std-latency-ms", dest="std_latency_ms", type=float)
    parser.add_argument("--min-latency-ms", dest="min_latency_ms", type=float)
    parser.add_argument("--max-latency-ms", dest="max_latency_ms", type=float)
    parser.add_argument("--latency-seed", dest="latency_seed", type=int)
    parser.add_argument("--latency-profile-path", dest="latency_profile_path", type=str)
    parser.add_argument("--latency-profile-worker-slot", dest="latency_profile_worker_slot", type=int)
    parser.add_argument("--simulated-worker-capacity", dest="simulated_worker_capacity", type=int)
    parser.add_argument("--add-latency-info", type=str2bool, default=False)
    parser.add_argument("--max-pending-actions", dest="max_pending_actions", type=int)


def sample_factory_latency_seed_from_flat_config(
    cfg: Any,
    *,
    fallback_seed: int,
    env_id: int,
) -> int:
    latency_seed = getattr(cfg, "latency_seed", None)
    if latency_seed is None:
        return int(fallback_seed)
    return env_instance_seed(
        base_seed=int(latency_seed),
        env_id=env_id,
    )


def latency_ms_to_raw_frames(latency_ms: Any, *, env_fps: float) -> int:
    return ms_to_raw_frames(float(latency_ms), float(env_fps))


def latency_ms_to_raw_frame_value(latency_ms: Any, *, env_fps: float) -> float:
    return float(latency_ms) * float(env_fps) / 1000.0


def sample_factory_latency_params_from_flat_config(cfg: Any, *, env_fps: float) -> SampleFactoryLatencyParams:
    latency_type = cfg.latency_type
    simulated_worker_capacity = cfg.simulated_worker_capacity
    if latency_type == "zero":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=0,
            latency_distribution_spec=None,
            simulated_worker_capacity=simulated_worker_capacity,
        )
    if latency_type == "fixed":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=latency_ms_to_raw_frames(cfg.fixed_latency_ms, env_fps=env_fps),
            latency_distribution_spec=None,
            simulated_worker_capacity=simulated_worker_capacity,
        )
    if latency_type == "iid":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=0,
            latency_distribution_spec=None,
            profile_path=cfg.latency_profile_path,
            profile_worker_slot=cfg.latency_profile_worker_slot,
            profile_method="iid",
            simulated_worker_capacity=simulated_worker_capacity,
        )
    if latency_type == "profile_sample":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=0,
            latency_distribution_spec=None,
            profile_path=cfg.latency_profile_path,
            profile_worker_slot=cfg.latency_profile_worker_slot,
            profile_method="temporal",
            simulated_worker_capacity=simulated_worker_capacity,
        )
    return SampleFactoryLatencyParams(
        fixed_latency_raw_frames=0,
        latency_distribution_spec=latency_distribution_spec_from_flat_config(cfg, env_fps=env_fps),
        sample_distribution_per_episode=latency_type in _PER_EPISODE_LATENCY_TYPES,
        simulated_worker_capacity=simulated_worker_capacity,
    )


def latency_distribution_spec_from_flat_config(cfg: Any, *, env_fps: float) -> LatencyDistributionSpec:
    latency_type = cfg.latency_type
    if latency_type in _NORMAL_LATENCY_TYPES:
        return LatencyDistributionSpec(
            type="normal",
            mean=latency_ms_to_raw_frame_value(cfg.mean_latency_ms, env_fps=env_fps),
            std=latency_ms_to_raw_frame_value(cfg.std_latency_ms, env_fps=env_fps),
            min=latency_ms_to_raw_frame_value(cfg.min_latency_ms, env_fps=env_fps),
            max=latency_ms_to_raw_frame_value(cfg.max_latency_ms, env_fps=env_fps),
        )
    if latency_type in _UNIFORM_LATENCY_TYPES:
        return LatencyDistributionSpec(
            type="uniform",
            mean=None,
            std=None,
            min=latency_ms_to_raw_frame_value(cfg.min_latency_ms, env_fps=env_fps),
            max=latency_ms_to_raw_frame_value(cfg.max_latency_ms, env_fps=env_fps),
        )
    raise ValueError(f"Sample Factory latency_distribution_spec does not support latency_type={latency_type}")
