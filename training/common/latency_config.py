from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock, ms_to_raw_frames
from latency_bench.core.latency_distribution import LatencyDistributionSpec
from latency_bench.latency.samplers import build_sampler

from training.common.utils import env_instance_seed


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


def build_latency_info(
    *,
    episode_idx: Any,
    decision_step_idx: Any,
    env_step_idx: Any,
    applied_action: Any,
    pending_action_count: Any,
    latency_raw_frames: int,
    clock: Any,
    hold_policy: Any,
    ordering_policy: Any,
    latency_distribution_spec: LatencyDistributionSpec | None,
    latency_distribution_label: Any,
) -> dict[str, Any]:
    latency_raw_frames = int(latency_raw_frames)
    info: dict[str, Any] = {
        "episode_idx": int(episode_idx),
        "decision_step_idx": int(decision_step_idx),
        "env_step_idx": int(env_step_idx),
        "raw_frame": int(env_step_idx),
        "applied_action": applied_action,
        "pending_action_count": int(pending_action_count),
        "latency_raw_frames": latency_raw_frames,
        "latency_ms": float(latency_raw_frames) * clock.frame_ms,
        "env_fps": float(clock.env_fps),
        "obs_fps": float(clock.obs_fps),
        "obs_stride_raw_frames": int(clock.obs_stride_raw_frames),
        "hold_policy": hold_policy.value,
        "ordering_policy": ordering_policy.value,
    }
    if latency_distribution_spec is not None:
        info["latency_distribution"] = latency_distribution_spec.to_dict()
        info["latency_distribution_label"] = str(latency_distribution_label)
    return info


def sample_factory_latency_params_from_flat_config(cfg: Any, *, env_fps: float) -> SampleFactoryLatencyParams:
    latency_type = cfg.latency_type
    if latency_type == "zero":
        return SampleFactoryLatencyParams(fixed_latency_raw_frames=0, latency_distribution_spec=None)
    if latency_type == "fixed":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=latency_ms_to_raw_frames(cfg.fixed_latency_ms, env_fps=env_fps),
            latency_distribution_spec=None,
        )
    if latency_type == "iid":
        return SampleFactoryLatencyParams(
            fixed_latency_raw_frames=0,
            latency_distribution_spec=None,
            profile_path=cfg.latency_profile_path,
            profile_worker_slot=cfg.latency_profile_worker_slot,
        )
    return SampleFactoryLatencyParams(
        fixed_latency_raw_frames=0,
        latency_distribution_spec=latency_distribution_spec_from_flat_config(cfg, env_fps=env_fps),
        sample_distribution_per_episode=latency_type in _PER_EPISODE_LATENCY_TYPES,
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


def build_iid_profile_sampler(
    profile_path: str,
    *,
    seed: int,
    env_fps: float,
    obs_fps: float,
):
    return build_sampler(
        {"method": "iid", "profile_path": profile_path, "seed": int(seed)},
        clock=EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps)),
    )


def profile_sampler_max_latency_raw_frames(sampler: Any, *, worker_slot: int, env_fps: float) -> int:
    distribution = sampler.distributions[str(worker_slot)]["all"]
    latency_ms = max(distribution["observation_to_action_latency_ms"]["latency_ms"])
    return latency_ms_to_raw_frames(latency_ms, env_fps=env_fps)
