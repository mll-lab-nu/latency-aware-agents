from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Any

from latency_bench.core.clock import EnvClock, ms_to_raw_frames
from latency_bench.core.latency_distribution import LatencyDistributionSpec
from latency_bench.latency.samplers import HiddenRegimeSampler, JointBurstSampler, build_sampler

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


def build_profile_sampler(
    profile_path: str,
    *,
    method: str,
    seed: int,
    env_fps: float,
    obs_fps: float,
):
    return build_sampler(
        {"method": method, "profile_path": profile_path, "seed": int(seed)},
        clock=EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps)),
    )


def temporal_profile_sampler_for_reset(
    sampler: HiddenRegimeSampler | JointBurstSampler, seed: int,
) -> HiddenRegimeSampler | JointBurstSampler:
    """Start an independent temporal stream for one explicitly seeded rollout."""
    return type(sampler)(
        sampler.model,
        {"worker_slots": sampler.distributions},
        clock=sampler.clock,
        seed=seed,
    )


def build_iid_profile_sampler(
    profile_path: str,
    *,
    seed: int,
    env_fps: float,
    obs_fps: float,
):
    return build_profile_sampler(
        profile_path,
        method="iid",
        seed=seed,
        env_fps=env_fps,
        obs_fps=obs_fps,
    )


def profile_sampler_max_latency_raw_frames(sampler: Any, *, worker_slot: int, env_fps: float) -> int:
    """Maximum ready delay over the sampler's generated support."""
    slot = str(worker_slot)
    distribution = sampler.distributions[slot]["steady" if isinstance(sampler, (HiddenRegimeSampler, JointBurstSampler)) else "all"]
    max_latency_ms = max(distribution["observation_to_action_latency_ms"]["latency_ms"])
    if isinstance(sampler, HiddenRegimeSampler):
        for rank in sampler.rank_processes:
            residual = rank["level_residual_ms"]
            spike_ms = max(
                max(rank["level_ms"]["latency_ms"]) + (max(residual["latency_ms"]) if residual is not None else 0.0),
                math.nextafter(sampler.model["spike_threshold_ms_by_worker_slot"][slot], math.inf),
            )
            max_latency_ms = max(max_latency_ms, spike_ms + max(sampler.pre_worker_time["latency_ms"]))
    elif isinstance(sampler, JointBurstSampler):
        for event in [*sampler.model["initial_events"], *sampler.model["burst_events"]]:
            if event is not None and slot in event["worker_slots"]:
                worker = event["worker_slots"][slot]
                for name in ("spike", "steady"):
                    if worker[name] is not None:
                        max_latency_ms = max(max_latency_ms, max(worker[name]["observation_to_action_latency_ms"]["latency_ms"]))
    return latency_ms_to_raw_frames(max_latency_ms, env_fps=env_fps)


def profile_sampler_max_latency_raw_frames_for_workers(
    sampler: Any,
    *,
    profile_worker_slot: int,
    simulated_worker_capacity: int | None,
    env_fps: float,
) -> int:
    worker_slots = (
        [profile_worker_slot]
        if simulated_worker_capacity is None
        else range(simulated_worker_capacity)
    )
    return max(
        profile_sampler_max_latency_raw_frames(
            sampler,
            worker_slot=worker_slot,
            env_fps=env_fps,
        )
        for worker_slot in worker_slots
    )
