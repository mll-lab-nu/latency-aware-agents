"""Shared decision metadata for latency-aware environments."""

from __future__ import annotations

from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import LatencyDistributionSpec
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy


def build_latency_info(
    *,
    episode_idx: Any,
    decision_step_idx: Any,
    env_step_idx: Any,
    applied_action: Any,
    pending_action_count: Any,
    latency_raw_frames: int,
    clock: EnvClock,
    hold_policy: ActionHoldPolicy,
    ordering_policy: OrderingPolicy,
    latency_distribution_spec: LatencyDistributionSpec | None,
    latency_distribution_label: Any,
) -> dict[str, Any]:
    """Describe one decision using the same fields for scalar and GPU wrappers."""
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
