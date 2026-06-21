"""Shared test fixtures and helpers reused across the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord


def realtime_profile(bench_overhead_ms: float = 0.0) -> dict[str, float]:
    return {
        "env_observe_ms": 0.0,
        "env_to_parent_ipc_ms": 0.0,
        "parent_observation_handle_ms": 0.0,
        "parent_to_worker_ipc_ms": 0.0,
        "worker_recv_overhead_ms": 0.0,
        "worker_to_parent_ipc_ms": 0.0,
        "parent_release_ms": 0.0,
        "parent_to_env_ipc_ms": bench_overhead_ms,
        "env_action_recv_to_enqueue_ms": 0.0,
        "env_scheduler_update_ms": 0.0,
        "worker_observation_attach_ms": 0.0,
        "worker_observation_localize_ms": 0.0,
        "policy_build_example_ms": 0.0,
        "policy_wrapper_predict_action_ms": 0.0,
        "policy_decode_ms": 0.0,
        "env_step_ms": 0.0,
        "env_wall_lag_ms": 0.0,
    }


def make_action_event(action_id: int, action: Action, ready_step: int, clock: EnvClock) -> ActionEvent:
    ready_time_ms = clock.step_to_time_ms(ready_step)
    return ActionEvent(
        action_id=action_id,
        action=action,
        obs_env_step=0,
        ready_time_ms=ready_time_ms,
        ready_step_float=float(ready_step),
        latency=LatencyRecord(
            obs_id=action_id,
            latency_ms=ready_time_ms,
        ),
    )


def save_test_image(path: Path, fill: int) -> None:
    from PIL import Image

    Image.fromarray(np.full((2, 2, 3), fill, dtype=np.uint8)).save(path)
