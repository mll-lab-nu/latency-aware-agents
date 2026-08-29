from __future__ import annotations

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy
from latency_bench.executors.realtime.messages import (
    _action_event_dispatch_payload,
    _action_event_from_dispatch_payload,
)
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


def _chunk_event(
    *,
    action_id: int,
    obs_env_step: int,
    ready_raw_frame: int,
    latency_ms: float,
    chunk: list[Action],
) -> ActionEvent:
    return ActionEvent(
        action_id=action_id,
        action=chunk[0],
        action_chunk=chunk,
        obs_env_step=obs_env_step,
        ready_time_ms=ready_raw_frame * 100.0,
        ready_step_float=float(ready_raw_frame),
        ready_raw_frame=ready_raw_frame,
        latency=LatencyRecord(obs_id=action_id, latency_ms=latency_ms),
    )


def test_realtime_dispatch_preserves_action_chunk():
    chunk = np.zeros((40, 29), dtype=np.float32)
    event = ActionEvent(
        action_id=0,
        action=Action("first"),
        action_chunk=chunk,
        obs_env_step=0,
        ready_time_ms=0.0,
        ready_step_float=0.0,
        ready_raw_frame=0,
        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
    )

    restored = _action_event_from_dispatch_payload(_action_event_dispatch_payload(event))

    assert np.array_equal(restored.action_chunk, chunk)


def test_chunk_uses_elapsed_raw_steps_and_new_chunk_replaces_old():
    clock = EnvClock(env_fps=10, obs_fps=10)
    noop = Action("noop", is_noop=True)
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    old_chunk = [Action("old0"), Action("old1"), Action("old2")]
    scheduler.enqueue(
        _chunk_event(
            action_id=0,
            obs_env_step=0,
            ready_raw_frame=2,
            latency_ms=0.0,
            chunk=old_chunk,
        )
    )

    assert scheduler.update(0, 0.0) is noop
    assert scheduler.update(1, 100.0) is noop
    assert scheduler.update(2, 200.0) is old_chunk[2]
    assert scheduler.stale_prefix_frames == 2
    assert scheduler.chunk_metrics()["chunk_age_ms"] == 200.0

    new_chunk = [Action("new0"), Action("new1")]
    scheduler.enqueue(
        _chunk_event(
            action_id=1,
            obs_env_step=3,
            ready_raw_frame=3,
            latency_ms=0.0,
            chunk=new_chunk,
        )
    )
    assert scheduler.update(3, 300.0) is new_chunk[0]


def test_chunk_underflow_is_counted_and_can_hold_last_action():
    clock = EnvClock(env_fps=10, obs_fps=10)
    noop = Action("noop", is_noop=True)
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        OrderingPolicy.ISSUE_ORDER_FIFO,
        hold_last_chunk_action=True,
    )
    chunk = [Action("last")]
    scheduler.enqueue(
        _chunk_event(
            action_id=0,
            obs_env_step=0,
            ready_raw_frame=0,
            latency_ms=0.0,
            chunk=chunk,
        )
    )

    assert scheduler.update(0, 0.0) is chunk[0]
    assert scheduler.update(1, 100.0) is chunk[0]
    assert scheduler.update(2, 200.0) is chunk[0]
    assert scheduler.underflow_count == 1
    assert scheduler.underflow_frames == 2
    assert scheduler.underflow_duration_ms == 200.0
    assert scheduler.get_applied_events()[0].metadata["underflow_frames"] == 2
