from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy, Observation
from latency_bench.executors.realtime.messages import (
    _action_event_dispatch_payload,
    _action_event_from_dispatch_payload,
)
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy
from latency_bench.policy.action_prefix import action_prefix_arrays, with_action_prefix
from latency_bench.policy.starvla_tasks import TaskStarVlaPolicyRunner


@pytest.mark.parametrize(("prefix_frames", "delay", "expected"), [
    (0, 0, [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111]),
    (0, 2, [0, 0, 3, 4, 23, 24, 43, 44, 63, 64, 83, 84]),
    (2, 0, [0, 0, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93]),
    (2, 1, [0, 0, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93]),
    (2, 2, [0, 0, 3, 4, 23, 24, 43, 44, 63, 64, 83, 84]),
    (2, 5, [0, 0, 0, 0, 0, 6, 7, 8, 8, 8, 56, 57]),
])
def test_async_chunk_timeline_preserves_frozen_prefix(prefix_frames, delay, expected):
    """Long-lived scheduler/policy contract: arrival age and prefix are independent."""
    scheduler = ActionScheduler(EnvClock(env_fps=20, obs_fps=20), Action([0.0]),
                                HoldPolicy(), OrderingPolicy.LATEST_READY,
                                hold_last_chunk_action=True)
    policy = TaskStarVlaPolicyRunner(
        SimpleNamespace(predict_action=lambda **_: {
            "normalized_actions": (2 * (frame * 10 + np.arange(8) + 1) - 1).reshape(1, 8, 1),
        }),
        policy_config={"state_info_key": "state", "image_views_info_key": "views"},
        model_config={"datasets": {"vla_data": {"obs_image_size": [8, 8], "image_mode": "single"}}},
        contract={"task": "timeline", "state_dim": 1, "prompt": "task", "normalization": {
            "state": {"min": [0], "max": [1]}, "action": {"min": [0], "max": [1]},
        }},
    )
    settings = ({"mode": "fixed_frames", "frames": prefix_frames, "action_horizon": 8,
                 "action_dim": 1} if prefix_frames else {"mode": "none"})
    # Run two episodes through the same objects to cover resetting an active plan.
    for _ in range(2):
        scheduler.reset()
        pending = None
        actual = []
        for frame in range(len(expected)):
            if pending is not None and pending.ready_raw_frame == frame:
                scheduler.enqueue(pending)
                pending = None
            if pending is None:
                observation = with_action_prefix(Observation(None, frame, frame * 50, {
                    "state": np.zeros(1), "views": [np.zeros((8, 8, 3), dtype=np.uint8)],
                }), scheduler, settings)
                output = policy.predict(observation)
                pending = _chunk_event(action_id=frame, obs_env_step=frame,
                    ready_raw_frame=frame + delay, latency_ms=delay * 50,
                    chunk=[Action(value) for value in output.action_chunk])
                if delay == 0:
                    scheduler.enqueue(pending)
                    pending = None
            actual.append(scheduler.update(frame, frame * 50).value[0])
        assert actual == expected


@pytest.mark.parametrize("delay", [1, 3, 5])
def test_fixed_prefix_and_arrival_index_are_independent(delay):
    """Evaluation contract: known actions condition K heads, arrival selects head d."""
    scheduler = ActionScheduler(EnvClock(env_fps=50, obs_fps=50), Action([0.0]),
                                HoldPolicy(), OrderingPolicy.LATEST_READY,
                                hold_last_chunk_action=True)
    old = [Action([float(i + 1)]) for i in range(8)]
    scheduler.enqueue(_chunk_event(action_id=0, obs_env_step=0, ready_raw_frame=0,
                                   latency_ms=0, chunk=old))
    assert scheduler.update(0, 0) is old[0]
    prefix, mask = action_prefix_arrays(scheduler, 1, {"mode": "profile_mean", "frames": 3, "action_horizon": 8, "action_dim": 1})
    np.testing.assert_array_equal(prefix[:3, 0], [2, 3, 4])
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False, False, False])
    assert action_prefix_arrays(scheduler, 1, {"mode": "none"}) == (None, None)
    new = [Action([prefix[i, 0] if mask[i] else 100.0 + i]) for i in range(8)]
    scheduler.enqueue(_chunk_event(action_id=1, obs_env_step=1, ready_raw_frame=1 + delay,
                                   latency_ms=delay * 20, chunk=new))
    for frame in range(1, 1 + delay):
        assert scheduler.update(frame, frame * 20) is old[frame]
    assert scheduler.update(1 + delay, (1 + delay) * 20) is new[delay]


def test_evaluation_prefix_uses_startup_and_hold_without_mutating_live_queue():
    scheduler = ActionScheduler(EnvClock(env_fps=50, obs_fps=50), Action([7.0]),
                                HoldPolicy(), OrderingPolicy.LATEST_READY,
                                hold_last_chunk_action=True)
    prefix, _ = action_prefix_arrays(scheduler, 0, {"mode": "profile_mean", "frames": 3, "action_horizon": 4, "action_dim": 1})
    np.testing.assert_array_equal(prefix[:3, 0], [7, 7, 7])
    scheduler.enqueue(_chunk_event(action_id=0, obs_env_step=0, ready_raw_frame=0,
                                   latency_ms=0, chunk=[Action([9.0])]))
    prefix, _ = action_prefix_arrays(scheduler, 0, {"mode": "profile_mean", "frames": 3, "action_horizon": 4, "action_dim": 1})
    np.testing.assert_array_equal(prefix[:3, 0], [9, 9, 9])
    assert scheduler.applied_event_count == 0
    assert scheduler.underflow_frames == 0


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
        HoldPolicy(),
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
        HoldPolicy(),
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
    assert scheduler.get_applied_events()[0].metadata["chunk_index"] == 0
    assert scheduler.underflow_duration_ms == 200.0
    assert scheduler.get_applied_events()[0].metadata["underflow_frames"] == 2


def test_returned_action_state_and_prefix_share_the_observation_snapshot():
    """Long-lived Balance input contract: preserve normalized chunk/index for baseline too."""
    from latency_bench.core.types import Observation
    from latency_bench.policy.action_prefix import with_action_prefix

    scheduler = ActionScheduler(EnvClock(env_fps=50, obs_fps=50), Action([0.0]),
                                HoldPolicy(), OrderingPolicy.LATEST_READY, hold_last_chunk_action=True)
    old = [Action([value]) for value in [0.0, 1.0, 2.0, 3.0]]
    scheduler.enqueue(_chunk_event(action_id=0, obs_env_step=0, ready_raw_frame=0,
                                   latency_ms=0, chunk=old))
    scheduler.update(0, 0)
    observation = Observation(data=None, env_step=1, sim_time_ms=20, metadata={
        "gymnasium_state": np.array([42.0]),
        "returned_action_context": {"state_key": "gymnasium_state", "horizon": 4,
                                    "frames": None, "order": [0], "low": np.array([0.0]),
                                    "high": np.array([4.0])},
    })
    request = with_action_prefix(observation, scheduler, {"mode": "none"})
    np.testing.assert_array_equal(request.metadata["gymnasium_state"],
                                  [42, 0, -0.5, 0, 0.5, 0, 1, 0, 0])
    assert "action_prefix" not in request.metadata
    assert "returned_action_context" not in request.metadata
    assert scheduler.update(1, 20) is old[1]
