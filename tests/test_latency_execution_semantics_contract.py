from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, Observation, OrderingPolicy
from latency_bench.executors.realtime import BoundedObservationQueue, RealtimeObservationTask
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


def _event(action_id: int, action: Action, ready_step: int, clock: EnvClock) -> ActionEvent:
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


def _task(obs_id: int) -> RealtimeObservationTask:
    return RealtimeObservationTask(
        obs_id=obs_id,
        observation=Observation(data=f"obs-{obs_id}", env_step=obs_id, sim_time_ms=float(obs_id)),
        submit_env_step=obs_id,
        submit_time_ms=float(obs_id),
    )


def test_env_fps_obs_fps_validation_and_stride_contract():
    clock = EnvClock(env_fps=60, obs_fps=15)
    assert clock.env_fps == 60
    assert clock.obs_fps == 15
    assert clock.obs_stride_raw_frames == 4

    with pytest.raises(ValueError, match="env_fps.*obs_fps"):
        EnvClock(env_fps=60, obs_fps=17)


def test_latency_ms_to_raw_frames_uses_ceil_contract():
    clock = EnvClock(env_fps=30, obs_fps=10)

    assert clock.latency_ms_to_raw_frames(0.0) == 0
    assert clock.latency_ms_to_raw_frames(1.0) == 1
    assert clock.latency_ms_to_raw_frames(1000.0 / 30.0) == 1
    assert clock.latency_ms_to_raw_frames((1000.0 / 30.0) + 0.001) == 2
    assert clock.latency_ms_to_raw_frames(100.0) == 3


def test_issue_order_fifo_blocks_later_ready_action_contract():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=100),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    scheduler.enqueue(_event(0, first, ready_step=10, clock=clock))
    scheduler.enqueue(_event(1, second, ready_step=1, clock=clock))

    assert scheduler.update(1, clock.step_to_time_ms(1)) is noop
    assert scheduler.get_applied_events() == []

    assert scheduler.update(10, clock.step_to_time_ms(10)) is first
    assert scheduler.update(11, clock.step_to_time_ms(11)) is second


def test_latest_ready_is_explicit_ablation_not_default_contract():
    clock = EnvClock(env_fps=30, obs_fps=30)
    scheduler = ActionScheduler(
        clock,
        Action("noop", is_noop=True),
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
    )
    assert scheduler.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO


def test_latest_ready_ablation_can_be_requested_explicitly():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    first_event = _event(0, first, ready_step=1, clock=clock)
    second_event = _event(1, second, ready_step=1, clock=clock)
    scheduler.enqueue(first_event)
    scheduler.enqueue(second_event)

    assert scheduler.update(1, clock.step_to_time_ms(1)) is second
    assert first_event.dropped
    assert second_event.applied

def test_one_frame_then_noop_hold_policy_contract():
    from latency_bench.scheduler.persistence import OneFrameThenNoopPolicy

    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    action = Action("move", is_oneshot=False)
    scheduler = ActionScheduler(
        clock,
        noop,
        OneFrameThenNoopPolicy(),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    scheduler.enqueue(_event(0, action, ready_step=0, clock=clock))

    assert scheduler.update(0, clock.step_to_time_ms(0)) is action
    assert scheduler.update(1, clock.step_to_time_ms(1)) is noop


def test_hold_policy_contract():
    clock = EnvClock(env_fps=30, obs_fps=10)
    noop = Action("noop", is_noop=True)
    action = Action("move")
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    scheduler.enqueue(_event(0, action, ready_step=0, clock=clock))

    assert scheduler.update(0, clock.step_to_time_ms(0)) is action
    assert scheduler.update(1, clock.step_to_time_ms(1)) is action
    assert scheduler.update(2, clock.step_to_time_ms(2)) is action
    assert scheduler.update(3, clock.step_to_time_ms(3)) is noop

@pytest.mark.xfail(
    strict=True,
    reason="Current realtime queue exposes drop-oldest behavior; Phase 4 should make drop_new_observation the default.",
)
def test_realtime_queue_full_default_is_drop_new_observation_contract():
    queue = BoundedObservationQueue(max_len=1)
    first = _task(0)
    second = _task(1)

    assert queue.push_drop_oldest(first) is None
    assert queue.push_drop_oldest(second) is second
    assert queue.pop() is first


@pytest.mark.xfail(
    strict=True,
    reason="Phase 4 has not introduced timeout/skipped job state or FIFO timeout unblocking.",
)
def test_timeout_marks_skipped_unblocks_fifo_and_ignores_late_results_contract():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    scheduler.enqueue(_event(0, Action("first"), ready_step=100, clock=clock))
    scheduler.enqueue(_event(1, Action("second"), ready_step=1, clock=clock))

    scheduler.mark_timed_out(action_id=0, timeout_raw_frame=2)
    assert scheduler.update(2, clock.step_to_time_ms(2)).value == "second"

    scheduler.ignore_late_result(action_id=0)
    assert [event.action_id for event in scheduler.get_applied_events()] == [1]


@pytest.mark.xfail(
    strict=True,
    reason="Phase 5 has not added the required raw-frame and job-status logging fields.",
)
def test_required_action_event_logging_fields_contract():
    required_fields = {
        "action_age_raw_frames",
        "job_status",
    }

    assert required_fields.issubset(ActionEvent.__dataclass_fields__)
