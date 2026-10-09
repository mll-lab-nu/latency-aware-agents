from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy
from latency_bench.logging.metrics import compute_episode_metrics

from helpers import make_action_event as _event


def test_latest_ready_ablation_can_be_requested_explicitly():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(),
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
        HoldPolicy(),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    scheduler.enqueue(_event(0, action, ready_step=0, clock=clock))

    for frame in range(10):
        assert scheduler.update(frame, clock.step_to_time_ms(frame)) is action

@pytest.mark.parametrize("delta,expected", [(5e-11, 1), (5e-10, 2)])
def test_timestamp_and_integer_readiness_share_the_log_rounding_boundary(delta, expected):
    clock = EnvClock(env_fps=35, obs_fps=35)
    for use_integer in (False, True):
        noop, action = Action("noop", is_noop=True), Action("move")
        timestamp = (1 + delta) * clock.frame_ms
        event = ActionEvent(0, action, 0, timestamp, 1 + delta,
                            LatencyRecord(0, timestamp),
                            ready_raw_frame=expected if use_integer else None)
        scheduler = ActionScheduler(clock, noop, HoldPolicy(), OrderingPolicy.ISSUE_ORDER_FIFO)
        scheduler.enqueue(event)
        assert scheduler.update(1, clock.frame_ms).value == ("move" if expected == 1 else "noop")
        assert scheduler.update(2, 2 * clock.frame_ms).value == "move"
        compute_episode_metrics(0, [], [event], [timestamp], frame_ms=clock.frame_ms)
        assert event.ready_raw_frame == expected
