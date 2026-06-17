from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy

from helpers import make_action_event as _event


def test_fixed_two_frame_latency_action_applies_at_correct_step():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    action = Action("jump")
    scheduler = ActionScheduler(clock, noop, HoldPolicy(duration_raw_frames=100))
    scheduler.enqueue(_event(0, action, ready_step=2, clock=clock))

    assert scheduler.update(0, clock.step_to_time_ms(0)) is noop
    assert scheduler.update(1, clock.step_to_time_ms(1)) is noop
    assert scheduler.update(2, clock.step_to_time_ms(2)) is action
    assert scheduler.get_applied_events()[0].apply_raw_frame == 2


def test_default_mode_is_issue_order_fifo():
    scheduler = ActionScheduler(
        EnvClock(env_fps=30, obs_fps=30),
        Action("noop", is_noop=True),
        HoldPolicy(duration_raw_frames=100),
    )

    assert scheduler.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO


def test_one_frame_latency_ready_on_float_boundary_applies_at_next_step():
    clock = EnvClock(env_fps=35, obs_fps=35)
    noop = Action("noop", is_noop=True)
    action = Action("jump")
    obs_env_step = 6
    ready_step = obs_env_step + 1
    ready_time_ms = clock.step_to_time_ms(obs_env_step) + clock.frame_ms
    event = ActionEvent(
        action_id=0,
        action=action,
        obs_env_step=obs_env_step,
        ready_time_ms=ready_time_ms,
        ready_step_float=clock.time_ms_to_step_float(ready_time_ms),
        latency=LatencyRecord(
            obs_id=0,
            latency_ms=clock.frame_ms,
        ),
    )
    scheduler = ActionScheduler(clock, noop, HoldPolicy(duration_raw_frames=100))
    scheduler.enqueue(event)

    assert ready_time_ms > clock.step_to_time_ms(ready_step)
    assert scheduler.update(ready_step, clock.step_to_time_ms(ready_step)) is action
    assert event.applied
    assert not event.dropped
    assert scheduler.get_pending_events() == []


def test_fifo_mode_applies_ready_actions_in_order():
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
    scheduler.enqueue(_event(0, first, ready_step=1, clock=clock))
    scheduler.enqueue(_event(1, second, ready_step=1, clock=clock))

    assert scheduler.update(1, clock.step_to_time_ms(1)) is first
    assert scheduler.update(2, clock.step_to_time_ms(2)) is second
    assert [event.action.value for event in scheduler.get_applied_events()] == ["first", "second"]


def test_fifo_mode_blocks_later_ready_action_until_head_of_line_ready():
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
    scheduler.enqueue(_event(1, second, ready_step=5, clock=clock))

    assert scheduler.update(5, clock.step_to_time_ms(5)) is noop
    assert scheduler.get_applied_events() == []

    assert scheduler.update(10, clock.step_to_time_ms(10)) is first
    assert scheduler.update(11, clock.step_to_time_ms(11)) is second
    assert [event.action.value for event in scheduler.get_applied_events()] == ["first", "second"]


def test_latest_ready_mode_allows_explicit_overtake_ablation():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=100),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    scheduler.enqueue(_event(0, first, ready_step=10, clock=clock))
    scheduler.enqueue(_event(1, second, ready_step=5, clock=clock))

    assert scheduler.update(5, clock.step_to_time_ms(5)) is second
    assert scheduler.get_applied_events()[0].action is second
    assert scheduler.get_pending_events()[0].action is first


def test_latest_ready_mode_drops_older_ready_actions():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    first_event = _event(0, first, ready_step=1, clock=clock)
    second_event = _event(1, second, ready_step=1, clock=clock)
    scheduler = ActionScheduler(
        clock,
        noop,
        HoldPolicy(duration_raw_frames=100),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    scheduler.enqueue(first_event)
    scheduler.enqueue(second_event)

    assert scheduler.update(1, clock.step_to_time_ms(1)) is second
    assert first_event.dropped
    assert not second_event.dropped
    assert scheduler.get_pending_events() == []


def test_applied_event_incremental_access_does_not_require_full_copy():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    first = Action("first")
    second = Action("second")
    scheduler = ActionScheduler(clock, noop, HoldPolicy(duration_raw_frames=100))
    scheduler.enqueue(_event(0, first, ready_step=0, clock=clock))
    scheduler.enqueue(_event(1, second, ready_step=1, clock=clock))

    assert scheduler.latest_applied_event is None
    assert scheduler.update(0, clock.step_to_time_ms(0)) is first
    first_applied = scheduler.get_applied_events_since(0)

    assert [event.action.value for event in first_applied] == ["first"]
    assert scheduler.latest_applied_event is first_applied[-1]

    cursor = scheduler.applied_event_count
    assert scheduler.update(1, clock.step_to_time_ms(1)) is second
    second_applied = scheduler.get_applied_events_since(cursor)

    assert [event.action.value for event in second_applied] == ["second"]
    assert scheduler.latest_applied_event is second_applied[-1]


def test_persistence_policy_controls_active_action():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    oneshot = Action("flap", is_oneshot=True)
    scheduler = ActionScheduler(clock, noop, OneFrameThenNoopPolicy())
    scheduler.enqueue(_event(0, oneshot, ready_step=0, clock=clock))

    assert scheduler.update(0, clock.step_to_time_ms(0)) is oneshot
    assert scheduler.update(1, clock.step_to_time_ms(1)) is noop


def test_reset_clears_pending_and_active_action():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", is_noop=True)
    action = Action("jump")
    scheduler = ActionScheduler(clock, noop, HoldPolicy(duration_raw_frames=100))
    scheduler.enqueue(_event(0, action, ready_step=0, clock=clock))
    assert scheduler.update(0, clock.step_to_time_ms(0)) is action

    scheduler.reset()

    assert scheduler.get_pending_events() == []
    assert scheduler.get_applied_events() == []
    assert scheduler.update(1, clock.step_to_time_ms(1)) is noop
