from __future__ import annotations

from latency_bench.core.types import Action

from latency_bench.scheduler.persistence import (
    HoldPolicy,
    OneFrameThenNoopPolicy,
)


def test_hold_policy_keeps_current_action():
    noop = Action("noop", is_noop=True)
    action = Action("move")
    policy = HoldPolicy()

    policy.on_action_applied(action)

    assert [policy.get_action_for_step(action, noop) for _ in range(10)] == [action] * 10


def test_one_frame_then_noop_applies_any_action_once_then_noop():
    noop = Action("noop", is_noop=True)
    action = Action("move", is_oneshot=False)
    policy = OneFrameThenNoopPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is noop


def test_one_frame_then_noop_reset_discards_pending_action():
    noop = Action("noop", is_noop=True)
    action = Action("hold", is_oneshot=False)
    policy = OneFrameThenNoopPolicy()

    policy.on_action_applied(action)
    policy.reset()
    assert policy.get_action_for_step(action, noop) is noop
