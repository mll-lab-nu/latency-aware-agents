from __future__ import annotations

from latency_bench.core.types import Action
from latency_bench.scheduler.persistence import HoldPreviousActionPolicy, NoopAfterOneShotPolicy


def test_hold_previous_keeps_action_across_frames():
    noop = Action("noop", is_noop=True)
    action = Action("move")
    policy = HoldPreviousActionPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action


def test_noop_after_oneshot_applies_once_then_noop():
    noop = Action("noop", is_noop=True)
    action = Action("flap", is_oneshot=True)
    policy = NoopAfterOneShotPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is noop


def test_noop_after_oneshot_holds_non_oneshot_actions():
    noop = Action("noop", is_noop=True)
    action = Action("hold")
    policy = NoopAfterOneShotPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action

