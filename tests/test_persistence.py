from __future__ import annotations

from latency_bench.core.types import Action
import pytest

from latency_bench.scheduler.persistence import (
    HoldForFixedDurationPolicy,
    HoldPreviousActionPolicy,
    HoldUntilNextActionPolicy,
    NoopAfterOneShotPolicy,
    OneFrameThenNoopPolicy,
)


def test_hold_previous_keeps_action_across_frames():
    noop = Action("noop", is_noop=True)
    action = Action("move")
    policy = HoldPreviousActionPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action


def test_hold_until_next_action_keeps_action_across_frames():
    noop = Action("noop", is_noop=True)
    action = Action("move")
    policy = HoldUntilNextActionPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action


def test_one_frame_then_noop_applies_any_action_once_then_noop():
    noop = Action("noop", is_noop=True)
    action = Action("move", is_oneshot=False)
    policy = OneFrameThenNoopPolicy()

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is noop


def test_hold_for_fixed_duration_applies_action_for_configured_frames():
    noop = Action("noop", is_noop=True)
    action = Action("left")
    policy = HoldForFixedDurationPolicy(duration_raw_frames=3)

    policy.on_action_applied(action)

    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is action
    assert policy.get_action_for_step(action, noop) is noop


def test_hold_for_fixed_duration_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration_raw_frames"):
        HoldForFixedDurationPolicy(duration_raw_frames=0)


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
