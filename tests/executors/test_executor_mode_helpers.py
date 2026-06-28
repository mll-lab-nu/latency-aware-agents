from __future__ import annotations

import importlib

from latency_bench.core.types import Action, Observation

from executor_mode_helpers import SequencePolicyRunner, make_dummy_parts


def test_sequence_policy_runner_returns_actions_in_order_then_repeats_last():
    first = Action("first")
    second = Action("second")
    runner = SequencePolicyRunner([first, second])
    observation = Observation(data=None, env_step=0, sim_time_ms=0.0)

    assert runner.predict(observation).action is first
    assert runner.predict(observation).action is second
    assert runner.predict(observation).action is second


def test_make_dummy_parts_returns_reusable_executor_parts():
    env, clock, noop_action, jump_action, action_scheduler = make_dummy_parts(max_steps=3, env_fps=60.0)

    assert env.env_fps == 60.0
    assert clock.env_fps == 60.0
    assert noop_action.is_noop
    assert jump_action.value == "jump"
    assert action_scheduler.update(0, 0.0) is noop_action


def test_executor_mode_helpers_do_not_import_unimplemented_executors():
    module = importlib.import_module("executor_mode_helpers")

    assert not hasattr(module, "PausedLatencyExecutor")
    assert not hasattr(module, "RealtimeLatencyExecutor")
