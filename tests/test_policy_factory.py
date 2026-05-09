from __future__ import annotations

import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.base import ConstantPolicyRunner, RandomPolicyRunner
from latency_bench.policy.factory import build_policy


def _config(policy: dict) -> dict:
    return {
        "experiment": {
            "name": "policy_factory_test",
            "seed": 7,
        },
        "env": {
            "name": "flappy",
            "env_fps": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": 0,
                "flap": 1,
            },
            "oneshot_actions": ["flap"],
        },
        "policy": policy,
    }


def test_random_policy_type_builds_random_policy_runner():
    config = _config({"type": "random", "actions": ["noop", "flap"]})

    policy = build_policy(config, ActionResolver(config["env"]))

    assert isinstance(policy, RandomPolicyRunner)
    assert [action.value for action in policy.actions] == [0, 1]


def test_constant_policy_type_builds_constant_policy_runner():
    config = _config({"type": "constant", "action": "noop"})

    policy = build_policy(config, ActionResolver(config["env"]))

    assert isinstance(policy, ConstantPolicyRunner)
    assert policy.action.value == 0
    assert policy.action.is_noop


def test_unknown_policy_type_reports_registered_options():
    config = _config({"type": "unknown"})

    with pytest.raises(ValueError, match="Unsupported policy.type: unknown"):
        build_policy(config, ActionResolver(config["env"]))


def test_flappy_constant_flap_resolves_to_env_action_value():
    config = _config({"type": "constant", "action": "flap"})

    policy = build_policy(config, ActionResolver(config["env"]))

    assert policy.action.value == 1
    assert policy.action.name == "flap"
    assert policy.action.is_oneshot
