from __future__ import annotations

import sys
from types import ModuleType

import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation
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


def test_unknown_policy_type_is_rejected():
    config = _config({"type": "unknown"})

    with pytest.raises(ValueError, match="Unsupported policy.type: unknown"):
        build_policy(config, ActionResolver(config["env"]))


def test_build_policy_accepts_python_factory():
    config = _config({"type": "user_policy", "action": "flap"})

    def make_policy(config, action_resolver):
        return ConstantPolicyRunner(action_resolver.resolve(config["policy"]["action"]))

    policy = build_policy(config, ActionResolver(config["env"]), factory=make_policy)
    result = policy.predict(Observation(data=None, env_step=0, sim_time_ms=0))
    assert result.action.value == 1


@pytest.mark.parametrize("reference,error,match", [
    ("outside_policy", ValueError, "module:function"),
    ("outside_policy:not_callable", TypeError, "not callable"),
])
def test_custom_policy_rejects_invalid_factory_reference(monkeypatch, reference, error, match):
    """Reject malformed references and non-callable factory targets."""
    module = ModuleType("outside_policy")
    module.not_callable = 7
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _config({"type": "custom", "factory": reference})
    with pytest.raises(error, match=match):
        build_policy(config, ActionResolver(config["env"]))


def test_sample_factory_policy_type_is_supported(monkeypatch):
    policy_type = "sample_factory"
    policy_module = ModuleType("latency_bench.policy.sample_factory")
    built = []
    sentinel = object()

    def build_sample_factory_policy(config, action_resolver):
        built.append((config, action_resolver, policy_type))
        return sentinel

    policy_module.build_sample_factory_policy = build_sample_factory_policy
    monkeypatch.setitem(sys.modules, policy_module.__name__, policy_module)
    config = _config({"type": policy_type})
    action_resolver = ActionResolver(config["env"])

    policy = build_policy(config, action_resolver)

    assert policy is sentinel
    assert built == [(config, action_resolver, policy_type)]


def test_flappy_constant_flap_resolves_to_env_action_value():
    config = _config({"type": "constant", "action": "flap"})

    policy = build_policy(config, ActionResolver(config["env"]))

    assert policy.action.value == 1
    assert policy.action.name == "flap"
    assert policy.action.is_oneshot
