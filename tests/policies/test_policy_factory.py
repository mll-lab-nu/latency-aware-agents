from __future__ import annotations

import sys
from types import ModuleType

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


def test_official_gr00t_policy_type_uses_the_separate_policy_builder(monkeypatch):
    import latency_bench.policy.official_gr00t as official_gr00t

    sentinel = object()
    built = []

    def build_official_gr00t_policy(config, action_resolver):
        built.append((config, action_resolver))
        return sentinel

    monkeypatch.setattr(official_gr00t, "build_official_gr00t_policy", build_official_gr00t_policy)
    config = _config(
        {
            "type": "official_gr00t",
            "model_path": "checkpoint",
            "embodiment_tag": "NEW_EMBODIMENT",
            "device": "cuda:0",
        }
    )
    action_resolver = ActionResolver(config["env"])

    assert build_policy(config, action_resolver) is sentinel
    assert built == [(config, action_resolver)]


def test_unknown_policy_type_is_rejected():
    config = _config({"type": "unknown"})

    with pytest.raises(ValueError, match="Unsupported policy.type: unknown"):
        build_policy(config, ActionResolver(config["env"]))


@pytest.mark.parametrize("policy_type", ["flappy_sf", "demon_attack_sf", "deadly_corridor_sf"])
def test_sample_factory_policy_types_are_supported(monkeypatch, policy_type):
    policy_module = ModuleType("latency_bench.policy.sample_factory")
    built = []
    sentinel = object()

    def build_sample_factory_policy(config, action_resolver, *, policy_type):
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
