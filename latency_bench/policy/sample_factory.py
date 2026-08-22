from __future__ import annotations

from importlib import import_module
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.integrations.sample_factory.runtime import SampleFactoryIntegrationModule
from latency_bench.policy.sample_factory_common import SampleFactoryPolicyRunner

_INTEGRATIONS = {
    "flappy_sf": ("training.flappy_sf.integration", "register_flappy_envs"),
    "demon_attack_sf": ("training.demon_attack_sf.integration", "register_demon_attack_envs"),
    "deadly_corridor_sf": (
        "training.deadly_corridor_sf.integration",
        "register_deadly_corridor_components",
    ),
    "gymnasium_sf": (
        "training.gymnasium_sf.integration",
        "register_gymnasium_envs",
    ),
}


class _GymnasiumPolicyActionResolver:
    def __init__(self, action_resolver: ActionResolver, action_order: list[Any]):
        self._action_resolver = action_resolver
        self._action_order = action_order

    def resolve(self, policy_action_id: Any):
        return self._action_resolver.resolve(self._action_order[policy_action_id])


def build_sample_factory_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    *,
    policy_type: str,
) -> SampleFactoryPolicyRunner:
    integration_module, register_components = _INTEGRATIONS[policy_type]
    policy_action_resolver = (
        _GymnasiumPolicyActionResolver(action_resolver, config["env"]["action_order"])
        if policy_type == "gymnasium_sf" and "action_order" in config["env"]
        else action_resolver
    )
    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=policy_action_resolver,
        integration_module=SampleFactoryIntegrationModule(import_module(integration_module), register_components),
        policy_type=policy_type,
    )
