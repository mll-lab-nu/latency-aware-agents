from __future__ import annotations

from importlib import import_module
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.sample_factory_common import SampleFactoryIntegrationModule, SampleFactoryPolicyRunner

_INTEGRATIONS = {
    "flappy_sf": ("training.flappy_sf.integration", "register_flappy_envs"),
    "demon_attack_sf": ("training.demon_attack_sf.integration", "register_demon_attack_envs"),
    "deadly_corridor_sf": (
        "training.deadly_corridor_sf.integration",
        "register_deadly_corridor_components",
    ),
}


def build_sample_factory_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    *,
    policy_type: str,
) -> SampleFactoryPolicyRunner:
    integration_module, register_components = _INTEGRATIONS[policy_type]
    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=SampleFactoryIntegrationModule(import_module(integration_module), register_components),
        policy_type=policy_type,
    )
