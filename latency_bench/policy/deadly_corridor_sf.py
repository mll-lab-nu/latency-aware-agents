from __future__ import annotations

from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.factory import register_policy
from latency_bench.policy.sample_factory_common import SampleFactoryIntegrationModule, SampleFactoryPolicyRunner


@register_policy("deadly_corridor_sf")
def _build_deadly_corridor_sf_policy(config: dict[str, Any], action_resolver: ActionResolver):
    from training.deadly_corridor_sf import integration

    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=SampleFactoryIntegrationModule(integration, "register_deadly_corridor_components"),
        policy_type="deadly_corridor_sf",
    )
