from __future__ import annotations

from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.factory import register_policy
from latency_bench.policy.sample_factory_common import SampleFactoryIntegrationModule, SampleFactoryPolicyRunner


@register_policy("flappy_sf")
def _build_flappy_sf_policy(config: dict[str, Any], action_resolver: ActionResolver):
    from training.flappy_sf import integration

    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=SampleFactoryIntegrationModule(integration, "register_flappy_envs"),
        policy_type="flappy_sf",
    )
