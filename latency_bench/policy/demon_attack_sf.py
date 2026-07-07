from __future__ import annotations

from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.factory import register_policy
from latency_bench.policy.sample_factory_common import SampleFactoryIntegrationModule, SampleFactoryPolicyRunner


@register_policy("demon_attack_sf")
def _build_demon_attack_sf_policy(config: dict[str, Any], action_resolver: ActionResolver):
    from training.demon_attack_sf import integration

    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=SampleFactoryIntegrationModule(integration, "register_demon_attack_envs"),
        policy_type="demon_attack_sf",
    )
