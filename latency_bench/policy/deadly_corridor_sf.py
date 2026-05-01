from __future__ import annotations

from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.factory import register_policy
from latency_bench.policy.sample_factory_common import SampleFactoryPolicyRunner


@register_policy("deadly_corridor_sf")
def _build_deadly_corridor_sf_policy(config: dict[str, Any], action_resolver: ActionResolver):
    from training.deadly_corridor_sf import integration

    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=_DeadlyCorridorIntegrationModule(integration),
        policy_type="deadly_corridor_sf",
    )


class _DeadlyCorridorIntegrationModule:
    def __init__(self, module):
        self.module = module

    def register_components(self):
        self.module.register_deadly_corridor_components()

    def build_sf_cfg_from_config(self, config: dict[str, Any]):
        return self.module.parse_cfg(self.module.build_cli_args_from_config(config))

    def make_reference_env(self, cfg):
        return self.module.make_reference_env(cfg)
