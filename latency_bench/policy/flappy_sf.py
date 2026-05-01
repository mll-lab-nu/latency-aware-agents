from __future__ import annotations

from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.factory import register_policy
from latency_bench.policy.sample_factory_common import SampleFactoryPolicyRunner


@register_policy("flappy_sf")
def _build_flappy_sf_policy(config: dict[str, Any], action_resolver: ActionResolver):
    from training.flappy_sf import integration

    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=action_resolver,
        integration_module=_FlappyIntegrationModule(integration),
        policy_type="flappy_sf",
    )


class _FlappyIntegrationModule:
    def __init__(self, module):
        self.module = module

    def register_components(self):
        self.module.register_flappy_envs()

    def build_sf_cfg_from_config(self, config: dict[str, Any]):
        return self.module.parse_cfg(self.module.build_cli_args_from_config(config))

    def make_reference_env(self, cfg):
        return self.module.make_reference_env(cfg)
