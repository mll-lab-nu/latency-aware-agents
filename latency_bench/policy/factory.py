from __future__ import annotations

from collections.abc import Callable
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_factory
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner, RandomPolicyRunner

_SAMPLE_FACTORY_POLICY_TYPES = {
    "flappy_sf",
    "demon_attack_sf",
    "deadly_corridor_sf",
    "gymnasium_sf",
}


def build_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    *,
    factory: Callable[[dict[str, Any], ActionResolver], PolicyRunner] | None = None,
) -> PolicyRunner:
    """Construct a policy; an explicit Python factory overrides config selection."""
    if factory is not None:
        return factory(config, action_resolver)
    policy_cfg = config["policy"]
    policy_type = policy_cfg["type"]
    if policy_type == "coding_agent":
        from latency_bench.policy.codex import build_codex_policy
        from latency_bench.policy.claude_code import build_claude_code_policy

        return {"codex": build_codex_policy, "claude_code": build_claude_code_policy}[
            policy_cfg["provider"]
        ](config, action_resolver)
    if policy_type == "custom":
        return load_factory(policy_cfg["factory"])(config, action_resolver)
    if policy_type == "random":
        return _build_random_policy(config, action_resolver)
    if policy_type == "constant":
        return _build_constant_policy(config, action_resolver)
    if policy_type in _SAMPLE_FACTORY_POLICY_TYPES:
        from latency_bench.policy.sample_factory import build_sample_factory_policy

        return build_sample_factory_policy(config, action_resolver, policy_type=policy_type)
    if policy_type == "starvla":
        from latency_bench.policy.starvla import build_starvla_policy

        return build_starvla_policy(config, action_resolver)
    raise ValueError(f"Unsupported policy.type: {policy_type}")


def _build_random_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
    actions = action_resolver.resolve_many(action_refs)
    return RandomPolicyRunner(actions=actions, seed=config.get("experiment", {}).get("seed"))


def _build_constant_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    default_action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
    if "action" in policy_cfg:
        action_ref = policy_cfg["action"]
    else:
        if not default_action_refs:
            raise ValueError("policy.action is required when no default actions are available")
        action_ref = default_action_refs[0]
    return ConstantPolicyRunner(action=action_resolver.resolve(action_ref))
