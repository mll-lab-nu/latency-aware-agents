from __future__ import annotations

from collections.abc import Callable
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner, RandomPolicyRunner


PolicyBuilder = Callable[[dict[str, Any], ActionResolver], PolicyRunner]

_POLICY_BUILDERS: dict[str, PolicyBuilder] = {}


def register_policy(policy_type: str) -> Callable[[PolicyBuilder], PolicyBuilder]:
    normalized_type = _normalize_policy_type(policy_type)

    def decorator(builder: PolicyBuilder) -> PolicyBuilder:
        if normalized_type in _POLICY_BUILDERS:
            raise ValueError(f"Policy already registered: {normalized_type}")
        _POLICY_BUILDERS[normalized_type] = builder
        return builder

    return decorator


def build_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    policy_type = _normalize_policy_type(policy_cfg["type"])
    builder = _POLICY_BUILDERS.get(policy_type)
    if builder is None:
        _lazy_import_policy(policy_type)
        builder = _POLICY_BUILDERS.get(policy_type)
    if builder is None:
        registered = ", ".join(get_registered_policy_types()) or "<none>"
        raise ValueError(f"Unsupported policy.type: {policy_type}. Registered policies: {registered}")
    return builder(config, action_resolver)


def get_registered_policy_types() -> list[str]:
    return sorted(_POLICY_BUILDERS)


def _normalize_policy_type(policy_type: str) -> str:
    normalized = str(policy_type).strip().lower()
    if not normalized:
        raise ValueError("policy.type cannot be empty")
    return normalized


def _lazy_import_policy(policy_type: str) -> None:
    if policy_type == "flappy_openvla":
        from latency_bench.policy import flappy_openvla  # noqa: F401
    if policy_type == "demon_attack_openvla":
        from latency_bench.policy import demon_attack_openvla  # noqa: F401
    if policy_type == "deadly_corridor_openvla":
        from latency_bench.policy import deadly_corridor_openvla  # noqa: F401
    if policy_type == "flappy_sf":
        from latency_bench.policy import flappy_sf  # noqa: F401
    if policy_type == "demon_attack_sf":
        from latency_bench.policy import demon_attack_sf  # noqa: F401
    if policy_type == "deadly_corridor_sf":
        from latency_bench.policy import deadly_corridor_sf  # noqa: F401


@register_policy("random")
def _build_random_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
    actions = action_resolver.resolve_many(action_refs)
    return RandomPolicyRunner(actions=actions, seed=config.get("experiment", {}).get("seed"))


@register_policy("constant")
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
