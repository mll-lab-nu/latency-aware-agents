from __future__ import annotations

from collections.abc import Callable
from typing import Any

from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter


EnvAdapterBuilder = Callable[[dict[str, Any], Action], EnvAdapter]

_ENV_ADAPTER_BUILDERS: dict[str, EnvAdapterBuilder] = {}


def register_env_adapter(name: str) -> Callable[[EnvAdapterBuilder], EnvAdapterBuilder]:
    normalized_name = _normalize_env_name(name)

    def decorator(builder: EnvAdapterBuilder) -> EnvAdapterBuilder:
        if normalized_name in _ENV_ADAPTER_BUILDERS:
            raise ValueError(f"Env adapter already registered: {normalized_name}")
        _ENV_ADAPTER_BUILDERS[normalized_name] = builder
        return builder

    return decorator


def build_env_adapter(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    env_name = _normalize_env_name(env_cfg["name"])
    builder = _ENV_ADAPTER_BUILDERS.get(env_name)
    if builder is None:
        registered = ", ".join(get_registered_env_names()) or "<none>"
        raise ValueError(f"Unsupported env.name: {env_name}. Registered env adapters: {registered}")
    return builder(config, noop_action)


def get_registered_env_names() -> list[str]:
    return sorted(_ENV_ADAPTER_BUILDERS)


def _normalize_env_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("env.name cannot be empty")
    return normalized


@register_env_adapter("dummy")
def _build_dummy_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    experiment_cfg = config["experiment"]
    return DummyEnvAdapter(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(experiment_cfg.get("max_env_steps", 10000)),
        noop_action=noop_action,
    )

