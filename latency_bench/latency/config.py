from __future__ import annotations

from typing import Any, Mapping


_LOGGED_LATENCY_TYPES_BY_METHOD = {
    "measured": "measured",
    "zero": "zero",
    "fixed": "fixed",
    "normal": "normal_distribution",
    "uniform": "uniform_distribution",
    "iid": "profile_sample",
    "temporal": "profile_sample",
}

SEEDED_STOCHASTIC_METHODS = frozenset({"normal", "uniform", "iid", "temporal"})
BATCHED_SIMULATED_METHODS = frozenset({"zero", "fixed"}) | SEEDED_STOCHASTIC_METHODS


def resolve_latency_method(latency_cfg: Mapping[str, Any]) -> str:
    return latency_cfg["method"]


def latency_type_from_config(latency_cfg: Mapping[str, Any]) -> str:
    method = resolve_latency_method(latency_cfg)
    if latency_cfg.get("per_episode") and method in {"normal", "uniform"}:
        return f"mixed_{_LOGGED_LATENCY_TYPES_BY_METHOD[method]}"
    return _LOGGED_LATENCY_TYPES_BY_METHOD[method]
