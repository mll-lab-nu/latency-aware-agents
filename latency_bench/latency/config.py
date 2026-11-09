from __future__ import annotations

from typing import Any, Mapping


_LOGGED_LATENCY_TYPES_BY_METHOD = {
    "zero": "zero",
    "fixed": "fixed",
    "normal": "normal_distribution",
    "uniform": "uniform_distribution",
    "iid": "profile_sample",
    "temporal": "profile_sample",
    "hmm": "hmm",
    "block": "block",
    "ar1": "ar1",
    "arma": "arma",
}

STATEFUL_PROFILE_METHODS = frozenset({"temporal", "hmm", "block", "ar1", "arma"})
PROFILE_METHODS = frozenset({"iid"}) | STATEFUL_PROFILE_METHODS
SEEDED_STOCHASTIC_METHODS = frozenset({"normal", "uniform"}) | PROFILE_METHODS
BATCHED_SIMULATED_METHODS = frozenset({"zero", "fixed"}) | SEEDED_STOCHASTIC_METHODS


def latency_type_from_config(latency_cfg: Mapping[str, Any]) -> str:
    method = latency_cfg["method"]
    if latency_cfg.get("per_episode") and method in {"normal", "uniform"}:
        return f"mixed_{_LOGGED_LATENCY_TYPES_BY_METHOD[method]}"
    return _LOGGED_LATENCY_TYPES_BY_METHOD[method]
