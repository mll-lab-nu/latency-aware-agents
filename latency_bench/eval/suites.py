"""Read named evaluation conditions using the ordinary latency configuration."""

from __future__ import annotations

from typing import Any
from pathlib import Path


def has_eval_suites(config: dict[str, Any]) -> bool:
    return bool(parse_eval_suites(config))


def parse_eval_suites(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve simulated evaluation conditions; other modes run their own timing."""
    if config["executor"]["mode"] != "simulated":
        return []
    suites = config["evaluation"].get("suites", [])
    names = set()
    for suite in suites:
        name = suite["name"]
        if Path(name).name != name or name in {".", "..", ""}:
            raise ValueError(
                f"Evaluation suite name must be one path segment: {name!r}"
            )
        if name in names:
            raise ValueError(f"Duplicate evaluation suite name: {name}")
        names.add(name)
    return suites
