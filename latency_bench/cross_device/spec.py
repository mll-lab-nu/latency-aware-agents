"""Protocol spec loading for cross-device experiments.

The spec defines the protocol rows once at top level; experiments only carry
their panel label and, for cross-device prediction, the replay pairs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_protocol_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Protocol spec must be a mapping: {spec_path}")
    return spec
