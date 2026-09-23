from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    base_ref = current.get("__base__")
    if not base_ref:
        return {k: v for k, v in current.items() if k != "__base__"}

    if isinstance(base_ref, (str, Path)):
        base_paths = [base_ref]
    elif isinstance(base_ref, list):
        base_paths = base_ref
    else:
        raise ValueError("__base__ must be a path string or list of path strings")

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, load_config(base_path))

    return merge_dicts(merged, current)
