from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def save_profile_manifest(profile: Mapping[str, Any], path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(dict(profile), f, indent=2, sort_keys=True)
        f.write("\n")


def load_profile_manifest(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_latency_distribution(distribution: Mapping[str, Any], path: str | Path) -> None:
    distribution_path = Path(path)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    with distribution_path.open("w", encoding="utf-8") as f:
        json.dump(dict(distribution), f, indent=2, sort_keys=True)
        f.write("\n")
