from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latency_bench.data.prompts import BASE_PROMPTS


def load_latency_prompt_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_starvla_prompt(
    *,
    env_name: str,
    observation_metadata: dict[str, Any],
    latency_prompt_map: dict[str, Any] | None,
) -> str:
    if latency_prompt_map is not None:
        return latency_prompt_map[str(observation_metadata["latency_raw_frames"])]["prompt"]
    return BASE_PROMPTS[str(env_name)]
