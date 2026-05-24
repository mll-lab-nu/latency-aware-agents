from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.rollout_data.prompts import BASE_PROMPTS


def load_latency_prompt_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_starvla_prompt(
    *,
    env_name: str,
    profile: str,
    observation_metadata: dict[str, Any],
    latency_prompt_map: dict[str, Any] | None,
) -> str:
    return {
        "frame_stack": lambda: BASE_PROMPTS[str(env_name)],
        "single_latency": lambda: latency_prompt_map[str(observation_metadata["latency_raw_frames"])]["prompt"],
    }[profile]()
