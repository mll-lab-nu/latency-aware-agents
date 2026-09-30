from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latency_bench.data.prompts import BASE_PROMPTS, render_teacher_prompt


def _apply_prompt_mode(prompt: str, prompt_mode: str | None) -> str:
    mode = str(prompt_mode or "").strip().lower()
    if mode in {"", "default", "none", "raw"}:
        return prompt
    if mode == "latency_neutral":
        marker = " Current action latency is "
        head, found, _tail = str(prompt).partition(marker)
        return head.rstrip() if found else str(prompt)
    raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")


def load_latency_prompt_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_starvla_prompt(
    *,
    env_name: str,
    observation_metadata: dict[str, Any],
    latency_prompt_map: dict[str, Any] | None,
    base_prompt: str | None = None,
    latency_prompt_key: int | str | None = None,
    prompt_mode: str | None = None,
) -> str:
    if latency_prompt_map is not None:
        key = observation_metadata.get("latency_prompt_key", latency_prompt_key)
        if key is None:
            key = observation_metadata["latency_raw_frames"]
        return _apply_prompt_mode(latency_prompt_map[str(key)]["prompt"], prompt_mode)

    raw_frames = observation_metadata.get("latency_raw_frames")
    if raw_frames is not None:
        prompt = render_teacher_prompt(
            env_name,
            latency_raw_frames=int(raw_frames),
            latency_ms=float(observation_metadata["latency_ms"]),
            env_fps=float(observation_metadata["env_fps"]),
            obs_fps=float(observation_metadata["obs_fps"]),
            image_transform=str(observation_metadata.get("image_transform", "raw_rgb")),
            base_prompt=base_prompt,
        )
        return _apply_prompt_mode(prompt, prompt_mode)
    prompt = BASE_PROMPTS[str(env_name)] if base_prompt is None else base_prompt
    return _apply_prompt_mode(prompt, prompt_mode)
