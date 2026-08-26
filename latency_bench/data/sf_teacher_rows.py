"""Build canonical dataset rows from Sample Factory teacher rollout steps."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from latency_bench.data.action_layouts import ActionSpec, action_spec_for_env, encode_action_for_env
from latency_bench.data.prompts import render_teacher_prompt
from latency_bench.data.rollout_schema import CANONICAL_ROW_FIELDS
from latency_bench.integrations.sample_factory.actions import to_python_action


def teacher_dataset_row(
    *,
    env_name: str,
    spec: Mapping[str, Any],
    decision_step: int,
    image: str,
    raw_action: Any,
    info: Mapping[str, Any],
    reward: Any,
    env_fps: float,
    obs_fps: float,
    image_transform: str = "raw_rgb",
    action_spec: ActionSpec | None = None,
    base_prompt: str | None = None,
    state: Any | None = None,
) -> dict[str, Any]:
    """Build one canonical rollout row from a teacher decision.

    Args:
        env_name: Canonical benchmark environment name.
        spec: Episode spec containing index, split, and seed.
        decision_step: Zero-based policy decision step within the episode.
        image: Path to the saved pre-step image.
        raw_action: Raw teacher action for this decision.
        info: Environment step info after applying the action.
        reward: Reward returned by the environment step.
        env_fps: Raw environment FPS.
        obs_fps: Observation decision FPS.
        image_transform: Exported-image transform name used to render prompt
            context for transformed observations.
        action_spec: Explicit action layout for a generic Gymnasium task.
        base_prompt: Explicit prompt for a generic Gymnasium task.

    Returns:
        Canonical rollout row with environment-specific action fields.
    """
    latency_raw_frames = int(to_python_action(info["latency_raw_frames"]))
    latency_ms = float(to_python_action(info["latency_ms"]))
    resolved_action_spec = action_spec if action_spec is not None else action_spec_for_env(env_name)
    base_values = {
        "image": image,
        "env_name": str(env_name),
        "episode_idx": int(spec["episode_idx"]),
        "decision_step": int(decision_step),
        "split": str(spec["split"]),
        "seed": int(spec["seed"]),
        "prompt": render_teacher_prompt(
            env_name,
            latency_raw_frames=latency_raw_frames,
            latency_ms=latency_ms,
            env_fps=env_fps,
            obs_fps=obs_fps,
            image_transform=image_transform,
            base_prompt=base_prompt,
        ),
        "latency_raw_frames": latency_raw_frames,
        "latency_ms": latency_ms,
        resolved_action_spec.reward_field: float(reward),
    }
    row = {field: base_values[field] for field in CANONICAL_ROW_FIELDS}
    row[resolved_action_spec.reward_field] = base_values[resolved_action_spec.reward_field]
    row.update(encode_action_for_env(env_name, raw_action, action_spec=resolved_action_spec))
    if state is not None:
        row["state"] = np.asarray(state, dtype=np.float32).reshape(-1).tolist()
    return row
