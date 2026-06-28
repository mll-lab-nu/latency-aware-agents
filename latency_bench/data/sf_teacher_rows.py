"""Build canonical dataset rows from Sample Factory teacher rollout steps."""

from __future__ import annotations

from typing import Any, Mapping

from latency_bench.data.action_layouts import action_spec_for_env, encode_action_for_env
from latency_bench.data.prompts import render_teacher_prompt
from latency_bench.data.rollout_schema import CANONICAL_ROW_FIELDS
from latency_bench.policy.sample_factory_common import _to_python_action


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

    Returns:
        Canonical rollout row with environment-specific action fields.
    """
    latency_raw_frames = int(_to_python_action(info["latency_raw_frames"]))
    latency_ms = float(_to_python_action(info["latency_ms"]))
    action_spec = action_spec_for_env(env_name)
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
        ),
        "latency_raw_frames": latency_raw_frames,
        "latency_ms": latency_ms,
        action_spec.reward_field: float(reward),
    }
    row = {field: base_values[field] for field in CANONICAL_ROW_FIELDS}
    row[action_spec.reward_field] = base_values[action_spec.reward_field]
    row.update(encode_action_for_env(env_name, raw_action))
    return row
