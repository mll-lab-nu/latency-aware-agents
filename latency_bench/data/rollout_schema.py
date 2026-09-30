from __future__ import annotations

from typing import Any, Mapping, Sequence

from datasets import Features, Image as HFImage, Sequence as HFSequence, Value

from latency_bench.data.action_layouts import ActionSpec


SCHEMA_VERSION = 2
CANONICAL_ROW_FIELDS = (
    "image",
    "env_name",
    "episode_idx",
    "decision_step",
    "split",
    "seed",
    "prompt",
    "latency_raw_frames",
    "latency_ms",
)


def rollout_features(
    action_spec: ActionSpec,
    *,
    include_context_images: bool = False,
    include_state: bool = False,
    include_latency_trace: bool = False,
) -> Features:
    """Build the canonical rollout schema, including the requested optional columns."""
    feature_map: dict[str, Any] = {
        "image": HFImage(),
        "env_name": Value("string"),
        "episode_idx": Value("int64"),
        "decision_step": Value("int64"),
        "split": Value("string"),
        "seed": Value("int64"),
        "prompt": Value("string"),
        "latency_raw_frames": Value("int64"),
        "latency_ms": Value("float64"),
        action_spec.reward_field: Value("float64"),
    }
    if "action_id" in action_spec.row_fields:
        feature_map["action_id"] = Value("int64")
    if "action" in action_spec.row_fields:
        feature_map["action"] = HFSequence(Value("float32"))
    feature_map["action_text"] = Value("string")
    if include_state:
        feature_map["state"] = HFSequence(Value("float32"))
    if include_latency_trace:
        feature_map.update(
            {
                "command_admitted": Value("bool"),
                "observation_dropped": Value("bool"),
                "applied_action": HFSequence(Value("float32")),
                "applied_source_raw_frame": Value("int64"),
                "applied_source_obs_id": Value("int64"),
                "raw_done": Value("bool"),
            }
        )
    if include_context_images:
        feature_map["context_images"] = HFSequence(HFImage())
    return Features(feature_map)


def sort_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort rollout rows by episode and decision step.

    Args:
        rows: Canonical rollout row mappings.

    Returns:
        New list ordered by ``episode_idx`` then ``decision_step``.
    """
    return sorted([dict(row) for row in rows], key=lambda row: (row["episode_idx"], row["decision_step"]))
