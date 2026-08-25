from __future__ import annotations

from typing import Any, Mapping, Sequence

from datasets import Features, Image as HFImage, Sequence as HFSequence, Value

from latency_bench.data.action_layouts import ActionSpec


SCHEMA_VERSION = 3
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


def _rollout_feature_map(action_spec: ActionSpec, *, include_state: bool) -> dict[str, Any]:
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
    if "action_tuple" in action_spec.row_fields:
        feature_map["action_tuple"] = HFSequence(Value("int64"))
    if "action" in action_spec.row_fields:
        feature_map["action"] = HFSequence(Value("float32"))
    if include_state:
        feature_map["state"] = HFSequence(Value("float32"))
    feature_map["action_text"] = Value("string")
    return feature_map


def rollout_features(action_spec: ActionSpec, *, include_state: bool = False) -> Features:
    """Build Hugging Face dataset features for canonical rollout rows.

    Args:
        action_spec: Environment action spec defining action-specific fields.

    Returns:
        Hugging Face ``Features`` for one environment's rollout table.
    """
    return Features(_rollout_feature_map(action_spec, include_state=include_state))


def rollout_features_with_context_images(
    action_spec: ActionSpec, *, include_state: bool = False
) -> Features:
    """Build rollout features for rows that carry historical context images.

    Args:
        action_spec: Environment action spec defining action-specific fields.

    Returns:
        Hugging Face ``Features`` including ``context_images``.
    """
    feature_map = _rollout_feature_map(action_spec, include_state=include_state)
    feature_map["context_images"] = HFSequence(HFImage())
    return Features(feature_map)


def empty_records(action_spec: ActionSpec, *, include_state: bool = False) -> dict[str, list[Any]]:
    """Build empty column arrays matching the rollout features.

    Args:
        action_spec: Environment action spec defining action-specific fields.

    Returns:
        Empty column mapping suitable for ``Dataset.from_dict``.
    """
    return {name: [] for name in rollout_features(action_spec, include_state=include_state)}


def empty_records_with_context_images(
    action_spec: ActionSpec,
    include_context_images: bool,
    *,
    include_state: bool = False,
) -> dict[str, list[Any]]:
    """Build empty column arrays for single-image or context-image rollout features.

    Args:
        action_spec: Environment action spec defining action-specific fields.
        include_context_images: Whether to include the ``context_images`` feature.

    Returns:
        Empty column mapping suitable for ``Dataset.from_dict``.
    """
    features = (
        rollout_features_with_context_images(action_spec, include_state=include_state)
        if include_context_images
        else rollout_features(action_spec, include_state=include_state)
    )
    return {name: [] for name in features}


def sort_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort rollout rows by episode and decision step.

    Args:
        rows: Canonical rollout row mappings.

    Returns:
        New list ordered by ``episode_idx`` then ``decision_step``.
    """
    return sorted([dict(row) for row in rows], key=lambda row: (int(row["episode_idx"]), int(row["decision_step"])))
