from __future__ import annotations

from typing import Any, Mapping, Sequence

from datasets import Features, Image as HFImage, Sequence as HFSequence, Value

from latency_bench.data.action_layouts import ActionSpec


SCHEMA_VERSION = 1
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
    "reward",
)


def rollout_features(action_spec: ActionSpec) -> Features:
    """Build Hugging Face dataset features for canonical rollout rows.

    Args:
        action_spec: Environment action spec defining action-specific fields.

    Returns:
        Hugging Face ``Features`` for one environment's rollout table.
    """
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
        "reward": Value("float64"),
    }
    if "action_id" in action_spec.row_fields:
        feature_map["action_id"] = Value("int64")
    if "action_tuple" in action_spec.row_fields:
        feature_map["action_tuple"] = HFSequence(Value("int64"))
    feature_map["action"] = HFSequence(Value("float32"))
    feature_map["action_text"] = Value("string")
    return Features(feature_map)


def empty_records(action_spec: ActionSpec) -> dict[str, list[Any]]:
    """Build empty column arrays matching the rollout features.

    Args:
        action_spec: Environment action spec defining action-specific fields.

    Returns:
        Empty column mapping suitable for ``Dataset.from_dict``.
    """
    return {name: [] for name in rollout_features(action_spec)}


def sort_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort rollout rows by episode and decision step.

    Args:
        rows: Canonical rollout row mappings.

    Returns:
        New list ordered by ``episode_idx`` then ``decision_step``.
    """
    return sorted([dict(row) for row in rows], key=lambda row: (int(row["episode_idx"]), int(row["decision_step"])))
