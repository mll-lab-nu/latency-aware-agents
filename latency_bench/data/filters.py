"""Rollout dataset filters for environment-specific clean presets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from latency_bench.data.action_layouts import action_spec_for_env


FILTER_PRESET_CLEAN_V1 = "clean_v1"
FLAPPY_CONTEXT_WINDOW = 2
FLAPPY_BACKGROUND_KEEP_PROB = 0.2
DEMON_CONTEXT_WINDOW = 8
DEMON_SURVIVAL_ANCHOR_STRIDE = 16
DEMON_BACKGROUND_KEEP_PROB = 0.5


def episode_return(rows: Sequence[Mapping[str, Any]], reward_field: str) -> float:
    """Sum one episode's reward column.

    Args:
        rows: Rows from one complete episode.
        reward_field: Environment-specific reward column name.

    Returns:
        Episode return using the requested reward semantics.
    """
    return float(sum(row[reward_field] for row in rows))


def episode_summary(env_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the report fields used by online rollout filtering.

    Args:
        env_name: Canonical benchmark environment name.
        rows: Rows from one complete episode.

    Returns:
        Episode length and env-specific return fields.
    """
    action_spec = action_spec_for_env(env_name)
    return {
        "episode_length": len(rows),
        action_spec.episode_return_field: episode_return(rows, action_spec.reward_field),
    }


def episode_passes_clean_v1(env_name: str, rows: Sequence[Mapping[str, Any]], *, max_steps_per_episode: int) -> bool:
    """Evaluate the clean_v1 episode-level gate.

    Args:
        env_name: Canonical benchmark environment name.
        rows: Rows from one complete episode.
        max_steps_per_episode: Export step cap used by Flappy's full-length gate.

    Returns:
        Whether this episode should count toward the requested dataset size.
    """
    summary = episode_summary(env_name, rows)
    return {
        "flappy": lambda: summary["episode_length"] >= max_steps_per_episode,
        "deadly_corridor": lambda: summary["episode_raw_return"] >= 2200.0,
        "demon_attack": lambda: summary["episode_clipped_return"] > 50.0,
    }[env_name]()


def _rows_near_steps(rows: Sequence[Mapping[str, Any]], center_steps: set[int], *, window: int) -> set[int]:
    """Find row indices within a decision-step window around event steps.

    Args:
        rows: Candidate episode rows.
        center_steps: Event decision steps.
        window: Inclusive distance around each event step.

    Returns:
        Row indices that belong to event context.
    """
    return {
        row_idx
        for row_idx, row in enumerate(rows)
        if any(abs(row["decision_step"] - center_step) <= window for center_step in center_steps)
    }


def _sample_background_rows(
    rows: Sequence[Mapping[str, Any]],
    kept_indices: set[int],
    *,
    keep_prob: float,
    seed: int,
) -> set[int]:
    """Sample non-critical background rows for train density control.

    Args:
        rows: Candidate episode rows.
        kept_indices: Indices already kept by event or anchor rules.
        keep_prob: Probability for each remaining background row.
        seed: Episode reset seed used for deterministic sampling.

    Returns:
        Additional row indices selected from the background.
    """
    rng = np.random.default_rng(seed)
    return {
        row_idx
        for row_idx, _row in enumerate(rows)
        if row_idx not in kept_indices and rng.random() < keep_prob
    }


def _copy_kept_rows(rows: Sequence[Mapping[str, Any]], kept_indices: set[int]) -> list[dict[str, Any]]:
    """Copy rows selected by a filter while preserving episode order.

    Args:
        rows: Candidate episode rows.
        kept_indices: Selected row indices.

    Returns:
        Row copies in original order.
    """
    return [dict(row) for row_idx, row in enumerate(rows) if row_idx in kept_indices]


def _filter_flappy_train_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Keep Flappy FLAP events, local context, and sampled far NOOP rows.

    Args:
        rows: Accepted Flappy training episode rows.
        seed: Episode reset seed used for deterministic background sampling.

    Returns:
        Filtered row copies.
    """
    flap_steps = {row["decision_step"] for row in rows if row["action_text"] == "FLAP"}
    kept_indices = _rows_near_steps(rows, flap_steps, window=FLAPPY_CONTEXT_WINDOW)
    kept_indices |= _sample_background_rows(rows, kept_indices, keep_prob=FLAPPY_BACKGROUND_KEEP_PROB, seed=seed)
    return _copy_kept_rows(rows, kept_indices)


def _filter_demon_train_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Keep Demon Attack reward events, survival anchors, context, and background.

    Args:
        rows: Accepted Demon Attack training episode rows.
        seed: Episode reset seed used for deterministic background sampling.

    Returns:
        Filtered row copies.
    """
    reward_steps = {row["decision_step"] for row in rows if row["clipped_reward"] > 0.0}
    kept_indices = _rows_near_steps(rows, reward_steps, window=DEMON_CONTEXT_WINDOW)
    kept_indices |= {
        row_idx
        for row_idx, row in enumerate(rows)
        if row["decision_step"] % DEMON_SURVIVAL_ANCHOR_STRIDE == 0
    }
    kept_indices |= _sample_background_rows(rows, kept_indices, keep_prob=DEMON_BACKGROUND_KEEP_PROB, seed=seed)
    return _copy_kept_rows(rows, kept_indices)


def _keep_all_train_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Copy accepted train rows when an environment has no row-level filter.

    Args:
        rows: Accepted training episode rows.
        seed: Episode reset seed, unused because all rows are retained.

    Returns:
        Row copies in original order.
    """
    del seed
    return [dict(row) for row in rows]


def filter_train_rows_clean_v1(env_name: str, rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Apply clean_v1 train-only row density filtering.

    Args:
        env_name: Canonical benchmark environment name.
        rows: Accepted episode rows from the train split.
        seed: Episode reset seed used for deterministic background sampling.

    Returns:
        Filtered row copies. Validation rows should not use this function.
    """
    return {
        "flappy": _filter_flappy_train_rows,
        "demon_attack": _filter_demon_train_rows,
        "deadly_corridor": _keep_all_train_rows,
    }[env_name](rows, seed=seed)

