"""Small row filters for canonical rollout datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping


def filter_by_latency_raw_frames(rows: Iterable[Mapping[str, Any]], allowed_latencies: set[int]) -> list[dict[str, Any]]:
    """Keep rows whose raw-frame latency is in the allowed set.

    Args:
        rows: Canonical rollout rows.
        allowed_latencies: Accepted ``latency_raw_frames`` values.

    Returns:
        Filtered row copies.
    """
    return [dict(row) for row in rows if int(row["latency_raw_frames"]) in allowed_latencies]


def filter_by_episode_length(rows: Iterable[Mapping[str, Any]], *, min_length: int) -> list[dict[str, Any]]:
    """Keep complete episodes with at least ``min_length`` rows.

    Args:
        rows: Canonical rollout rows.
        min_length: Minimum number of decision rows per episode.

    Returns:
        Rows from episodes satisfying the length threshold.
    """
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_idx"]), []).append(row)
    return [dict(row) for episode_rows in grouped.values() if len(episode_rows) >= int(min_length) for row in episode_rows]


def filter_by_min_return(rows: Iterable[Mapping[str, Any]], *, min_return: float) -> list[dict[str, Any]]:
    """Keep complete episodes whose total reward reaches ``min_return``.

    Args:
        rows: Canonical rollout rows.
        min_return: Minimum episode reward sum.

    Returns:
        Rows from episodes satisfying the return threshold.
    """
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_idx"]), []).append(row)
    return [
        dict(row)
        for episode_rows in grouped.values()
        if sum(float(row["reward"]) for row in episode_rows) >= float(min_return)
        for row in episode_rows
    ]
