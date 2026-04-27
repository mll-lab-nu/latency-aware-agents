from __future__ import annotations

from statistics import mean
from typing import Any

from latency_bench.core.types import ActionEvent, EpisodeMetrics, StepRecord


def compute_episode_metrics(
    episode_id: int,
    step_records: list[StepRecord],
    action_events: list[ActionEvent],
    latency_values_ms: list[float],
    metadata: dict[str, Any] | None = None,
) -> EpisodeMetrics:
    episode_return = sum(record.reward for record in step_records)
    survival_steps = len(step_records)
    game_score = _extract_game_score(step_records)
    latencies = [float(value) for value in latency_values_ms]
    return EpisodeMetrics(
        episode_id=episode_id,
        episode_return=episode_return,
        survival_steps=survival_steps,
        game_score=game_score,
        mean_latency_ms=mean(latencies) if latencies else None,
        p90_latency_ms=_percentile(latencies, 90.0) if latencies else None,
        p99_latency_ms=_percentile(latencies, 99.0) if latencies else None,
        num_actions=len(action_events),
        num_dropped_actions=sum(1 for event in action_events if event.dropped),
        num_invalid_actions=sum(1 for record in step_records if record.info.get("invalid_action")),
        metadata=metadata or {},
    )


def _extract_game_score(step_records: list[StepRecord]) -> float | None:
    if not step_records:
        return None
    info = step_records[-1].info
    if "game_score" in info:
        return float(info["game_score"])
    if "score" in info:
        return float(info["score"])
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

