from __future__ import annotations

import math
from statistics import mean
from typing import Any

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.core.types import ActionEvent, EpisodeMetrics, StepRecord


def compute_episode_metrics(
    episode_id: int,
    step_records: list[StepRecord],
    action_events: list[ActionEvent],
    latency_values_ms: list[float],
    metadata: dict[str, Any] | None = None,
    frame_ms: float | None = None,
) -> EpisodeMetrics:
    _finalize_action_events(action_events, frame_ms=frame_ms)
    episode_return_env = sum(record.reward for record in step_records)
    survival_steps = len(step_records)
    game_score = _extract_game_score(step_records)
    return_raw = _extract_explicit_return(step_records, _RAW_RETURN_KEYS)
    return_clipped = _extract_explicit_return(step_records, _CLIPPED_RETURN_KEYS)
    latencies = [float(value) for value in latency_values_ms]
    return EpisodeMetrics(
        episode_id=episode_id,
        episode_return=episode_return_env,
        survival_steps=survival_steps,
        episode_return_env=episode_return_env,
        return_raw=return_raw,
        return_clipped=return_clipped,
        game_score=game_score,
        mean_latency_ms=mean(latencies) if latencies else None,
        p90_latency_ms=_percentile(latencies, 90.0) if latencies else None,
        p99_latency_ms=_percentile(latencies, 99.0) if latencies else None,
        num_actions=len(action_events),
        num_dropped_actions=sum(1 for event in action_events if event.dropped),
        num_invalid_actions=sum(1 for record in step_records if record.info.get("invalid_action")),
        metadata=metadata or {},
    )


PROFILE_METADATA_KEYS = (
    "profile_id",
    "profile_type",
    "latency_scope",
    "target_device",
    "profile_path",
    "sample_index",
    "trace_index",
)

_RAW_RETURN_KEYS = (
    "return_raw",
    "raw_return",
    "episodic_raw_return",
    "episode/raw_return",
)
_CLIPPED_RETURN_KEYS = (
    "return_clipped",
    "clipped_return",
    "episodic_clipped_return",
    "episode/clipped_return",
)


def latency_type_from_source(source: Any, *, measured: bool = False) -> str:
    if measured:
        return "measured"
    name = source.__class__.__name__
    by_class = {
        "ZeroLatencySource": "zero",
        "FixedLatencySource": "fixed",
        "NormalDistributionLatencySource": "normal_distribution",
        "UniformDistributionLatencySource": "uniform_distribution",
        "TraceReplayLatencySource": "trace_replay",
        "EmpiricalSampleLatencySource": "empirical_sample",
        "EmpiricalTraceReplayLatencySource": "empirical_trace_replay",
        "MeasuredLatencySource": "measured",
    }
    return by_class.get(name, name)


def profile_metadata_from_source(source: Any) -> dict[str, Any]:
    metadata = {key: None for key in PROFILE_METADATA_KEYS}
    last_metadata = getattr(source, "last_metadata", None)
    if isinstance(last_metadata, dict):
        for key in PROFILE_METADATA_KEYS:
            if key in last_metadata:
                metadata[key] = last_metadata[key]
    for key in PROFILE_METADATA_KEYS:
        value = getattr(source, key, None)
        if value is not None and metadata[key] is None:
            metadata[key] = str(value) if key == "profile_path" else value
    return metadata


def apply_latency_raw_fields(
    *,
    event: ActionEvent,
    latency_type: str,
    frame_ms: float,
    latency_ms_sampled: float | None = None,
    latency_ms_measured: float | None = None,
    profile_metadata: dict[str, Any] | None = None,
) -> None:
    event.obs_step = event.obs_env_step
    event.action_emit_time_ms = event.obs_time_ms
    event.action_ready_time_ms = event.ready_time_ms
    event.latency_type = latency_type
    event.latency_ms_sampled = None if latency_ms_sampled is None else float(latency_ms_sampled)
    event.latency_ms_measured = None if latency_ms_measured is None else float(latency_ms_measured)
    event.effective_k = None if frame_ms <= 0 else float(event.latency_ms) / float(frame_ms)
    _apply_profile_metadata(event, profile_metadata or {})


def apply_latency_record_raw_fields(
    *,
    record: Any,
    latency_type: str,
    latency_ms_sampled: float | None = None,
    latency_ms_measured: float | None = None,
    profile_metadata: dict[str, Any] | None = None,
) -> None:
    record.latency_type = latency_type
    record.latency_ms_sampled = None if latency_ms_sampled is None else float(latency_ms_sampled)
    record.latency_ms_measured = None if latency_ms_measured is None else float(latency_ms_measured)
    _apply_profile_metadata(record, profile_metadata or {})


def episode_raw_fact_metadata(
    *,
    mode: str,
    episode_seed: int | None,
    env_fps: float,
    obs_fps: float,
    frame_ms: float,
    latency_type: str,
    latency_source: Any,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "episode_seed": episode_seed,
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "frame_ms": frame_ms,
        "latency_type": latency_type,
        **profile_metadata_from_source(latency_source),
    }


def active_action_step_info(
    *,
    applied_events: list[ActionEvent],
    env_step: int,
    scheduled_time_ms: float,
    frame_ms: float,
    latency_type: str,
    profile_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_event = applied_events[-1] if applied_events else None
    profile_metadata = profile_metadata or {}
    active_obs_frame = None
    if active_event is not None:
        active_obs_frame = (
            active_event.obs_raw_frame
            if active_event.obs_raw_frame is not None
            else active_event.obs_env_step
        )
    return {
        "active_action_id": None if active_event is None else active_event.action_id,
        "active_action_obs_step": None if active_event is None else active_obs_frame,
        "active_action_age_frames": (
            None if active_event is None else int(env_step) - int(active_obs_frame)
        ),
        "scheduled_time_ms": float(scheduled_time_ms),
        "frame_time_ms": float(frame_ms),
        "latency_type": latency_type,
        "profile_id": profile_metadata.get("profile_id"),
    }


def _extract_game_score(step_records: list[StepRecord]) -> float | None:
    if not step_records:
        return None
    info = step_records[-1].info
    if "game_score" in info:
        return float(info["game_score"])
    if "score" in info:
        return float(info["score"])
    return None


def _extract_explicit_return(step_records: list[StepRecord], keys: tuple[str, ...]) -> float | None:
    for record in reversed(step_records):
        info = record.info or {}
        value = _first_number(info, keys)
        if value is not None:
            return value
        extra_stats = info.get("episode_extra_stats")
        if isinstance(extra_stats, dict):
            value = _first_number(extra_stats, keys)
            if value is not None:
                return value
    return None


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return None


def _finalize_action_events(action_events: list[ActionEvent], *, frame_ms: float | None) -> None:
    for event in action_events:
        _backfill_action_event_raw_fields(event, frame_ms=frame_ms)
        event.obs_step = event.obs_env_step
        event.action_emit_time_ms = event.obs_time_ms
        event.action_ready_time_ms = event.ready_time_ms
        if event.applied_env_step is None and event.apply_raw_frame is not None:
            event.applied_env_step = int(event.apply_raw_frame)
        if event.apply_raw_frame is None and event.applied_env_step is not None:
            event.apply_raw_frame = int(event.applied_env_step)
        if event.applied_time_ms is None and event.apply_raw_frame is not None and frame_ms is not None:
            event.applied_time_ms = int(event.apply_raw_frame) * float(frame_ms)
        event.action_applied_step = (
            event.applied_env_step if event.applied_env_step is not None else event.apply_raw_frame
        )
        event.action_apply_time_ms = event.applied_time_ms
        obs_frame = event.obs_raw_frame if event.obs_raw_frame is not None else event.obs_env_step
        apply_frame = event.apply_raw_frame if event.apply_raw_frame is not None else event.applied_env_step
        if apply_frame is not None:
            event.action_age_frames = int(apply_frame) - int(obs_frame)
        if frame_ms is not None and event.effective_k is None and float(frame_ms) > 0:
            event.effective_k = float(event.latency_ms) / float(frame_ms)


def _backfill_action_event_raw_fields(event: ActionEvent, *, frame_ms: float | None) -> None:
    if event.obs_raw_frame is None:
        event.obs_raw_frame = int(event.obs_env_step)
    if event.latency_raw_frames is None:
        latency_raw_frames = _latency_ms_to_raw_frames(event.latency_ms, frame_ms=frame_ms)
        if latency_raw_frames is not None:
            event.latency_raw_frames = latency_raw_frames
    if event.ready_raw_frame is None:
        if event.obs_raw_frame is not None and event.latency_raw_frames is not None:
            event.ready_raw_frame = int(event.obs_raw_frame) + int(event.latency_raw_frames)
        elif event.ready_step_float is not None:
            event.ready_raw_frame = _ceil_step_float(event.ready_step_float)
    if event.apply_raw_frame is None and event.applied_env_step is not None:
        event.apply_raw_frame = int(event.applied_env_step)


def _latency_ms_to_raw_frames(latency_ms: float, *, frame_ms: float | None) -> int | None:
    if frame_ms is None or float(frame_ms) <= 0:
        return None
    return ms_to_raw_frames(latency_ms, env_fps=1000.0 / float(frame_ms))


def _ceil_step_float(value: float) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if math.isclose(numeric, rounded, rel_tol=1e-12, abs_tol=1e-12):
        return int(rounded)
    return int(math.ceil(numeric))


def _apply_profile_metadata(target: Any, profile_metadata: dict[str, Any]) -> None:
    for key in PROFILE_METADATA_KEYS:
        value = profile_metadata.get(key)
        if value is not None:
            setattr(target, key, value)


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
