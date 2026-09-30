from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from latency_bench.core.clock import frames_float_to_raw_frames, ms_to_raw_frames
from latency_bench.core.types import ActionEvent, EpisodeMetrics, StepRecord
from latency_bench.utils.stats import percentile_linear


@dataclass
class EpisodeMetricAccumulator:
    """Accumulate episode metrics independently of trajectory retention."""

    episode_return_env: float = 0.0
    survival_steps: int = 0
    game_score: float | None = None
    _final_metadata: dict[str, Any] = field(default_factory=dict)
    return_raw: float | None = None
    num_invalid_actions: int = 0
    metric_metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, reward: float, info: dict[str, Any]) -> None:
        self.episode_return_env += reward
        self.survival_steps += 1
        if "invalid_action" in info and info["invalid_action"]:
            self.num_invalid_actions += 1
        self.game_score = (float(info["game_score"]) if "game_score" in info else
                           float(info["score"]) if "score" in info else None)
        self._final_metadata.clear()
        if "task_metrics" in info:
            self._final_metadata["task_metrics"] = info["task_metrics"]
        if "task_metric_moments" in info:
            self._final_metadata["task_metric_moments"] = info["task_metric_moments"]
        raw = _first_number(info, _RAW_RETURN_KEYS)
        if raw is None and "episode_extra_stats" in info:
            extra = info["episode_extra_stats"]
            if isinstance(extra, dict):
                raw = _first_number(extra, _RAW_RETURN_KEYS)
        if raw is not None:
            self.return_raw = raw
        if "stale_prefix_frames" in info:
            if "hold_last_chunk_action" not in self.metric_metadata:
                self.metric_metadata["hold_last_chunk_action"] = info["hold_last_chunk_action"]
            for key in ("underflow_count", "underflow_frames", "underflow_duration_ms"):
                self.metric_metadata[key] = max(self.metric_metadata.get(key, 0), info.get(key, 0))
            for key in ("stale_prefix_frames", "chunk_age_raw_frames", "chunk_age_ms"):
                self.metric_metadata[key] = max(self.metric_metadata.get(key, 0), info[key])

    def compute(self, *, episode_id: int, latency_values_ms: list[float],
                num_actions: int, num_dropped_actions: int,
                metadata: dict[str, Any] | None = None) -> EpisodeMetrics:
        return EpisodeMetrics(
            episode_id=episode_id,
            episode_return=self.episode_return_env,
            survival_steps=self.survival_steps,
            episode_return_env=self.episode_return_env,
            return_raw=self.return_raw,
            game_score=self.game_score,
            mean_latency_ms=mean(latency_values_ms) if latency_values_ms else None,
            p90_latency_ms=percentile_linear(latency_values_ms, 90.0) if latency_values_ms else None,
            p99_latency_ms=percentile_linear(latency_values_ms, 99.0) if latency_values_ms else None,
            num_actions=num_actions,
            num_dropped_actions=num_dropped_actions,
            num_invalid_actions=self.num_invalid_actions,
            metadata={**(metadata or {}), **self.metric_metadata, **self._final_metadata},
        )


def compute_episode_metrics(
    episode_id: int,
    step_records: list[StepRecord],
    action_events: list[ActionEvent],
    latency_values_ms: list[float],
    metadata: dict[str, Any] | None = None,
    frame_ms: float | None = None,
) -> EpisodeMetrics:
    _finalize_action_events(action_events, frame_ms=frame_ms)
    metrics = EpisodeMetricAccumulator()
    for record in step_records:
        metrics.update(record.reward, record.info)
    return metrics.compute(
        episode_id=episode_id,
        latency_values_ms=latency_values_ms,
        num_actions=len(action_events),
        num_dropped_actions=sum(event.dropped for event in action_events),
        metadata=metadata,
    )


PROFILE_METADATA_KEYS = (
    "model_id",
    "gpu_class",
    "workload_id",
    "instance_id",
    "source_run_id",
)

_RAW_RETURN_KEYS = (
    "return_raw",
    "raw_return",
    "episodic_raw_return",
    "episode/raw_return",
)


def latency_type_from_source(source: Any, *, measured: bool = False) -> str:
    if measured:
        return "measured"
    return getattr(source, "latency_type", source.__class__.__name__)


def profile_metadata_from_source(source: Any) -> dict[str, Any]:
    metadata = {key: None for key in PROFILE_METADATA_KEYS}
    profile_metadata = getattr(source, "profile_metadata", None)
    if isinstance(profile_metadata, dict):
        for key in PROFILE_METADATA_KEYS:
            if key in profile_metadata and profile_metadata[key] is not None:
                metadata[key] = profile_metadata[key]
    return metadata


def apply_latency_record_raw_fields(
    *,
    record: Any,
    latency_type: str,
    profile_metadata: dict[str, Any] | None = None,
) -> None:
    record.latency_type = latency_type
    _apply_profile_metadata(record, profile_metadata or {})


def episode_raw_fact_metadata(
    *,
    mode: str,
    episode_seed: int | None,
    env_fps: float,
    obs_fps: float,
    frame_ms: float,
    latency_type: str,
    latency_source: Any | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "episode_seed": episode_seed,
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "frame_ms": frame_ms,
        "latency_type": latency_type,
        **({} if latency_source is None else profile_metadata_from_source(latency_source)),
    }


def active_action_step_info(
    *,
    active_event: ActionEvent | None,
    scheduled_time_ms: float,
    frame_ms: float,
    latency_type: str,
) -> dict[str, Any]:
    active_obs_frame = None if active_event is None else active_event.obs_env_step
    info = {
        "active_action_id": None if active_event is None else active_event.action_id,
        "active_action_obs_env_step": None if active_event is None else active_obs_frame,
        "scheduled_time_ms": float(scheduled_time_ms),
        "frame_time_ms": float(frame_ms),
        "latency_type": latency_type,
    }
    if active_event is not None and active_event.action_chunk is not None:
        info.update(
            {
                "hold_last_chunk_action": active_event.metadata["hold_last_chunk_action"],
                "stale_prefix_frames": active_event.metadata["stale_prefix_frames"],
                "chunk_age_raw_frames": active_event.metadata["chunk_age_raw_frames"],
                "chunk_age_ms": active_event.metadata["chunk_age_ms"],
                "underflow_count": active_event.metadata.get("underflow_count", 0),
                "underflow_frames": active_event.metadata.get("underflow_frames", 0),
                "underflow_duration_ms": active_event.metadata.get("underflow_duration_ms", 0.0),
            }
        )
    return info


def _first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return None


def _finalize_action_events(action_events: list[ActionEvent], *, frame_ms: float | None) -> None:
    for event in action_events:
        _backfill_action_event_raw_fields(event, frame_ms=frame_ms)


def _backfill_action_event_raw_fields(event: ActionEvent, *, frame_ms: float | None) -> None:
    if event.latency_raw_frames is None:
        latency_raw_frames = _latency_ms_to_raw_frames(event.latency.latency_ms, frame_ms=frame_ms)
        if latency_raw_frames is not None:
            event.latency_raw_frames = latency_raw_frames
    if event.ready_raw_frame is None:
        if event.latency_raw_frames is not None:
            event.ready_raw_frame = int(event.obs_env_step) + int(event.latency_raw_frames)
        elif event.ready_step_float is not None:
            event.ready_raw_frame = frames_float_to_raw_frames(event.ready_step_float)


def _latency_ms_to_raw_frames(latency_ms: float, *, frame_ms: float | None) -> int | None:
    if frame_ms is None:
        return None
    return ms_to_raw_frames(latency_ms, env_fps=1000.0 / frame_ms)


def _apply_profile_metadata(target: Any, profile_metadata: dict[str, Any]) -> None:
    for key in PROFILE_METADATA_KEYS:
        value = profile_metadata.get(key)
        if value is not None:
            setattr(target, key, value)
