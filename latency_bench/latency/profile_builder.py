from __future__ import annotations

from pathlib import Path
from typing import Any

from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.analysis.utils import write_latency_distribution_plot
from latency_bench.latency.burst_profile import (
    build_burst_samples,
    build_hidden_regime_burst_model_from_sessions,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.utils.io import load_jsonl_records, write_json
from latency_bench.utils.stats import mean, percentile_linear


def build_profile_from_traces(
    *,
    sessions: list[dict[str, Any]],
    output_dir: str | Path,
    model_id: str,
    gpu_class: str,
    workload_id: str,
    instance_id: str,
    source_run_id: str,
    env_fps: float,
    metadata: dict[str, Any],
    temporal_model_version: int = 13,
) -> dict[str, Any]:
    """Build one profile from independent measurement sessions.

    Each session supplies ``measure_session_id`` and
    ``observation_attempts``. Capacity drops remain in the source sequence but
    are excluded from latency emissions.
    Marginal samples are pooled, while temporal fitting preserves every
    session boundary.
    """
    sessions = [
        {
            **session,
            "observation_attempts": Path(session["observation_attempts"]),
            "attempt_records": load_jsonl_records(session["observation_attempts"]),
        }
        for session in sessions
    ]
    output_path = Path(output_dir)
    paths = {
        "profile": output_path / "profile.json",
        "distribution": output_path / "latency_distribution.json",
        "burst_model": output_path / "latency_burst_model.json",
        "visualization": output_path / "latency_profile.png",
    }
    existing_artifacts = [path for path in paths.values() if path.exists()]
    if existing_artifacts:
        existing = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Profile artifacts already exist: {existing}")

    output_path.mkdir(parents=True, exist_ok=True)
    session_burst_samples = [
        build_burst_samples(
            session["attempt_records"],
            measure_session_id=session["measure_session_id"],
        )
        for session in sessions
    ]
    burst_samples = [sample for samples in session_burst_samples for sample in samples]
    burst_model = build_hidden_regime_burst_model_from_sessions(
        session_burst_samples, model_version=temporal_model_version,
    )
    distribution = build_worker_slot_distributions_from_sessions(
        session_burst_samples,
        spike_threshold_ms_by_worker_slot=burst_model[
            "spike_threshold_ms_by_worker_slot"
        ],
    )

    write_json(paths["distribution"], distribution)
    write_latency_distribution_plot(
        [sample["observation_to_action_latency_ms"] for sample in burst_samples],
        paths["visualization"],
    )
    manifest = {
        "model_id": model_id,
        "gpu_class": gpu_class,
        "workload_id": workload_id,
        "instance_id": instance_id,
        "source_run_id": source_run_id,
        "env_fps": env_fps,
        "frame_ms": 1000.0 / env_fps,
        "n_observation_attempts": sum(
            len(session["attempt_records"]) for session in sessions
        ),
        "n_admitted_observations": len(burst_samples),
        "n_capacity_drops": sum(
            not record["submitted"]
            for session in sessions
            for record in session["attempt_records"]
        ),
        "latency_kind": "observation_to_action_latency",
        "latency_method": "temporal",
        "sample_model_type": burst_model["model_type"],
        "distribution_path": paths["distribution"].name,
        "burst_model_path": paths["burst_model"].name,
        "visualization_path": paths["visualization"].name,
        "summary": compute_latency_profile_summary(
            [
                sample["observation_to_action_latency_ms"]
                for sample in burst_samples
            ],
            env_fps,
        ),
        "per_slot_summary": _worker_slot_summaries(
            burst_samples=burst_samples,
        ),
        "provenance": metadata,
    }
    write_json(paths["burst_model"], burst_model)
    write_json(paths["profile"], manifest)
    return manifest


def _worker_slot_summaries(
    *,
    burst_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    by_slot: dict[int, list[dict[str, Any]]] = {}
    for sample in burst_samples:
        by_slot.setdefault(sample["worker_id"], []).append(sample)
    summaries = {}
    for slot, samples in sorted(by_slot.items()):
        observation_to_action_samples = [
            sample["observation_to_action_latency_ms"]
            for sample in samples
        ]
        summaries[str(slot)] = {
            "admitted_count": len(samples),
            "mean_worker_service_time_ms": mean(
                [sample["worker_service_time_ms"] for sample in samples]
            ),
            "p95_worker_service_time_ms": percentile_linear(
                [sample["worker_service_time_ms"] for sample in samples], 95.0
            ),
            "p99_worker_service_time_ms": percentile_linear(
                [sample["worker_service_time_ms"] for sample in samples], 99.0
            ),
            "mean_observation_to_action_latency_ms": mean(
                observation_to_action_samples
            ),
            "p95_observation_to_action_latency_ms": percentile_linear(
                observation_to_action_samples, 95.0
            ),
        }
    return summaries
