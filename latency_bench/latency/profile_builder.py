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
    method: str = "temporal",
    hmm_states: int = 2,
    hmm_max_iterations: int = 1000,
    block_mean_length: float = 32,
) -> dict[str, Any]:
    """Build one profile from independent measurement sessions.

    Each session supplies ``measure_session_id`` and
    ``observation_attempts``. Capacity drops remain in the source sequence but
    are excluded from latency emissions.
    Marginal samples are pooled, while temporal fitting preserves every
    session boundary.
    """
    if method not in {"temporal", "hmm", "block", "ar1", "arma"}:
        raise ValueError(f"Unsupported fitted profile method: {method}")
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
        "visualization": output_path / "latency_profile.png",
    }
    if method == "temporal":
        paths.update(distribution=output_path / "latency_distribution.json",
                     burst_model=output_path / "latency_burst_model.json")
    else:
        paths["sequence_model"] = output_path / f"latency_{method}_model.json"
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
    if method == "temporal":
        model = build_hidden_regime_burst_model_from_sessions(
            session_burst_samples, model_version=temporal_model_version,
        )
        distribution = build_worker_slot_distributions_from_sessions(
            session_burst_samples,
            spike_threshold_ms_by_worker_slot=model["spike_threshold_ms_by_worker_slot"],
        )
        write_json(paths["distribution"], distribution)
        write_json(paths["burst_model"], model)
        companion_paths = {"distribution_path": paths["distribution"].name,
                           "burst_model_path": paths["burst_model"].name}
    else:
        from latency_bench.latency.sequence_profiles import build_sequence_model

        model = build_sequence_model(session_burst_samples, method=method,
                                     hmm_states=hmm_states, hmm_max_iterations=hmm_max_iterations,
                                     block_mean_length=block_mean_length)
        write_json(paths["sequence_model"], model)
        companion_paths = {"sequence_model_path": paths["sequence_model"].name}
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
        "latency_method": method,
        "sample_model_type": model["model_type"],
        **companion_paths,
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
