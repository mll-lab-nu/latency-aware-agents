from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from latency_bench.analysis.utils import write_latency_distribution_plot
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.utils.io import load_jsonl_records, write_csv, write_json, write_jsonl
from latency_bench.utils.stats import autocorrelation, mean, percentile_linear, series_stats
from latency_bench.latency.burst_profile import (
    build_burst_samples,
    build_hidden_regime_burst_model,
    build_hidden_regime_burst_model_from_sessions,
)
from latency_bench.latency.profile import (
    build_inverse_cdf_distribution,
    load_latency_samples,
    save_latency_distribution,
    save_profile_manifest,
    write_latency_samples,
)


def build_profile_from_trace(
    *,
    latencies: str | Path,
    output_dir: str | Path,
    profile_id: str,
    policy_id: str,
    env_id: str,
    target_device: str,
    env_fps: float,
    metadata: dict[str, Any],
    latency_kind: str = "inference_latency",
    inference_latencies: str | Path | None = None,
    episode_metrics: str | Path | None = None,
    actions: str | Path | None = None,
    steps: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    session = {
        "measure_session_id": None,
        "latencies": Path(latencies),
        "samples_ms": load_latency_samples(latencies),
        "inference_latencies": None if inference_latencies is None else Path(inference_latencies),
        "inference_samples_ms": (
            None if inference_latencies is None else load_latency_samples(inference_latencies)
        ),
        "episode_metrics": None if episode_metrics is None else Path(episode_metrics),
        "metrics_records": (
            None if episode_metrics is None else load_jsonl_records(episode_metrics)
        ),
        "actions": None if actions is None else Path(actions),
        "action_records": None if actions is None else load_jsonl_records(actions),
        "steps": None if steps is None else Path(steps),
    }
    return _build_profile_from_loaded_sessions(
        sessions=[session],
        output_dir=output_dir,
        profile_id=profile_id,
        policy_id=policy_id,
        env_id=env_id,
        target_device=target_device,
        env_fps=env_fps,
        metadata=metadata,
        latency_kind=latency_kind,
        overwrite=overwrite,
        multi_session=False,
    )


def build_profile_from_traces(
    *,
    sessions: list[dict[str, Any]],
    output_dir: str | Path,
    profile_id: str,
    policy_id: str,
    env_id: str,
    target_device: str,
    env_fps: float,
    metadata: dict[str, Any],
    latency_kind: str = "e2e_latency",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one profile from independent measurement sessions.

    Each session supplies ``measure_session_id``, ``latencies``,
    ``inference_latencies``, ``episode_metrics``, ``actions``, and ``steps``.
    Marginal samples are pooled, while temporal fitting preserves every
    session boundary.
    """
    loaded_sessions = [
        {
            **session,
            "latencies": Path(session["latencies"]),
            "samples_ms": load_latency_samples(session["latencies"]),
            "inference_latencies": Path(session["inference_latencies"]),
            "inference_samples_ms": load_latency_samples(session["inference_latencies"]),
            "episode_metrics": Path(session["episode_metrics"]),
            "metrics_records": load_jsonl_records(session["episode_metrics"]),
            "actions": Path(session["actions"]),
            "action_records": load_jsonl_records(session["actions"]),
            "steps": Path(session["steps"]),
        }
        for session in sessions
    ]
    return _build_profile_from_loaded_sessions(
        sessions=loaded_sessions,
        output_dir=output_dir,
        profile_id=profile_id,
        policy_id=policy_id,
        env_id=env_id,
        target_device=target_device,
        env_fps=env_fps,
        metadata=metadata,
        latency_kind=latency_kind,
        overwrite=overwrite,
        multi_session=True,
    )


def _build_profile_from_loaded_sessions(
    *,
    sessions: list[dict[str, Any]],
    output_dir: str | Path,
    profile_id: str,
    policy_id: str,
    env_id: str,
    target_device: str,
    env_fps: float,
    metadata: dict[str, Any],
    latency_kind: str,
    overwrite: bool,
    multi_session: bool,
) -> dict[str, Any]:
    samples_ms = [
        sample
        for session in sessions
        for sample in session["samples_ms"]
    ]
    output_path = Path(output_dir)
    trace_name_by_kind = {
        "inference_latency": "infer_latencies.jsonl",
        "e2e_latency": "e2e_latencies.jsonl",
    }
    paths = {
        "profile": output_path / "profile.json",
        "trace": output_path / trace_name_by_kind[latency_kind],
        "distribution": output_path / "latency_distribution.json",
        "plot": output_path / "latency_distribution.png",
        "summary_json": output_path / "profile_summary.json",
        "summary_csv": output_path / "profile_summary.csv",
    }
    has_actions = sessions[0]["action_records"] is not None
    if has_actions:
        paths["burst_samples"] = output_path / "latency_burst_samples.jsonl"
        paths["burst_model"] = output_path / "latency_burst_model.json"
    if latency_kind == "e2e_latency":
        paths["inference_trace"] = output_path / trace_name_by_kind["inference_latency"]
    existing_artifacts = [path for path in paths.values() if path.exists()]
    if existing_artifacts and not overwrite:
        existing = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Refusing to overwrite existing profile artifacts without --overwrite: {existing}")

    output_path.mkdir(parents=True, exist_ok=True)
    summary = compute_latency_profile_summary(samples_ms, env_fps)
    distribution = build_inverse_cdf_distribution(samples_ms)
    write_latency_samples(samples_ms, paths["trace"])
    inference_summary = None
    if latency_kind == "e2e_latency":
        inference_samples_ms = [
            sample
            for session in sessions
            for sample in session["inference_samples_ms"]
        ]
        write_latency_samples(inference_samples_ms, paths["inference_trace"])
        inference_summary = compute_latency_profile_summary(inference_samples_ms, env_fps)
    save_latency_distribution(distribution, paths["distribution"])
    write_latency_distribution_plot(samples_ms, paths["plot"])
    write_json(paths["summary_json"], summary)
    write_csv(paths["summary_csv"], [summary], summary.keys())

    manifest = {
        "profile_id": profile_id,
        "policy_id": policy_id,
        "env_id": env_id,
        "target_device": target_device,
        "env_fps": env_fps,
        "frame_ms": summary["frame_ms"],
        "n_samples": len(samples_ms),
        "trace_path": paths["trace"].name,
        "latency_kind": latency_kind,
        "distribution_path": paths["distribution"].name,
        "plot_path": paths["plot"].name,
        "summary": summary,
    }
    if inference_summary is not None:
        manifest["inference_trace_path"] = paths["inference_trace"].name
        manifest["inference_summary"] = inference_summary
    metrics_records = (
        None
        if sessions[0]["metrics_records"] is None
        else [
            record
            for session in sessions
            for record in session["metrics_records"]
        ]
    )
    if metrics_records is not None:
        manifest["inference_worker_count"] = metrics_records[0]["metadata"]["inference_worker_count"]
        if "episode_return" in metrics_records[0]:
            manifest["control_summary"] = _control_summary(metrics_records)
    if has_actions:
        session_burst_samples = [
            build_burst_samples(
                session["action_records"],
                session["metrics_records"],
                measure_session_id=(
                    session["measure_session_id"] if multi_session else None
                ),
            )
            for session in sessions
        ]
        burst_samples = [
            sample
            for samples in session_burst_samples
            for sample in samples
        ]
        burst_model = (
            build_hidden_regime_burst_model_from_sessions(session_burst_samples)
            if multi_session
            else build_hidden_regime_burst_model(burst_samples)
        )
        write_jsonl(paths["burst_samples"], burst_samples)
        write_json(paths["burst_model"], burst_model)
        manifest["latency_method"] = "temporal"
        manifest["sample_model_type"] = burst_model["model_type"]
        manifest["burst_samples_path"] = paths["burst_samples"].name
        manifest["burst_model_path"] = paths["burst_model"].name
        if metrics_records is not None:
            manifest["worker_capacity_summary"] = _worker_capacity_summary(
                burst_samples=burst_samples,
                env_fps=env_fps,
                inference_worker_count=manifest["inference_worker_count"],
            )
        manifest["temporal_summary"] = _temporal_summary_from_sessions(
            session_burst_samples, burst_model
        )
    if multi_session:
        manifest["fit_sessions"] = [
            _fit_session_manifest(session)
            for session in sessions
        ]
    else:
        session = sessions[0]
        raw_artifacts = _raw_artifacts(
            episode_metrics=session["episode_metrics"],
            actions=session["actions"],
            steps=session["steps"],
        )
        if raw_artifacts:
            manifest["raw_artifacts"] = raw_artifacts
        _copy_optional_summary(
            source=(
                None
                if session["episode_metrics"] is None
                else session["episode_metrics"].parent / "realtime_pipeline_summary.json"
            ),
            output_path=output_path,
            manifest=manifest,
            manifest_key="pipeline_summary_path",
        )
        _copy_optional_summary(
            source=(
                None
                if session["episode_metrics"] is None
                else session["episode_metrics"].parent / "runtime_telemetry_summary.json"
            ),
            output_path=output_path,
            manifest=manifest,
            manifest_key="runtime_telemetry_summary_path",
        )
    manifest.update(metadata)
    save_profile_manifest(manifest, paths["profile"])
    return manifest


def _fit_session_manifest(session: dict[str, Any]) -> dict[str, Any]:
    result = {
        "measure_session_id": session["measure_session_id"],
        "n_samples": len(session["samples_ms"]),
        "n_inference_samples": len(session["inference_samples_ms"]),
        "n_episodes": len(session["metrics_records"]),
        "n_actions": len(session["action_records"]),
        "raw_artifacts": _raw_artifacts(
            episode_metrics=session["episode_metrics"],
            actions=session["actions"],
            steps=session["steps"],
        ),
    }
    if "provenance" in session:
        result["provenance"] = session["provenance"]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a formal latency profile from an existing JSONL trace.")
    parser.add_argument("--latencies", type=Path, required=True, help="Path to input JSONL latency trace.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where profile artifacts are written.")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--env-fps", type=float, required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--latency-kind", choices=["inference_latency", "e2e_latency"], default="inference_latency")
    parser.add_argument("--inference-latencies", type=Path)
    parser.add_argument("--episode-metrics", type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    build_profile_from_trace(
        latencies=args.latencies,
        output_dir=args.output_dir,
        profile_id=args.profile_id,
        policy_id=args.policy_id,
        env_id=args.env_id,
        target_device=args.target_device,
        env_fps=args.env_fps,
        metadata=json.loads(args.metadata_json),
        latency_kind=args.latency_kind,
        inference_latencies=args.inference_latencies,
        episode_metrics=args.episode_metrics,
        actions=args.actions,
        steps=args.steps,
        overwrite=args.overwrite,
    )


# Per-episode metadata counters summed across the trace into the control summary.
_CONTROL_METADATA_SUM_FIELDS = (
    "submitted_observation_frames",
    "dropped_observation_count",
    "ignored_late_result_count",
    "discarded_ready_result_count",
    "fifo_blocked_count",
    "env_deadline_miss_count",
)


def _control_summary(metrics_records: list[dict[str, Any]]) -> dict[str, Any]:
    survival_steps = [float(record["survival_steps"]) for record in metrics_records]
    sums = {
        field: sum(int(record["metadata"][field]) for record in metrics_records)
        for field in _CONTROL_METADATA_SUM_FIELDS
    }
    num_actions = sum(int(record["num_actions"]) for record in metrics_records)
    return {
        "episodes": len(metrics_records),
        "mean_return": mean([float(record["episode_return"]) for record in metrics_records]),
        "mean_survival_steps": mean(survival_steps),
        "num_actions": num_actions,
        "num_dropped_actions": sum(int(record["num_dropped_actions"]) for record in metrics_records),
        "actions_per_step": num_actions / sum(survival_steps),
        "drop_rate_per_1k_observations": sums["dropped_observation_count"] * 1000.0 / sums["submitted_observation_frames"],
        "actual_env_fps": series_stats([float(record["metadata"]["actual_env_fps"]) for record in metrics_records]),
        "max_wall_clock_lag_ms": max(float(record["metadata"]["max_wall_clock_lag_ms"]) for record in metrics_records),
        **sums,
    }


def _worker_capacity_summary(
    *,
    burst_samples: list[dict[str, Any]],
    env_fps: float,
    inference_worker_count: int,
) -> dict[str, Any]:
    worker_service_values = [float(sample["worker_service_ms"]) for sample in burst_samples]
    action_ready_values = [float(sample["action_ready_ms"]) for sample in burst_samples]
    mean_worker_service_ms = mean(worker_service_values)
    return {
        "inference_worker_count": int(inference_worker_count),
        "mean_worker_service_ms": mean_worker_service_ms,
        "p95_worker_service_ms": percentile_linear(worker_service_values, 95.0),
        "p99_worker_service_ms": percentile_linear(worker_service_values, 99.0),
        "mean_action_ready_ms": mean(action_ready_values),
        "p95_action_ready_ms": percentile_linear(action_ready_values, 95.0),
        "worker_utilization_estimate": mean_worker_service_ms * float(env_fps) / 1000.0 / float(inference_worker_count),
        "p95_worker_service_frames": percentile_linear(worker_service_values, 95.0) * float(env_fps) / 1000.0,
    }


def _temporal_summary_from_sessions(
    session_burst_samples: list[list[dict[str, Any]]],
    burst_model: dict[str, Any],
) -> dict[str, Any]:
    action_ready_sessions = [
        [float(sample["action_ready_ms"]) for sample in samples]
        for samples in session_burst_samples
    ]
    worker_service_sessions = [
        [float(sample["worker_service_ms"]) for sample in samples]
        for samples in session_burst_samples
    ]
    dwell_lengths = [float(length) for length in burst_model["burst_dwell_lengths"]]
    return {
        "action_ready_ms": _series_temporal_summary_from_sessions(action_ready_sessions),
        "worker_service_ms": _series_temporal_summary_from_sessions(worker_service_sessions),
        "burst_regime": {
            "spike_threshold_ms": burst_model["spike_threshold_ms"],
            "burst_count": len(dwell_lengths),
            "regime_step_counts": burst_model["regime_step_counts"],
            "burst_dwell_lengths": series_stats(dwell_lengths) if dwell_lengths else None,
            "rank_processes": [
                {
                    "draw_count": process["draw_count"],
                    "spike_count": process["spike_count"],
                    "severity": series_stats(process["severity"]["latency_ms"]),
                    "level_ms": (
                        None if process["level_ms"] is None
                        else series_stats(process["level_ms"]["latency_ms"])
                    ),
                }
                for process in burst_model["burst_rank_processes"]
            ],
        },
    }


def _series_temporal_summary_from_sessions(
    value_sessions: list[list[float]],
) -> dict[str, Any]:
    values = [value for session in value_sessions for value in session]
    threshold = percentile_linear(values, 90.0)
    run_lengths = [
        float(run)
        for session in value_sessions
        for run in _high_tail_run_lengths(session, threshold)
    ]
    return {
        "acf": {
            "lag1": _session_autocorrelation(value_sessions, 1),
            "lag5": _session_autocorrelation(value_sessions, 5),
            "lag10": _session_autocorrelation(value_sessions, 10),
            "lag20": _session_autocorrelation(value_sessions, 20),
        },
        "high_tail_threshold_ms": threshold,
        "high_tail_run_lengths": series_stats(run_lengths),
    }


def _session_autocorrelation(value_sessions: list[list[float]], lag: int) -> float | None:
    if len(value_sessions) == 1:
        return autocorrelation(value_sessions[0], lag)
    values = [value for session in value_sessions for value in session]
    if not any(lag < len(session) for session in value_sessions):
        return None
    series_mean = mean(values)
    variance = sum((value - series_mean) ** 2 for value in values)
    if variance == 0.0:
        return 0.0
    return sum(
        (session[index] - series_mean) * (session[index - lag] - series_mean)
        for session in value_sessions
        for index in range(lag, len(session))
    ) / variance


def _raw_artifacts(
    *,
    episode_metrics: str | Path | None,
    actions: str | Path | None,
    steps: str | Path | None,
) -> dict[str, str]:
    artifacts = {}
    if episode_metrics is not None:
        artifacts["episode_metrics"] = Path(episode_metrics).name
    if actions is not None:
        artifacts["actions"] = Path(actions).name
    if steps is not None:
        artifacts["steps"] = Path(steps).name
    return artifacts


def _copy_optional_summary(
    *,
    source: Path | None,
    output_path: Path,
    manifest: dict[str, Any],
    manifest_key: str,
) -> None:
    if source is not None and source.exists():
        destination = output_path / source.name
        shutil.copyfile(source, destination)
        manifest[manifest_key] = destination.name


def _high_tail_run_lengths(values: list[float], threshold: float) -> list[int]:
    runs = []
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


if __name__ == "__main__":
    main()
