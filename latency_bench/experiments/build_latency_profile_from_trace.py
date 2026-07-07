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
    build_bayes_markov_burst_model,
    build_bayes_markov_burst_samples,
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
    samples_ms = load_latency_samples(latencies)
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
    if actions is not None:
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
        inference_samples_ms = load_latency_samples(inference_latencies)
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
    metrics_records = None
    if episode_metrics is not None:
        metrics_records = load_jsonl_records(episode_metrics)
        manifest["inference_worker_count"] = metrics_records[0]["metadata"]["inference_worker_count"]
        if "episode_return" in metrics_records[0]:
            manifest["control_summary"] = _control_summary(metrics_records)
    if actions is not None:
        action_records = load_jsonl_records(actions)
        burst_samples = build_bayes_markov_burst_samples(action_records, metrics_records)
        burst_model = build_bayes_markov_burst_model(burst_samples)
        write_jsonl(paths["burst_samples"], burst_samples)
        write_json(paths["burst_model"], burst_model)
        manifest["latency_method"] = "temporal"
        manifest["sample_model_type"] = burst_model["model_type"]
        manifest["burst_samples_path"] = paths["burst_samples"].name
        manifest["burst_model_path"] = paths["burst_model"].name
        manifest["burst_state_key"] = burst_model["state_key"]
        if metrics_records is not None:
            manifest["worker_capacity_summary"] = _worker_capacity_summary(
                burst_samples=burst_samples,
                env_fps=env_fps,
                inference_worker_count=manifest["inference_worker_count"],
            )
        manifest["temporal_summary"] = _temporal_summary(burst_samples)
    raw_artifacts = _raw_artifacts(
        episode_metrics=episode_metrics,
        actions=actions,
        steps=steps,
    )
    if raw_artifacts:
        manifest["raw_artifacts"] = raw_artifacts
    _copy_optional_summary(
        source=None if episode_metrics is None else Path(episode_metrics).parent / "realtime_pipeline_summary.json",
        output_path=output_path,
        manifest=manifest,
        manifest_key="pipeline_summary_path",
    )
    _copy_optional_summary(
        source=None if episode_metrics is None else Path(episode_metrics).parent / "runtime_telemetry_summary.json",
        output_path=output_path,
        manifest=manifest,
        manifest_key="runtime_telemetry_summary_path",
    )
    manifest.update(metadata)
    save_profile_manifest(manifest, paths["profile"])
    return manifest


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


def _temporal_summary(burst_samples: list[dict[str, Any]]) -> dict[str, Any]:
    action_ready_values = [float(sample["action_ready_ms"]) for sample in burst_samples]
    worker_service_values = [float(sample["worker_service_ms"]) for sample in burst_samples]
    return {
        "action_ready_ms": _series_temporal_summary(action_ready_values),
        "worker_service_ms": _series_temporal_summary(worker_service_values),
    }


def _series_temporal_summary(values: list[float]) -> dict[str, Any]:
    threshold = percentile_linear(values, 90.0)
    return {
        "acf": {
            "lag1": autocorrelation(values, 1),
            "lag5": autocorrelation(values, 5),
            "lag10": autocorrelation(values, 10),
            "lag20": autocorrelation(values, 20),
        },
        "high_tail_threshold_ms": threshold,
        "high_tail_run_lengths": series_stats([float(run) for run in _high_tail_run_lengths(values, threshold)]),
    }


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
