from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from latency_bench.analysis.utils import write_latency_distribution_plot
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.utils.io import load_jsonl_records, write_json
from latency_bench.latency.burst_profile import build_latency_burst_model, build_latency_burst_samples
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
    _write_summary_csv(summary, paths["summary_csv"])

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
        manifest["e2e_trace_path"] = paths["trace"].name
        manifest["e2e_summary"] = summary
    metrics_records = None
    if episode_metrics is not None:
        metrics_records = load_jsonl_records(episode_metrics)
        manifest["inference_worker_count"] = metrics_records[0]["metadata"]["inference_worker_count"]
    if actions is not None:
        action_records = load_jsonl_records(actions)
        burst_samples = build_latency_burst_samples(action_records, metrics_records)
        burst_model = build_latency_burst_model(burst_samples)
        _write_jsonl(paths["burst_samples"], burst_samples)
        write_json(paths["burst_model"], burst_model)
        manifest["sample_model_type"] = "semi_markov_latency_frames"
        manifest["burst_samples_path"] = paths["burst_samples"].name
        manifest["burst_model_path"] = paths["burst_model"].name
        manifest["burst_state_key"] = "latency_raw_frames"
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
        overwrite=args.overwrite,
    )


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
