from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from latency_bench.analysis.utils import write_latency_distribution_plot
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
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
    n_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    samples_ms = load_latency_samples(latencies)
    if n_samples is not None:
        samples_ms = samples_ms[:n_samples]
    output_path = Path(output_dir)
    paths = {
        "profile": output_path / "profile.json",
        "trace": output_path / "latencies.jsonl",
        "distribution": output_path / "latency_distribution.json",
        "plot": output_path / "latency_distribution.png",
        "summary_json": output_path / "profile_summary.json",
        "summary_csv": output_path / "profile_summary.csv",
    }
    existing_artifacts = [path for path in paths.values() if path.exists()]
    if existing_artifacts and not overwrite:
        existing = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Refusing to overwrite existing profile artifacts without --overwrite: {existing}")

    output_path.mkdir(parents=True, exist_ok=True)
    summary = compute_latency_profile_summary(samples_ms, env_fps)
    distribution = build_inverse_cdf_distribution(samples_ms)
    write_latency_samples(samples_ms, paths["trace"])
    save_latency_distribution(distribution, paths["distribution"])
    write_latency_distribution_plot(samples_ms, paths["plot"])
    _write_json(summary, paths["summary_json"])
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
        "distribution_path": paths["distribution"].name,
        "plot_path": paths["plot"].name,
        "summary": summary,
    }
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
    parser.add_argument("--n-samples", type=int)
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
        n_samples=args.n_samples,
        overwrite=args.overwrite,
    )


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
