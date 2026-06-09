from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.latency.profile import (
    SUPPORTED_PROFILE_TYPES,
    load_latency_samples,
    save_profile_manifest,
    write_latency_samples,
)


PROFILE_FILENAME = "profile.json"
TRACE_FILENAME = "latencies.jsonl"
SUMMARY_JSON_FILENAME = "profile_summary.json"
SUMMARY_CSV_FILENAME = "profile_summary.csv"
PROFILE_ARTIFACT_FILENAMES = (
    PROFILE_FILENAME,
    TRACE_FILENAME,
    SUMMARY_JSON_FILENAME,
    SUMMARY_CSV_FILENAME,
)


def build_profile_from_trace(
    *,
    latencies: str | Path,
    output_dir: str | Path,
    profile_id: str,
    profile_type: str,
    policy_id: str,
    env_id: str,
    target_device: str,
    fps: float,
    batch_size: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if profile_type not in SUPPORTED_PROFILE_TYPES:
        raise ValueError(f"Unsupported profile_type {profile_type!r}; expected one of {sorted(SUPPORTED_PROFILE_TYPES)}")

    samples_ms = load_latency_samples(latencies)

    output_path = Path(output_dir)
    artifact_paths = [output_path / filename for filename in PROFILE_ARTIFACT_FILENAMES]
    existing_artifacts = [path for path in artifact_paths if path.exists()]
    if existing_artifacts and not overwrite:
        existing = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Refusing to overwrite existing profile artifacts without --overwrite: {existing}")

    output_path.mkdir(parents=True, exist_ok=True)
    summary = compute_latency_profile_summary(samples_ms, fps)
    write_latency_samples(samples_ms, output_path / TRACE_FILENAME)
    _write_json(summary, output_path / SUMMARY_JSON_FILENAME)
    _write_summary_csv(summary, output_path / SUMMARY_CSV_FILENAME)

    manifest = {
        "schema_version": 1,
        "profile_id": profile_id,
        "profile_type": profile_type,
        "policy_id": policy_id,
        "env_id": env_id,
        "target_device": target_device,
        "fps": fps,
        "frame_ms": summary["frame_ms"],
        "n_samples": len(samples_ms),
        "batch_size": batch_size,
        "trace_path": TRACE_FILENAME,
        "summary": summary,
    }
    save_profile_manifest(manifest, output_path / PROFILE_FILENAME)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a formal latency profile from an existing JSONL trace.")
    parser.add_argument("--latencies", type=Path, required=True, help="Path to input JSONL latency trace.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where profile artifacts are written.")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-type", choices=sorted(SUPPORTED_PROFILE_TYPES), required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    build_profile_from_trace(
        latencies=args.latencies,
        output_dir=args.output_dir,
        profile_id=args.profile_id,
        profile_type=args.profile_type,
        policy_id=args.policy_id,
        env_id=args.env_id,
        target_device=args.target_device,
        fps=args.fps,
        batch_size=args.batch_size,
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
