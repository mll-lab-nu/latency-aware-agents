from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
    profile_type: str,
    policy_id: str,
    env_id: str,
    target_device: str,
    fps: float,
    batch_size: int = 1,
    overwrite: bool = False,
    policy_type: str | None = None,
    checkpoint_kind: str | None = None,
    checkpoint_path_or_id: str | None = None,
    checkpoint_fingerprint: str | None = None,
    base_model_id_or_path: str | None = None,
) -> dict[str, Any]:
    samples_ms = load_latency_samples(latencies)
    output_path = Path(output_dir)
    paths = {
        "profile": output_path / "profile.json",
        "trace": output_path / "latencies.jsonl",
        "distribution": output_path / "latency_distribution.json",
        "summary_json": output_path / "profile_summary.json",
        "summary_csv": output_path / "profile_summary.csv",
    }
    existing_artifacts = [path for path in paths.values() if path.exists()]
    if existing_artifacts and not overwrite:
        existing = ", ".join(str(path) for path in existing_artifacts)
        raise FileExistsError(f"Refusing to overwrite existing profile artifacts without --overwrite: {existing}")

    output_path.mkdir(parents=True, exist_ok=True)
    summary = compute_latency_profile_summary(samples_ms, fps)
    distribution = build_inverse_cdf_distribution(samples_ms)
    write_latency_samples(samples_ms, paths["trace"])
    save_latency_distribution(distribution, paths["distribution"])
    _write_json(summary, paths["summary_json"])
    _write_summary_csv(summary, paths["summary_csv"])

    manifest = {
        "schema_version": 2,
        "profile_id": profile_id,
        "profile_type": profile_type,
        "policy_id": policy_id,
        "env_id": env_id,
        "target_device": target_device,
        "fps": fps,
        "frame_ms": summary["frame_ms"],
        "n_samples": len(samples_ms),
        "batch_size": batch_size,
        "trace_path": paths["trace"].name,
        "distribution_type": "inverse_cdf",
        "distribution_path": paths["distribution"].name,
        "summary": summary,
    }
    if policy_type is not None:
        manifest["policy_type"] = policy_type
    if checkpoint_kind is not None:
        manifest["checkpoint_kind"] = checkpoint_kind
    if checkpoint_path_or_id is not None:
        manifest["checkpoint_path_or_id"] = checkpoint_path_or_id
    if checkpoint_fingerprint is not None:
        manifest["checkpoint_fingerprint"] = checkpoint_fingerprint
    if base_model_id_or_path is not None:
        manifest["base_model_id_or_path"] = base_model_id_or_path
    save_profile_manifest(manifest, paths["profile"])
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a formal latency profile from an existing JSONL trace.")
    parser.add_argument("--latencies", type=Path, required=True, help="Path to input JSONL latency trace.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where profile artifacts are written.")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-type", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--policy-type")
    parser.add_argument("--checkpoint-kind")
    parser.add_argument("--checkpoint-path-or-id")
    parser.add_argument("--checkpoint-fingerprint")
    parser.add_argument("--base-model-id-or-path")
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
        policy_type=args.policy_type,
        checkpoint_kind=args.checkpoint_kind,
        checkpoint_path_or_id=args.checkpoint_path_or_id,
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        base_model_id_or_path=args.base_model_id_or_path,
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
