from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.latency.burst_profile import build_latency_burst_model, build_latency_burst_samples
from latency_bench.latency.profile import load_profile_manifest, save_profile_manifest
from latency_bench.utils.io import load_jsonl_records, write_json, write_jsonl
from scripts.cross_device.profile_cache import upload_folder_to_hf


@dataclass(frozen=True)
class BackfillTarget:
    run_dir: Path
    actions_path: Path
    metrics_path: Path
    profile_path: Path


def discover_targets(root: Path) -> list[BackfillTarget]:
    targets: list[BackfillTarget] = []
    for metrics_path in sorted(root.rglob("episode_metrics.jsonl")):
        run_dir = metrics_path.parent
        parts = run_dir.parts
        run_index = _measure_run_index(parts)
        if run_index is None:
            continue
        records = load_jsonl_records(metrics_path)
        profile_id = records[0]["metadata"]["profile_id"]
        prefix = Path(*parts[:run_index])
        policy = parts[run_index + 2]
        env = parts[run_index + 3]
        profile_path = prefix / "profiles" / policy / env / profile_id / "profile.json"
        targets.append(
            BackfillTarget(
                run_dir=run_dir,
                actions_path=run_dir / "actions.jsonl",
                metrics_path=metrics_path,
                profile_path=profile_path,
            )
        )
    return targets


def apply_backfill(target: BackfillTarget) -> dict[str, int]:
    action_records = load_jsonl_records(target.actions_path)
    metrics_records = load_jsonl_records(target.metrics_path)
    burst_samples = build_latency_burst_samples(action_records, metrics_records)
    burst_model = build_latency_burst_model(burst_samples)
    samples_path = target.profile_path.parent / "latency_burst_samples.jsonl"
    model_path = target.profile_path.parent / "latency_burst_model.json"
    write_jsonl(samples_path, burst_samples)
    write_json(model_path, burst_model)

    profile = load_profile_manifest(target.profile_path)
    profile["sample_model_type"] = "semi_markov_latency_frames"
    profile["burst_samples_path"] = samples_path.name
    profile["burst_model_path"] = model_path.name
    profile["burst_state_key"] = "latency_raw_frames"
    save_profile_manifest(profile, target.profile_path)
    return {"action_count": len(action_records), "burst_sample_count": len(burst_samples)}


def upload_target(*, root: Path, target: BackfillTarget, repo_id: str, repo_type: str) -> None:
    upload_folder_to_hf(
        repo_id=repo_id,
        local_dir=target.profile_path.parent,
        repo_type=repo_type,
        path_in_repo=_profile_path_in_repo(root=root, profile_dir=target.profile_path.parent),
        allow_patterns=[
            "profile.json",
            "latency_burst_samples.jsonl",
            "latency_burst_model.json",
        ],
    )


def _profile_path_in_repo(*, root: Path, profile_dir: Path) -> str:
    raw_relative = _raw_relative_path(profile_dir)
    if raw_relative is not None:
        return raw_relative
    return profile_dir.relative_to(root).as_posix()


def _raw_relative_path(path: Path) -> str | None:
    parts = path.parts
    if "_hf_raw" not in parts:
        return None
    index = parts.index("_hf_raw")
    return Path(*parts[index + 1:]).as_posix()


def _measure_run_index(parts: tuple[str, ...]) -> int | None:
    for index, part in enumerate(parts):
        if part == "runs" and index + 1 < len(parts) and parts[index + 1] == "measure":
            return index
    return None


def _target_summary(target: BackfillTarget, *, apply: bool, upload: bool) -> dict:
    action_records = load_jsonl_records(target.actions_path)
    metrics_records = load_jsonl_records(target.metrics_path)
    return {
        "run_dir": str(target.run_dir),
        "actions_path": str(target.actions_path),
        "metrics_path": str(target.metrics_path),
        "profile_path": str(target.profile_path),
        "action_count": len(action_records),
        "episode_count": len(metrics_records),
        "metric_action_count": sum(int(record["num_actions"]) for record in metrics_records),
        "apply": apply,
        "upload": upload,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill semi-Markov latency burst profile artifacts.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=os.environ.get("HF_PROFILE_REPO"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"))
    args = parser.parse_args(argv)

    if args.upload and not args.apply:
        raise ValueError("--upload requires --apply")

    targets = discover_targets(args.root)
    for target in targets:
        print(json.dumps(_target_summary(target, apply=bool(args.apply), upload=bool(args.upload)), sort_keys=True))
        if args.apply:
            apply_backfill(target)
        if args.upload:
            upload_target(root=args.root, target=target, repo_id=args.repo_id, repo_type=args.repo_type)


if __name__ == "__main__":
    main()
