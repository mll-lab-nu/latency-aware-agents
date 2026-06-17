from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.latency.profile import load_profile_manifest, save_profile_manifest
from latency_bench.utils.io import load_jsonl_records, read_json, write_jsonl
from scripts.cross_device.profile_cache import upload_folder_to_hf


@dataclass(frozen=True)
class BackfillTarget:
    run_dir: Path
    metrics_path: Path
    latencies_path: Path
    profile_path: Path
    worker_count: int


def infer_worker_count(latencies_path: Path) -> int:
    records = load_jsonl_records(latencies_path)
    worker_counts = set()
    worker_ids = set()
    for line_number, record in enumerate(records, start=1):
        try:
            metadata = record["metadata"]
            worker_counts.add(metadata["worker_cuda_device_count"])
            worker_ids.add(metadata["worker_id"])
        except KeyError as exc:
            raise KeyError(f"missing {exc.args[0]} at {latencies_path}:{line_number}") from exc
    if len(worker_counts) != 1:
        raise ValueError(f"inconsistent worker_cuda_device_count in {latencies_path}: {sorted(worker_counts)}")
    worker_count = worker_counts.pop()
    missing_worker_ids = sorted(set(range(worker_count)) - worker_ids)
    if missing_worker_ids:
        raise ValueError(f"latency trace did not use workers {missing_worker_ids}: {latencies_path}")
    return worker_count


def discover_targets(root: Path) -> list[BackfillTarget]:
    targets: list[BackfillTarget] = []
    for metrics_path in sorted(root.rglob("episode_metrics.jsonl")):
        run_dir = metrics_path.parent
        parts = run_dir.parts
        run_index = _measure_run_index(parts)
        if run_index is None:
            continue
        latencies_path = run_dir / "latencies.jsonl"
        records = load_jsonl_records(metrics_path)
        profile_id = records[0]["metadata"]["profile_id"]
        prefix = Path(*parts[:run_index])
        policy = parts[run_index + 2]
        env = parts[run_index + 3]
        profile_path = prefix / "profiles" / policy / env / profile_id / "profile.json"
        targets.append(
            BackfillTarget(
                run_dir=run_dir,
                metrics_path=metrics_path,
                latencies_path=latencies_path,
                profile_path=profile_path,
                worker_count=infer_worker_count(latencies_path),
            )
        )
    return targets


def apply_backfill(target: BackfillTarget) -> None:
    metrics_records = load_jsonl_records(target.metrics_path)
    for record in metrics_records:
        record["metadata"]["inference_worker_count"] = target.worker_count
    write_jsonl(target.metrics_path, metrics_records)

    profile = load_profile_manifest(target.profile_path)
    profile["inference_worker_count"] = target.worker_count
    save_profile_manifest(profile, target.profile_path)


def upload_target(*, root: Path, target: BackfillTarget, repo_id: str, repo_type: str) -> None:
    for local_dir, path_in_repo, allow_patterns in _upload_specs(root=root, target=target):
        upload_folder_to_hf(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            allow_patterns=allow_patterns,
        )


def _upload_specs(root: Path, target: BackfillTarget) -> list[tuple[Path, str, list[str]]]:
    raw_run_path = _raw_relative_path(target.run_dir)
    raw_profile_path = _raw_relative_path(target.profile_path.parent)
    if raw_run_path is not None and raw_profile_path is not None:
        return [
            (target.run_dir, raw_run_path, ["episode_metrics.jsonl"]),
            (target.profile_path.parent, raw_profile_path, ["profile.json"]),
        ]

    manifest = read_json(target.run_dir / "manifest.json")
    runtime = manifest["runtime_env_matched_alias"]
    base = Path("raw") / manifest["hardware_normalized"] / runtime / "cross_device"
    run_relative = target.run_dir.relative_to(root)
    profile_relative = target.profile_path.parent.relative_to(root)
    return [
        (target.run_dir, (base / run_relative).as_posix(), ["episode_metrics.jsonl"]),
        (target.profile_path.parent, profile_relative.as_posix(), ["profile.json"]),
        (target.profile_path.parent, (base / profile_relative).as_posix(), ["profile.json"]),
    ]


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill inference_worker_count into old cross-device artifacts.")
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
        print(
            json.dumps(
                {
                    "run_dir": str(target.run_dir),
                    "metrics_path": str(target.metrics_path),
                    "profile_path": str(target.profile_path),
                    "inference_worker_count": target.worker_count,
                    "apply": bool(args.apply),
                    "upload": bool(args.upload),
                },
                sort_keys=True,
            )
        )
        if args.apply:
            apply_backfill(target)
        if args.upload:
            upload_target(
                root=args.root,
                target=target,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
            )


if __name__ == "__main__":
    main()
