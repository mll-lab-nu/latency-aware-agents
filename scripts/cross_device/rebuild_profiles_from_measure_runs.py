from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import load_latency_samples, load_profile_manifest
from latency_bench.utils.io import load_jsonl_records
from scripts.cross_device.profile_cache import upload_folder_to_hf


PROFILE_GENERATED_KEYS = {
    "profile_id",
    "policy_id",
    "env_id",
    "target_device",
    "env_fps",
    "frame_ms",
    "n_samples",
    "trace_path",
    "latency_kind",
    "distribution_path",
    "plot_path",
    "summary",
    "inference_trace_path",
    "inference_summary",
    "e2e_trace_path",
    "e2e_summary",
    "inference_worker_count",
    "latency_method",
    "sample_model_type",
    "burst_samples_path",
    "burst_model_path",
    "burst_state_key",
}


@dataclass(frozen=True)
class RebuildTarget:
    run_dir: Path
    e2e_latencies_path: Path
    inference_latencies_path: Path
    actions_path: Path
    steps_path: Path
    episode_metrics_path: Path
    profile_path: Path
    profile_id: str
    policy_id: str
    env_id: str
    target_device: str
    env_fps: float
    metadata: dict


def discover_targets(*, root: Path, run_dir: Path | None = None) -> list[RebuildTarget]:
    run_dirs = [run_dir] if run_dir is not None else [path.parent for path in sorted(root.rglob("episode_metrics.jsonl"))]
    targets: list[RebuildTarget] = []
    for candidate in run_dirs:
        parts = candidate.parts
        run_index = _measure_run_index(parts)
        if run_index is None:
            continue
        episode_metrics_path = candidate / "episode_metrics.jsonl"
        records = load_jsonl_records(episode_metrics_path)
        metrics_metadata = records[0]["metadata"]
        profile_id = metrics_metadata["profile_id"]
        prefix = Path(*parts[:run_index])
        policy_slug = parts[run_index + 2]
        env_slug = parts[run_index + 3]
        profile_path = prefix / "profiles" / policy_slug / env_slug / profile_id / "profile.json"
        existing_profile = load_profile_manifest(profile_path)
        metadata = {
            key: value
            for key, value in existing_profile.items()
            if key not in PROFILE_GENERATED_KEYS
        }
        targets.append(
            RebuildTarget(
                run_dir=candidate,
                e2e_latencies_path=candidate / "e2e_latencies.jsonl",
                inference_latencies_path=candidate / "infer_latencies.jsonl",
                actions_path=candidate / "actions.jsonl",
                steps_path=candidate / "steps.jsonl",
                episode_metrics_path=episode_metrics_path,
                profile_path=profile_path,
                profile_id=profile_id,
                policy_id=existing_profile["policy_id"],
                env_id=existing_profile["env_id"],
                target_device=existing_profile["target_device"],
                env_fps=float(metrics_metadata["env_fps"]),
                metadata=metadata,
            )
        )
    return targets


def rebuild_target(target: RebuildTarget) -> None:
    build_profile_from_trace(
        latencies=target.e2e_latencies_path,
        output_dir=target.profile_path.parent,
        profile_id=target.profile_id,
        policy_id=target.policy_id,
        env_id=target.env_id,
        target_device=target.target_device,
        env_fps=target.env_fps,
        metadata=target.metadata,
        latency_kind="e2e_latency",
        inference_latencies=target.inference_latencies_path,
        episode_metrics=target.episode_metrics_path,
        actions=target.actions_path,
        steps=target.steps_path,
        overwrite=True,
    )


def upload_target(*, root: Path, target: RebuildTarget, repo_id: str, repo_type: str) -> None:
    upload_folder_to_hf(
        repo_id=repo_id,
        local_dir=target.profile_path.parent,
        repo_type=repo_type,
        path_in_repo=_profile_path_in_repo(root=root, profile_dir=target.profile_path.parent),
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


def _target_summary(target: RebuildTarget, *, apply: bool, upload: bool) -> dict:
    return {
        "run_dir": str(target.run_dir),
        "profile_path": str(target.profile_path),
        "e2e_latencies_path": str(target.e2e_latencies_path),
        "inference_latencies_path": str(target.inference_latencies_path),
        "actions_path": str(target.actions_path),
        "steps_path": str(target.steps_path),
        "episode_metrics_path": str(target.episode_metrics_path),
        "e2e_latency_count": len(load_latency_samples(target.e2e_latencies_path)),
        "inference_latency_count": len(load_latency_samples(target.inference_latencies_path)),
        "profile_id": target.profile_id,
        "target_device": target.target_device,
        "env_fps": target.env_fps,
        "apply": apply,
        "upload": upload,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild cross-device profiles from completed measure runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=os.environ.get("HF_PROFILE_REPO"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"))
    args = parser.parse_args(argv)

    if args.upload and not args.apply:
        raise ValueError("--upload requires --apply")

    targets = discover_targets(root=args.root, run_dir=args.run_dir)
    for target in targets:
        print(json.dumps(_target_summary(target, apply=bool(args.apply), upload=bool(args.upload)), sort_keys=True))
        if args.apply:
            rebuild_target(target)
        if args.upload:
            upload_target(root=args.root, target=target, repo_id=args.repo_id, repo_type=args.repo_type)


if __name__ == "__main__":
    main()
