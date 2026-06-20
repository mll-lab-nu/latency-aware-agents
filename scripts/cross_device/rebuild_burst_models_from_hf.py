"""One-time: rebuild curated device profiles to the v3 bayes-Markov burst model from HF.

Every curated `profiles/<policy>/<env>/<profile_id>/` on the HF repo is either missing
a burst model or carries an old v1/v2 schema the current `BayesMarkovSampler` cannot
read, so `profile_default` cannot resolve `temporal`. This script pulls each device's
canonical measure run from the repo's `raw/` tree, fully rebuilds the curated profile
(distribution + summary + plot + v3 burst model) via `build_profile_from_trace`, and
re-uploads it to the curated top-level path that fresh boxes download from.

Canonical run per `profile_id`: prefer the `_s50000` profiling run when it carries all
four rebuild inputs (actions / episode_metrics / e2e_latencies / infer_latencies), else
`best_seed_0_n100`; profiles with no qualifying run are skipped (e.g. L_3090, whose
measure actions were never uploaded). `env_fps` is taken from the canonical run's
episode metrics, which corrects devices whose curated profile was built from a bad run
(e.g. rtxpro6000blackwell calibrated to 15 fps from a throttled 94 ms run instead of
38 fps from the 40 ms run).
"""
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
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.utils.io import load_jsonl_records
from scripts.cross_device.profile_cache import upload_folder_to_hf
from scripts.cross_device.rebuild_profiles_from_measure_runs import PROFILE_GENERATED_KEYS

REBUILD_INPUTS = ("actions.jsonl", "episode_metrics.jsonl", "e2e_latencies.jsonl", "infer_latencies.jsonl")


@dataclass(frozen=True)
class RebuildTarget:
    profile_id: str
    profile_dir: Path
    run_dir: Path
    env_fps: float
    policy_id: str
    env_id: str
    target_device: str
    metadata: dict


def _measure_run_index(parts: tuple[str, ...]) -> int | None:
    for index, part in enumerate(parts):
        if part == "runs" and index + 1 < len(parts) and parts[index + 1] == "measure":
            return index
    return None


def _canonical_runs_by_profile(raw_root: Path) -> dict[str, Path]:
    """Map each profile_id to its canonical measure run dir (prefer `_s50000`)."""
    by_profile: dict[str, Path] = {}
    for metrics_path in sorted(raw_root.rglob("episode_metrics.jsonl")):
        run_dir = metrics_path.parent
        if _measure_run_index(run_dir.parts) is None:
            continue
        if not all((run_dir / name).exists() for name in REBUILD_INPUTS):
            continue
        profile_id = load_jsonl_records(metrics_path)[0]["metadata"]["profile_id"]
        current = by_profile.get(profile_id)
        if current is None or (run_dir.name.endswith("_s50000") and not current.name.endswith("_s50000")):
            by_profile[profile_id] = run_dir
    return by_profile


def discover_targets(snapshot: Path, *, policy: str, env: str) -> tuple[list[RebuildTarget], list[str]]:
    canonical = _canonical_runs_by_profile(snapshot / "raw")
    targets: list[RebuildTarget] = []
    skipped: list[str] = []
    for profile_path in sorted((snapshot / "profiles" / policy / env).glob("*/profile.json")):
        profile_id = profile_path.parent.name
        run_dir = canonical.get(profile_id)
        if run_dir is None:
            skipped.append(profile_id)
            continue
        profile = load_profile_manifest(profile_path)
        env_fps = float(load_jsonl_records(run_dir / "episode_metrics.jsonl")[0]["metadata"]["env_fps"])
        metadata = {key: value for key, value in profile.items() if key not in PROFILE_GENERATED_KEYS}
        targets.append(
            RebuildTarget(
                profile_id=profile_id,
                profile_dir=profile_path.parent,
                run_dir=run_dir,
                env_fps=env_fps,
                policy_id=profile["policy_id"],
                env_id=profile["env_id"],
                target_device=profile["target_device"],
                metadata=metadata,
            )
        )
    return targets, skipped


def rebuild_target(target: RebuildTarget, *, work_dir: Path) -> Path:
    output_dir = work_dir / target.profile_id
    build_profile_from_trace(
        latencies=target.run_dir / "e2e_latencies.jsonl",
        output_dir=output_dir,
        profile_id=target.profile_id,
        policy_id=target.policy_id,
        env_id=target.env_id,
        target_device=target.target_device,
        env_fps=target.env_fps,
        metadata=target.metadata,
        latency_kind="e2e_latency",
        inference_latencies=target.run_dir / "infer_latencies.jsonl",
        episode_metrics=target.run_dir / "episode_metrics.jsonl",
        actions=target.run_dir / "actions.jsonl",
        overwrite=True,
    )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild curated device profiles to v3 bayes-Markov from the HF repo.")
    parser.add_argument("--apply", action="store_true", help="Rebuild profiles locally under --work-dir.")
    parser.add_argument("--upload", action="store_true", help="Upload rebuilt profiles to the curated HF path (requires --apply).")
    parser.add_argument("--policy", default="openvla-flappy")
    parser.add_argument("--env", default="flappy")
    parser.add_argument("--work-dir", type=Path, default=Path("results/cross_device/_rebuild_staging"))
    parser.add_argument("--repo-id", default=os.environ.get("HF_PROFILE_REPO"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"))
    parser.add_argument("--revision")
    args = parser.parse_args(argv)

    if args.upload and not args.apply:
        raise ValueError("--upload requires --apply")
    if args.repo_id is None:
        raise ValueError("--repo-id is required (or set HF_PROFILE_REPO)")

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=[
                f"profiles/{args.policy}/{args.env}/**",
                *[f"raw/**/runs/measure/**/{name}" for name in REBUILD_INPUTS],
            ],
        )
    )

    targets, skipped = discover_targets(snapshot, policy=args.policy, env=args.env)
    for profile_id in skipped:
        print(f"SKIP {profile_id} (no canonical measure run with rebuild inputs on HF; needs re-measure)")
    for target in targets:
        print(json.dumps(
            {
                "profile_id": target.profile_id,
                "canonical_run": target.run_dir.name,
                "env_fps": target.env_fps,
                "apply": bool(args.apply),
                "upload": bool(args.upload),
            },
            sort_keys=True,
        ))
        if args.apply:
            output_dir = rebuild_target(target, work_dir=args.work_dir)
            if args.upload:
                upload_folder_to_hf(
                    repo_id=args.repo_id,
                    local_dir=output_dir,
                    repo_type=args.repo_type,
                    path_in_repo=f"profiles/{args.policy}/{args.env}/{target.profile_id}",
                )


if __name__ == "__main__":
    main()
