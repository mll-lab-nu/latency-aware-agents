from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.utils.io import load_jsonl_records, write_json, write_jsonl
from scripts.cross_device.normalize_raw_instance_layout import apply_migration, discover_migration_actions
from scripts.cross_device.profile_cache import upload_folder_to_hf
from scripts.cross_device.rebuild_profiles_from_measure_runs import PROFILE_GENERATED_KEYS


PROFILE_AGGREGATION_RAW_IGNORE = [
    "**/sample_batches/**",
    "**/*.png",
    "**/warmup_*",
    "**/generated_configs/**",
    "**/workers/**",
    "**/*.warmup.yaml",
    "**/*.calibration.yaml",
]


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    policy_slug: str
    env_slug: str
    profile_id: str
    target_device: str
    instance_id: str
    runtime_alias: str
    e2e_latencies_path: Path
    inference_latencies_path: Path
    actions_path: Path
    episode_metrics_path: Path


def aggregate_profiles(
    *,
    raw_root: Path,
    profiles_dir: Path,
    work_dir: Path,
    upload: bool,
    repo_id: str | None,
    repo_type: str,
    only_policy: str | None = None,
    only_env: str | None = None,
    only_device: str | None = None,
) -> dict[str, Any]:
    actions = discover_migration_actions(raw_root)
    if actions:
        apply_migration(raw_root, actions)

    accepted, excluded = discover_source_runs(
        raw_root=raw_root,
        only_policy=only_policy,
        only_env=only_env,
        only_device=only_device,
    )
    groups: dict[tuple[str, str, str, str], list[SourceRun]] = {}
    for source in accepted:
        key = (source.policy_slug, source.env_slug, source.profile_id, source.target_device)
        groups.setdefault(key, []).append(source)

    rebuilt = []
    for key in sorted(groups):
        sources = groups[key]
        rebuilt.append(
            rebuild_group(
                profiles_dir=profiles_dir,
                work_dir=work_dir,
                sources=sources,
                upload=upload,
                repo_id=repo_id,
                repo_type=repo_type,
            )
        )

    report = {
        "accepted_run_count": len(accepted),
        "excluded_run_count": len(excluded),
        "excluded_runs": excluded,
        "rebuilt_profiles": rebuilt,
    }
    write_json(work_dir / "profile_aggregation_report.json", report)
    return report


def discover_source_runs(
    *,
    raw_root: Path,
    only_policy: str | None = None,
    only_env: str | None = None,
    only_device: str | None = None,
) -> tuple[list[SourceRun], list[dict[str, str]]]:
    accepted: list[SourceRun] = []
    excluded: list[dict[str, str]] = []
    for metrics_path in sorted((raw_root / "raw").glob("*/instances/*/*/cross_device/runs/measure/**/episode_metrics.jsonl")):
        run_dir = metrics_path.parent
        parsed = _parse_run_dir(raw_root=raw_root, run_dir=run_dir)
        if only_policy is not None and parsed["policy_slug"] != only_policy:
            continue
        if only_env is not None and parsed["env_slug"] != only_env:
            continue
        if only_device is not None and parsed["target_device"] != only_device:
            continue
        missing = [
            name
            for name in ("e2e_latencies.jsonl", "infer_latencies.jsonl", "actions.jsonl")
            if not (run_dir / name).exists()
        ]
        if missing:
            excluded.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "missing " + ",".join(missing),
                }
            )
            continue
        metrics_records = load_jsonl_records(metrics_path)
        metadata = metrics_records[0]["metadata"]
        accepted.append(
            SourceRun(
                run_dir=run_dir,
                policy_slug=parsed["policy_slug"],
                env_slug=parsed["env_slug"],
                profile_id=str(metadata["profile_id"]),
                target_device=str(metadata["target_device"]),
                instance_id=parsed["instance_id"],
                runtime_alias=parsed["runtime_alias"],
                e2e_latencies_path=run_dir / "e2e_latencies.jsonl",
                inference_latencies_path=run_dir / "infer_latencies.jsonl",
                actions_path=run_dir / "actions.jsonl",
                episode_metrics_path=metrics_path,
            )
        )
    return accepted, excluded


def rebuild_group(
    *,
    profiles_dir: Path,
    work_dir: Path,
    sources: list[SourceRun],
    upload: bool,
    repo_id: str | None,
    repo_type: str,
) -> dict[str, Any]:
    first = sources[0]
    profile_dir = profiles_dir / first.policy_slug / first.env_slug / first.profile_id
    existing_profile = load_profile_manifest(profile_dir / "profile.json")
    group_work_dir = work_dir / first.policy_slug / first.env_slug / first.profile_id
    combined = _write_combined_inputs(sources=sources, output_dir=group_work_dir)
    source_instances = sorted({source.instance_id for source in sources})
    metadata = {
        key: value
        for key, value in existing_profile.items()
        if key not in PROFILE_GENERATED_KEYS
    }
    metadata.update(
        {
            "profile_scope": "hardware_class_aggregate",
            "source_instance_count": len(source_instances),
            "source_run_count": len(sources),
            "source_instances": source_instances,
            "source_runs": [
                {
                    "run_dir": str(source.run_dir),
                    "instance_id": source.instance_id,
                    "runtime_alias": source.runtime_alias,
                }
                for source in sources
            ],
        }
    )
    build_profile_from_trace(
        latencies=combined["e2e_latencies"],
        output_dir=profile_dir,
        profile_id=first.profile_id,
        policy_id=str(existing_profile["policy_id"]),
        env_id=str(existing_profile["env_id"]),
        target_device=first.target_device,
        env_fps=float(existing_profile["env_fps"]),
        metadata=metadata,
        latency_kind="e2e_latency",
        inference_latencies=combined["infer_latencies"],
        episode_metrics=combined["episode_metrics"],
        actions=combined["actions"],
        overwrite=True,
    )
    if upload:
        upload_folder_to_hf(
            repo_id=repo_id,
            local_dir=profile_dir,
            repo_type=repo_type,
            path_in_repo=(Path("profiles") / first.policy_slug / first.env_slug / first.profile_id).as_posix(),
        )
    return {
        "profile_id": first.profile_id,
        "target_device": first.target_device,
        "policy": first.policy_slug,
        "env": first.env_slug,
        "profile_dir": str(profile_dir),
        "accepted_run_count": len(sources),
        "source_instances": source_instances,
    }


def _write_combined_inputs(*, sources: list[SourceRun], output_dir: Path) -> dict[str, Path]:
    e2e_records: list[dict[str, Any]] = []
    infer_records: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    aggregate_episode_id = 0
    for source in sources:
        e2e_records.extend(load_jsonl_records(source.e2e_latencies_path))
        infer_records.extend(load_jsonl_records(source.inference_latencies_path))
        source_metrics = load_jsonl_records(source.episode_metrics_path)
        source_actions = load_jsonl_records(source.actions_path)
        action_cursor = 0
        for metric in source_metrics:
            num_actions = int(metric["num_actions"])
            source_episode_id = int(metric["episode_id"])
            metric_copy = dict(metric)
            metric_metadata = dict(metric["metadata"])
            metric_metadata.update(
                {
                    "source_instance_id": source.instance_id,
                    "source_run_dir": str(source.run_dir),
                    "source_episode_id": source_episode_id,
                }
            )
            metric_copy["episode_id"] = aggregate_episode_id
            metric_copy["metadata"] = metric_metadata
            metric_records.append(metric_copy)
            for action in source_actions[action_cursor:action_cursor + num_actions]:
                action_copy = dict(action)
                action_metadata = dict(action["metadata"])
                action_metadata.update(
                    {
                        "source_instance_id": source.instance_id,
                        "source_run_dir": str(source.run_dir),
                        "source_episode_id": source_episode_id,
                    }
                )
                action_copy["metadata"] = action_metadata
                action_records.append(action_copy)
            action_cursor += num_actions
            aggregate_episode_id += 1
        if action_cursor != len(source_actions):
            raise ValueError(
                f"episode_metrics num_actions sum {action_cursor} does not match actions count {len(source_actions)}"
            )

    paths = {
        "e2e_latencies": output_dir / "e2e_latencies.jsonl",
        "infer_latencies": output_dir / "infer_latencies.jsonl",
        "episode_metrics": output_dir / "episode_metrics.jsonl",
        "actions": output_dir / "actions.jsonl",
    }
    write_jsonl(paths["e2e_latencies"], e2e_records)
    write_jsonl(paths["infer_latencies"], infer_records)
    write_jsonl(paths["episode_metrics"], metric_records)
    write_jsonl(paths["actions"], action_records)
    return paths


def _parse_run_dir(*, raw_root: Path, run_dir: Path) -> dict[str, str]:
    relative = run_dir.relative_to(raw_root)
    parts = relative.parts
    instances_index = parts.index("instances")
    runs_index = _runs_measure_index(parts)
    return {
        "instance_id": parts[instances_index + 1],
        "runtime_alias": parts[instances_index + 2],
        "policy_slug": parts[runs_index + 2],
        "env_slug": parts[runs_index + 3],
        "target_device": parts[runs_index + 4],
    }


def _runs_measure_index(parts: tuple[str, ...]) -> int:
    for index, part in enumerate(parts):
        if part == "runs" and parts[index + 1] == "measure":
            return index
    raise ValueError("measure run path must contain runs/measure")


def download_raw_tree_for_profile_aggregation(
    *,
    repo_id: str,
    repo_type: str,
    local_root: Path,
    revision: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=["raw/**"],
            ignore_patterns=PROFILE_AGGREGATION_RAW_IGNORE,
        )
    )
    local_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot_dir / "raw", local_root / "raw", dirs_exist_ok=True)
    return local_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-instance raw measure runs into hardware-class profiles.")
    parser.add_argument("--root", type=Path, default=Path("results/cross_device"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--repo-id", default=os.environ.get("HF_PROFILE_REPO"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"))
    parser.add_argument("--revision", default=os.environ.get("HF_PROFILE_REVISION"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only-policy")
    parser.add_argument("--only-env")
    parser.add_argument("--only-device")
    args = parser.parse_args(argv)

    raw_root = args.raw_root
    if raw_root is None:
        if args.no_download:
            raise ValueError("--raw-root is required with --no-download")
        raw_root = download_raw_tree_for_profile_aggregation(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_root=args.root / "_hf_raw",
            revision=args.revision,
        )
    profiles_dir = args.profiles_dir if args.profiles_dir is not None else args.root / "profiles"
    work_dir = args.work_dir if args.work_dir is not None else args.root / "_aggregate_profiles_work"
    report = aggregate_profiles(
        raw_root=raw_root,
        profiles_dir=profiles_dir,
        work_dir=work_dir,
        upload=bool(args.upload),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        only_policy=args.only_policy,
        only_env=args.only_env,
        only_device=args.only_device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
