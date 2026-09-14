#!/usr/bin/env python3
"""Export the G1 Balance Simple checkpoint teacher for NVIDIA Isaac-GR00T."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.data.official_gr00t import convert_canonical_rollout_to_official_gr00t
from latency_bench.data.sf_teacher_export import export_sample_factory_teacher_dataset

DEFAULT_MANIFEST = REPO_ROOT / "configs/teachers/g1_sonic_gr00t.json"


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _selected_jobs(
    manifest: dict[str, Any], selected_names: Sequence[str] | None
) -> list[dict[str, Any]]:
    jobs = manifest["teachers"]
    if not selected_names:
        return list(jobs)
    jobs_by_name = {job["name"]: job for job in jobs}
    return [jobs_by_name[name] for name in selected_names]


def _job_dirs(output_root: Path, name: str) -> tuple[Path, Path]:
    base = output_root / name
    return base / "raw", base / "lerobot"


def export_checkpoint_teacher(
    *,
    job: dict[str, Any],
    raw_output_dir: Path,
    official_output_dir: Path,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
    num_envs: int,
    replay_num_envs: int,
    device: str,
    deterministic: bool,
    export_image_size: tuple[int, int] | None,
    image_writer_workers: int,
    row_buffer_size: int,
    context_window: int,
    latency_config_path: str | Path | None,
    balance_teacher_observation_state: bool,
    convert: bool,
    force: bool,
) -> dict[str, Any]:
    raw_output_dir = raw_output_dir.expanduser().resolve()
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    rollout = export_sample_factory_teacher_dataset(
        checkpoint_root=job["checkpoint"],
        output_dir=raw_output_dir,
        latency_config_path=latency_config_path,
        gym_camera_name="head_camera",
        env_name="",
        max_episodes=max_episodes,
        max_steps_per_episode=job["max_steps_per_episode"],
        val_fraction=val_fraction,
        split_seed=split_seed,
        num_envs=num_envs,
        replay_num_envs=replay_num_envs,
        device=device,
        deterministic=deterministic,
        export_image_size=export_image_size,
        image_writer_workers=image_writer_workers,
        row_buffer_size=row_buffer_size,
        push_to_hub=False,
        hf_repo_id="",
        hf_private=False,
        hf_token="",
        hf_max_shard_size="500MB",
        filter_preset="none",
        episode_acceptance=job["acceptance"],
        context_window=context_window,
        gym_gr00t_balance_teacher_observation_state=(
            balance_teacher_observation_state
        ),
    )
    result: dict[str, Any] = {
        "name": job["name"],
        "task": job["task"],
        "source": job["source"],
        "rollout": rollout,
    }
    if convert:
        result["official_gr00t"] = convert_canonical_rollout_to_official_gr00t(
            raw_output_dir,
            official_output_dir.expanduser().resolve(),
            force=force,
        )
    return result


def _parse_image_size(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    width, height = (int(item.strip()) for item in value.split(","))
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher", action="append", dest="teachers")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="override the selected checkpoint teacher's manifest path",
    )
    parser.add_argument("--no-convert", dest="convert", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replay-num-envs", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--export-image-size", default="")
    parser.add_argument("--image-writer-workers", type=int, default=0)
    parser.add_argument("--row-buffer-size", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--latency-config", default="")
    parser.add_argument("--balance-teacher-observation-state", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = _selected_jobs(_load_manifest(args.manifest), args.teachers)
    if args.checkpoint is not None:
        if len(jobs) != 1:
            raise ValueError("--checkpoint requires exactly one checkpoint teacher")
        jobs[0] = {**jobs[0], "checkpoint": str(args.checkpoint)}
    output_root = args.output_root.expanduser().resolve()
    image_size = _parse_image_size(args.export_image_size)
    results = []
    for job in jobs:
        raw_dir, official_dir = _job_dirs(output_root, job["name"])
        result = export_checkpoint_teacher(
            job=job,
            raw_output_dir=raw_dir,
            official_output_dir=official_dir,
            max_episodes=args.max_episodes,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            num_envs=args.num_envs,
            replay_num_envs=args.replay_num_envs,
            device=args.device,
            deterministic=args.deterministic,
            export_image_size=image_size,
            image_writer_workers=args.image_writer_workers,
            row_buffer_size=args.row_buffer_size,
            context_window=args.context_window,
            latency_config_path=args.latency_config or None,
            balance_teacher_observation_state=(
                args.balance_teacher_observation_state
            ),
            convert=args.convert,
            force=args.force,
        )
        results.append(result)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
