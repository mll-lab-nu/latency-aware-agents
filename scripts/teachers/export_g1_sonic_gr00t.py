#!/usr/bin/env python3
"""Export manifest-defined G1 SONIC teachers for NVIDIA Isaac-GR00T."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.data.action_layouts import G1_SONIC_GR00T_ACTION_SPEC
from latency_bench.data.filters import accepts_episode_contract
from latency_bench.data.hf_parquet import write_parquet_splits
from latency_bench.data.official_gr00t import (
    convert_canonical_rollout_to_official_gr00t,
)
from latency_bench.data.sf_teacher_export import export_sample_factory_teacher_dataset
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
    G1_GR00T_STATE_DIM,
)

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


def export_reference_playback_dataset(
    *,
    job: dict[str, Any],
    raw_output_dir: Path,
    official_output_dir: Path,
    max_episodes: int,
    val_fraction: float,
    split_seed: int,
    convert: bool,
    force: bool,
) -> dict[str, Any]:
    """Probe accepted reference episodes, then replay only those with images."""
    config = yaml.safe_load(
        Path(job["playback_config"]).expanduser().read_text(encoding="utf-8")
    )
    env_config = config["env"]
    for module_name in env_config["registration_imports"]:
        importlib.import_module(module_name)
    env = gym.make(env_config["env_id"], **env_config["make_kwargs"])
    raw_output_dir = raw_output_dir.expanduser().resolve()
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    acceptance = job["acceptance"]
    env_fps = env_config["env_fps"]
    obs_fps = env_config["obs_fps"]
    max_steps_per_episode = job["max_steps_per_episode"]
    accepted_specs: list[dict[str, Any]] = []
    attempted_episodes = 0
    rejected_episodes = 0
    attempt_idx = 0
    try:
        while len(accepted_specs) < max_episodes:
            episode_seed = split_seed + attempt_idx
            env.reset(seed=episode_seed)
            episode_steps = 0
            for _ in range(max_steps_per_episode):
                _, _, terminated, truncated, final_info = env.step(
                    np.zeros(G1_GR00T_ACTION_DIM, dtype=np.float32)
                )
                episode_steps += 1
                if terminated or truncated:
                    break
            summary = {
                "episode_raw_frames": episode_steps,
                "final_info": final_info,
            }
            attempted_episodes += 1
            if accepts_episode_contract(summary, acceptance, env_fps=env_fps):
                accepted_specs.append(
                    {
                        "episode_idx": len(accepted_specs),
                        "seed": episode_seed,
                        "episode_raw_frames": episode_steps,
                    }
                )
            else:
                rejected_episodes += 1
            attempt_idx += 1

        validation_count = int(max_episodes * val_fraction)
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        for spec in accepted_specs:
            episode_idx = spec["episode_idx"]
            split = "val" if episode_idx >= max_episodes - validation_count else "train"
            episode_rows = val_rows if split == "val" else train_rows
            _, reset_info = env.reset(seed=spec["seed"])
            state = reset_info["official_gr00t_proprio"]
            target = reset_info["action_target"]
            for decision_step in range(max_steps_per_episode):
                image = np.asarray(env.render())
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(G1_GR00T_ACTION_DIM, dtype=np.float32)
                )
                image_path = (
                    raw_output_dir
                    / "images"
                    / split
                    / f"episode_{episode_idx:06d}"
                    / f"decision_step_{decision_step:06d}.png"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(image_path)
                episode_rows.append(
                    teacher_dataset_row(
                        env_name=job["task"],
                        spec={
                            "episode_idx": episode_idx,
                            "split": split,
                            "seed": spec["seed"],
                        },
                        decision_step=decision_step,
                        image=str(image_path),
                        raw_action=target,
                        info={"latency_raw_frames": 0, "latency_ms": 0.0},
                        reward=reward,
                        env_fps=env_fps,
                        obs_fps=obs_fps,
                        action_spec=G1_SONIC_GR00T_ACTION_SPEC,
                        base_prompt=env_config["base_prompt"],
                        state=state,
                    )
                )
                state = info["official_gr00t_proprio"]
                target = info["action_target"]
                if terminated or truncated:
                    break
    finally:
        env.close()

    write_parquet_splits(
        raw_output_dir,
        train_rows,
        val_rows,
        action_spec=G1_SONIC_GR00T_ACTION_SPEC,
        keep_dataset=False,
    )
    filter_report = {
        "preset": "none",
        "episode_acceptance": acceptance,
        "attempted_episodes": attempted_episodes,
        "accepted_episodes": len(accepted_specs),
        "rejected_episodes": rejected_episodes,
        "accepted_train_episodes": len(accepted_specs) - validation_count,
        "accepted_val_episodes": validation_count,
        "train_rows_before_row_filter": len(train_rows),
        "train_rows_after_row_filter": len(train_rows),
        "val_rows_before_row_filter": len(val_rows),
        "val_rows_after_row_filter": len(val_rows),
    }
    state_space = {
        "labels": [f"state_{index}" for index in range(G1_GR00T_STATE_DIM)]
    }
    metadata = {
        "env_name": job["task"],
        "integration_name": "gymnasium",
        "action_layout": G1_SONIC_GR00T_ACTION_SPEC.layout,
        "action_labels": list(G1_SONIC_GR00T_ACTION_SPEC.labels),
        "reward_field": G1_SONIC_GR00T_ACTION_SPEC.reward_field,
        "rows_unit": "decision_step",
        "obs_fps": obs_fps,
        "obs_stride_raw_frames": int(env_fps / obs_fps),
        "base_prompt": env_config["base_prompt"],
        "episode_acceptance": acceptance,
        "filter_report": filter_report,
        "gymnasium_task": {
            "task_name": job["task"],
            "env_id": env_config["env_id"],
            "registration_imports": env_config["registration_imports"],
            "make_kwargs": env_config["make_kwargs"],
            "env_fps": env_fps,
            "obs_fps": obs_fps,
            "frame_stack": env_config["frame_stack"],
            "base_prompt": env_config["base_prompt"],
            "state_space": state_space,
            "action_space": {
                "type": "box",
                "labels": list(G1_SONIC_GR00T_ACTION_SPEC.labels),
                "dtype": "float32",
            },
        },
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    (raw_output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (raw_output_dir / "filter_report.json").write_text(
        json.dumps(filter_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result: dict[str, Any] = {
        "name": job["name"],
        "task": job["task"],
        "source": job["source"],
        "rollout": metadata,
    }
    if convert:
        result["official_gr00t"] = convert_canonical_rollout_to_official_gr00t(
            raw_output_dir, official_output_dir, force=force
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = _selected_jobs(_load_manifest(args.manifest), args.teachers)
    output_root = args.output_root.expanduser().resolve()
    image_size = _parse_image_size(args.export_image_size)
    results = []
    for job in jobs:
        raw_dir, official_dir = _job_dirs(output_root, job["name"])
        if job["source"] == "checkpoint":
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
                convert=args.convert,
                force=args.force,
            )
        elif job["source"] == "reference_playback":
            result = export_reference_playback_dataset(
                job=job,
                raw_output_dir=raw_dir,
                official_output_dir=official_dir,
                max_episodes=args.max_episodes,
                val_fraction=args.val_fraction,
                split_seed=args.split_seed,
                convert=args.convert,
                force=args.force,
            )
        else:
            raise KeyError(job["source"])
        results.append(result)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
