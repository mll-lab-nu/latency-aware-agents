#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot import (
    _canonical_prompt,
    _episode_key,
    _episode_sort_key,
    _filter_latency,
    _image_shape,
    _load_index_split,
    _load_split,
    _normalize_action_carrier,
    _normalize_prompt_map,
    _png_bytes,
    _row_done,
    _row_latency,
    _select_episode_ids,
    _write_jsonl,
    _write_metadata,
)
from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
    build_latency_prompt_map,
    resolve_latency_subdirs,
)

BRIDGE_ACTION_DIM = 7
EpisodeKey = int | tuple[int, int]


def _parse_contract(value: dict[str, Any] | str | None) -> dict[str, Any]:
    if value in (None, ""):
        raise ValueError("gymnasium_task_contract is required for Gymnasium conversion")
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    return json.loads(str(value))


def _active_action_labels(contract: dict[str, Any]) -> list[str]:
    if "action_labels" in contract:
        return [str(value) for value in contract["action_labels"]]
    action_space = contract.get("action_space") or {}
    if "labels" in action_space:
        return [str(value) for value in action_space["labels"]]
    raise ValueError("gymnasium_task_contract must contain action_labels or action_space.labels")


def _action_values(contract: dict[str, Any]) -> list[Any]:
    if "action_values" in contract:
        return list(contract["action_values"])
    action_space = contract.get("action_space") or {}
    if "values" in action_space:
        return list(action_space["values"])
    labels = _active_action_labels(contract)
    return list(range(len(labels)))


def _carrier_action_labels(active_labels: list[str], action_carrier: str) -> list[str]:
    if action_carrier == "native":
        return list(active_labels)
    if len(active_labels) > BRIDGE_ACTION_DIM:
        raise ValueError(
            f"bridge action carrier supports at most {BRIDGE_ACTION_DIM} actions, got {len(active_labels)}"
        )
    return [
        *active_labels,
        *(f"BRIDGE_PAD_{idx}" for idx in range(len(active_labels), BRIDGE_ACTION_DIM)),
    ]


def _state_labels(contract: dict[str, Any]) -> list[str]:
    state_space = contract.get("state_space") or {}
    labels = state_space.get("labels") or []
    return [str(value) for value in labels]


def _one_hot(action_id: int, *, active_action_dim: int, action_dim: int) -> list[float]:
    if action_id < 0 or action_id >= active_action_dim:
        raise ValueError(
            f"action_id={action_id} is outside Gymnasium action range [0, {active_action_dim - 1}]"
        )
    values = [0.0] * action_dim
    values[action_id] = 1.0
    return values


def _row_state(row: dict[str, Any], *, state_dim: int) -> list[float]:
    if state_dim <= 0:
        return []
    if "state" not in row or row["state"] is None:
        raise KeyError("Gymnasium row is missing required `state` column")
    state = [float(value) for value in row["state"]]
    if len(state) != state_dim:
        raise ValueError(f"Gymnasium state length is {len(state)}, expected {state_dim}")
    return state


def _write_episode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    action_dim: int,
    state_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.image": pa.array(
            [{"bytes": row["image_bytes"], "path": None} for row in rows],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "observation.state": pa.array(
            [row["state"] for row in rows],
            type=pa.list_(pa.float32(), state_dim),
        ),
        "action": pa.array(
            [row["action"] for row in rows],
            type=pa.list_(pa.float32(), action_dim),
        ),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        "latency": pa.array([row["latency"] for row in rows], type=pa.int64()),
        "latency_raw_frames": pa.array([row["latency_raw_frames"] for row in rows], type=pa.int64()),
        "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
        "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
        "action_id": pa.array([row["action_id"] for row in rows], type=pa.int64()),
    }
    pq.write_table(pa.table(columns), path)


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    max_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
    latency_filter: list[int] | None = None,
    train_latency_filter: list[int] | None = None,
    eval_latency_filter: list[int] | None = None,
    episodes_per_latency: int | None = None,
    train_episodes_per_latency: int | None = None,
    eval_episodes_per_latency: int | None = None,
    episodes_per_latency_by_latency: dict[int, int] | None = None,
    train_episodes_per_latency_by_latency: dict[int, int] | None = None,
    eval_episodes_per_latency_by_latency: dict[int, int] | None = None,
    prompt_map_override: dict[str, Any] | dict[int, Any] | None = None,
    default_latency: int | None = None,
    action_carrier: str = "native",
    fps: float,
    obs_stride_raw_frames: int,
    source_latency_column: str | None,
    target_latency_unit: str,
    source_rows_unit: str,
    gymnasium_task_contract: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    if source_rows_unit != "decision_step":
        raise ValueError(f"source_rows_unit must be decision_step, got {source_rows_unit!r}")
    action_carrier = _normalize_action_carrier(action_carrier)
    contract = _parse_contract(gymnasium_task_contract)
    active_labels = _active_action_labels(contract)
    action_labels = _carrier_action_labels(active_labels, action_carrier)
    action_dim = len(action_labels)
    active_action_dim = len(active_labels)
    state_labels = _state_labels(contract)
    state_dim = len(state_labels)
    if state_dim <= 0:
        raise ValueError("Gymnasium conversion requires rl_games.gymnasium.task_contract.state_space.labels")
    prompt_map_override = _normalize_prompt_map(prompt_map_override)
    if train_latency_filter is None:
        train_latency_filter = latency_filter
    if eval_latency_filter is None:
        eval_latency_filter = latency_filter
    if train_episodes_per_latency is None:
        train_episodes_per_latency = episodes_per_latency
    if eval_episodes_per_latency is None:
        eval_episodes_per_latency = episodes_per_latency
    if train_episodes_per_latency_by_latency is None:
        train_episodes_per_latency_by_latency = episodes_per_latency_by_latency
    if eval_episodes_per_latency_by_latency is None:
        eval_episodes_per_latency_by_latency = episodes_per_latency_by_latency
    if max_steps_per_episode is not None and int(max_steps_per_episode) <= 0:
        raise ValueError(f"max_steps_per_episode must be positive, got {max_steps_per_episode!r}")
    max_steps_per_episode = (
        int(max_steps_per_episode) if max_steps_per_episode is not None else None
    )

    output_dir = Path(output_dir)
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)

    def _convert_split(
        split: str,
        split_output_dir: Path,
        *,
        split_latency_filter: list[int] | None,
        split_episodes_per_latency: int | None,
        split_episodes_per_latency_by_latency: dict[int, int] | None,
    ) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        want_latency = bool(
            source_latency_column is not None
            or require_latency_prompt_map
            or split_latency_filter
            or prompt_map_override
            or split_episodes_per_latency is not None
            or split_episodes_per_latency_by_latency is not None
        )
        ds_meta, source_columns = _load_index_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            want_latency=want_latency,
            source_latency_column=source_latency_column,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
            latencies=split_latency_filter,
        )
        ds_meta = _filter_latency(
            ds_meta,
            split_latency_filter,
            latency_column=source_columns.latency,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
            default_latency=default_latency,
        )
        if len(ds_meta) == 0:
            raise ValueError(f"{dataset_name} has no {split} rows")

        episode_indices: dict[EpisodeKey, list[tuple[int, int]]] = {}
        episode_latencies: dict[EpisodeKey, int] = {}
        for row_idx, row in enumerate(tqdm(ds_meta, desc=f"Indexing Gymnasium {split} rows")):
            episode_idx = int(row["episode_idx"])
            latency = _row_latency(
                row,
                latency_column=source_columns.latency,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
                default_latency=default_latency,
            )
            episode_key = _episode_key(episode_idx, latency)
            episode_indices.setdefault(episode_key, []).append((int(row[source_columns.frame]), row_idx))
            if latency is not None:
                existing = episode_latencies.setdefault(episode_key, int(latency))
                if existing != int(latency):
                    raise ValueError(f"episode_key={episode_key!r} has inconsistent latencies: {existing} and {latency}")

        original_episode_ids = sorted(episode_indices, key=_episode_sort_key)
        original_episode_ids = _select_episode_ids(
            original_episode_ids,
            episode_latencies,
            max_episodes=max_episodes,
            require_latency_prompt_map=require_latency_prompt_map,
            episodes_per_latency=split_episodes_per_latency,
            episodes_per_latency_by_latency=split_episodes_per_latency_by_latency,
        )
        for episode_id in original_episode_ids:
            episode_indices[episode_id].sort(key=lambda item: item[0])

        ds_full = _filter_latency(
            _load_split(
                dataset_name,
                split,
                cache_dir=cache_dir,
                dataset_config_name=dataset_config_name,
                dataset_source_subdir=dataset_source_subdir,
                latencies=split_latency_filter,
            ),
            split_latency_filter,
            latency_column=source_columns.latency,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
            default_latency=default_latency,
        )
        if "image" in ds_full.column_names:
            ds_full = ds_full.cast_column("image", datasets.Image(decode=False))

        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []
        episode_lengths: list[int] = []
        image_shape: list[int] | None = None
        state_min: np.ndarray | None = None
        state_max: np.ndarray | None = None

        for new_episode_idx, original_episode_idx in enumerate(
            tqdm(original_episode_ids, desc=f"Writing Gymnasium {split} LeRobot episodes")
        ):
            row_indices = [row_idx for _, row_idx in episode_indices[original_episode_idx]]
            truncated_episode = False
            if max_steps_per_episode is not None and len(row_indices) > max_steps_per_episode:
                row_indices = row_indices[:max_steps_per_episode]
                truncated_episode = True
            episode = ds_full.select(row_indices)
            out_rows: list[dict[str, Any]] = []
            for frame_idx, row in enumerate(episode):
                prompt, latency, latency_ms = _canonical_prompt(
                    row,
                    prompt_map=prompt_map_override,
                    latency_column=source_columns.latency,
                    latency_ms_column=source_columns.latency_ms,
                    target_latency_unit=target_latency_unit,
                    obs_stride_raw_frames=obs_stride_raw_frames,
                    default_latency=default_latency,
                )
                if prompt not in prompt_to_task_index:
                    prompt_to_task_index[prompt] = len(task_prompts)
                    task_prompts.append(prompt)
                if latency is not None:
                    raw_frames = (
                        int(row["latency_raw_frames"])
                        if "latency_raw_frames" in row and row["latency_raw_frames"] is not None
                        else int(latency) * int(obs_stride_raw_frames)
                    )
                    latency_rows.append({
                        "latency": latency,
                        "latency_raw_frames": raw_frames,
                        "latency_ms": latency_ms,
                        "prompt": prompt,
                    })
                image_bytes = _png_bytes(row["image"])
                if image_shape is None:
                    image_shape = _image_shape(image_bytes)
                latency_id = int(latency) if latency is not None else int(default_latency or 0)
                state = _row_state(row, state_dim=state_dim)
                state_array = np.asarray(state, dtype=np.float32)
                state_min = state_array if state_min is None else np.minimum(state_min, state_array)
                state_max = state_array if state_max is None else np.maximum(state_max, state_array)
                out_rows.append({
                    "image_bytes": image_bytes,
                    "state": state,
                    "action": _one_hot(
                        int(row["action_id"]),
                        active_action_dim=active_action_dim,
                        action_dim=action_dim,
                    ),
                    "timestamp": frame_idx / fps,
                    "episode_index": new_episode_idx,
                    "frame_index": frame_idx,
                    "task_index": prompt_to_task_index[prompt],
                    "latency": latency_id,
                    "latency_raw_frames": (
                        int(row["latency_raw_frames"])
                        if "latency_raw_frames" in row and row["latency_raw_frames"] is not None
                        else latency_id * int(obs_stride_raw_frames)
                    ),
                    "done": (
                        _row_done(
                            row,
                            source_columns.done,
                            frame_idx=frame_idx,
                            episode_length=len(episode),
                        )
                        or (truncated_episode and frame_idx == len(episode) - 1)
                    ),
                    "reward": float(row[source_columns.reward]),
                    "action_id": int(row["action_id"]),
                })
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            _write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
                action_dim=action_dim,
                state_dim=state_dim,
            )

        if image_shape is None:
            raise ValueError(f"{dataset_name} {split} split produced no image rows")
        _write_metadata(
            split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            action_dim=action_dim,
            action_labels=action_labels,
            state_dim=state_dim,
            state_labels=state_labels,
            context_images_output_column=None,
            image_sequence_length=None,
            image_shape=image_shape,
            fps=fps,
        )

        if latency_rows:
            latency_prompt_map = build_latency_prompt_map(
                latency_rows,
                latency_column="latency",
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
            (split_output_dir / "latency_prompt_map.json").write_text(
                json.dumps(latency_prompt_map, indent=2),
                encoding="utf-8",
            )
        elif require_latency_prompt_map:
            raise ValueError(f"{dataset_name} {split} split has no latency rows; cannot build latency_prompt_map.json")

        manifest = {
            "dataset_name": split_output_dir.name,
            "split": split,
            "source": dataset_name,
            "source_config": dataset_config_name,
            "source_subdir": dataset_source_subdir,
            "latency_subdirs": [str(s) for s in resolve_latency_subdirs(dataset_source_subdir, split_latency_filter)],
            "format": "starvla_lerobot_v2_image_parquet",
            "integration_name": "gymnasium",
            "task_name": contract["task_name"],
            "gymnasium_task": contract,
            "action_layout": "gymnasium_discrete_v1",
            "action_labels": action_labels,
            "active_action_labels": active_labels,
            "action_values": _action_values(contract),
            "action_dim": action_dim,
            "active_action_dim": active_action_dim,
            "action_carrier": action_carrier,
            "bridge_action_dim": BRIDGE_ACTION_DIM if action_carrier == "bridge" else None,
            "latency_metadata": True,
            "source_latency_column": source_columns.latency,
            "source_latency_unit": (
                "raw_frames" if source_columns.latency == "latency_raw_frames"
                else "observation_steps" if source_columns.latency == "latency"
                else None
            ),
            "target_latency_unit": target_latency_unit,
            "obs_stride_raw_frames": obs_stride_raw_frames,
            "source_rows_unit": source_rows_unit,
            "fps": fps,
            "obs_fps": fps,
            "latency_filter": [int(value) for value in split_latency_filter] if split_latency_filter else None,
            "episodes_per_latency": int(split_episodes_per_latency) if split_episodes_per_latency is not None else None,
            "episodes_per_latency_by_latency": (
                {str(k): int(v) for k, v in sorted(split_episodes_per_latency_by_latency.items())}
                if split_episodes_per_latency_by_latency is not None
                else None
            ),
            "max_episodes": int(max_episodes) if max_episodes is not None else None,
            "max_steps_per_episode": (
                int(max_steps_per_episode) if max_steps_per_episode is not None else None
            ),
            "prompt_override": bool(prompt_map_override),
            "default_latency": default_latency,
            "uses_state": True,
            "state_dim": state_dim,
            "active_state_dim": state_dim,
            "state_labels": state_labels,
            "state_normalization": {
                "type": "min_max",
                "min": state_min.tolist() if state_min is not None else [0.0] * state_dim,
                "max": state_max.tolist() if state_max is not None else [1.0] * state_dim,
            },
            "state_carrier": "native",
            "episodes": len(episode_lengths),
            "frames": int(sum(episode_lengths)),
            "task_prompts": task_prompts,
        }
        (split_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    train_manifest = _convert_split(
        "train",
        output_dir,
        split_latency_filter=train_latency_filter,
        split_episodes_per_latency=train_episodes_per_latency,
        split_episodes_per_latency_by_latency=train_episodes_per_latency_by_latency,
    )
    val_manifest = _convert_split(
        "validation",
        val_output_dir,
        split_latency_filter=eval_latency_filter,
        split_episodes_per_latency=eval_episodes_per_latency,
        split_episodes_per_latency_by_latency=eval_episodes_per_latency_by_latency,
    )
    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    train_manifest["validation_latency_filter"] = val_manifest["latency_filter"]
    train_manifest["validation_episodes_per_latency"] = val_manifest["episodes_per_latency"]
    train_manifest["validation_episodes_per_latency_by_latency"] = val_manifest["episodes_per_latency_by_latency"]
    (output_dir / "manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    return train_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", "--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--latency-filter", "--latency_filter", default=None)
    parser.add_argument("--episodes-per-latency", "--episodes_per_latency", type=int, default=None)
    parser.add_argument("--episodes-per-latency-by-latency", "--episodes_per_latency_by_latency", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", "--action_carrier", choices=["native", "bridge"], default="native")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--obs-stride-raw-frames", "--obs_stride_raw_frames", type=int, required=True)
    parser.add_argument("--source-latency-column", "--source_latency_column", choices=["latency", "latency_raw_frames"], required=True)
    parser.add_argument("--target-latency-unit", "--target_latency_unit", choices=["raw_frames", "observation_steps"], required=True)
    parser.add_argument("--gymnasium-task-contract", "--gymnasium_task_contract", required=True)
    args = parser.parse_args()

    latency_filter = None
    if args.latency_filter:
        latency_filter = [int(item) for item in str(args.latency_filter).split(",") if item.strip()]
    episodes_per_latency_by_latency = None
    if args.episodes_per_latency_by_latency:
        episodes_per_latency_by_latency = {}
        for item in str(args.episodes_per_latency_by_latency).split(","):
            item = item.strip()
            if not item:
                continue
            latency, episodes = item.split(":", 1)
            episodes_per_latency_by_latency[int(latency.strip())] = int(episodes.strip())

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        dataset_config_name=args.dataset_config_name,
        dataset_source_subdir=args.dataset_source_subdir,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        force=args.force,
        require_latency_prompt_map=True,
        latency_filter=latency_filter,
        episodes_per_latency=args.episodes_per_latency,
        episodes_per_latency_by_latency=episodes_per_latency_by_latency,
        action_carrier=args.action_carrier,
        fps=args.fps,
        obs_stride_raw_frames=args.obs_stride_raw_frames,
        source_latency_column=args.source_latency_column,
        target_latency_unit=args.target_latency_unit,
        source_rows_unit="decision_step",
        gymnasium_task_contract=args.gymnasium_task_contract,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
