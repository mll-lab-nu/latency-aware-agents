#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, Image
from PIL import Image as PILImage


# ------------------------------------------------------------
# Helpers: loading pickle teacher rollouts
# ------------------------------------------------------------

def _is_step_record(x: Any) -> bool:
    return isinstance(x, dict) and any(
        k in x for k in ("obs", "observation", "state", "action", "reward", "done", "info")
    )


def _is_episode_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and (len(x) == 0 or _is_step_record(x[0]))


def _is_distill_episode_dict(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    return (
        ("base_observations" in x or "observations" in x)
        and "actions" in x
        and "rewards" in x
        and "dones" in x
    )


def _episode_dict_to_steps(item: Any) -> List[dict]:
    if isinstance(item, dict):
        obs_seq = item.get("base_observations", item.get("observations"))
        actions = item.get("actions")
        rewards = item.get("rewards")
        dones = item.get("dones")
    else:
        obs_seq = getattr(item, "base_observations", getattr(item, "observations", None))
        actions = getattr(item, "actions", None)
        rewards = getattr(item, "rewards", None)
        dones = getattr(item, "dones", None)

    if obs_seq is None or actions is None or rewards is None or dones is None:
        raise ValueError("Episode object is missing one of: base_observations/observations, actions, rewards, dones")

    n = len(actions)
    lengths = [len(obs_seq), len(rewards), len(dones)]
    if len(set([n] + lengths)) != 1:
        raise ValueError(
            f"Episode arrays have inconsistent lengths: obs={len(obs_seq)}, actions={len(actions)}, "
            f"rewards={len(rewards)}, dones={len(dones)}"
        )

    steps: List[dict] = []
    for i in range(n):
        steps.append(
            {
                "obs": obs_seq[i],
                "action": actions[i],
                "reward": rewards[i],
                "done": dones[i],
                "info": {},
            }
        )
    return steps


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_episodes(obj: Any) -> List[List[dict]]:
    if _is_episode_list(obj):
        return [list(obj)]

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []

        if all(_is_distill_episode_dict(ep) for ep in obj):
            return [_episode_dict_to_steps(ep) for ep in obj]

        if all(_is_episode_list(ep) for ep in obj):
            return [list(ep) for ep in obj]

        episodes: List[List[dict]] = []
        for item in obj:
            if _is_distill_episode_dict(item):
                episodes.append(_episode_dict_to_steps(item))
                continue

            if isinstance(item, dict):
                for key in ("steps", "trajectory", "transitions", "rollout"):
                    if key in item and _is_episode_list(item[key]):
                        episodes.append(list(item[key]))
                        break
        if episodes:
            return episodes

    if isinstance(obj, dict):
        for key in ("episodes", "rollouts", "trajectories", "data"):
            if key in obj:
                return _extract_episodes(obj[key])

        if _is_distill_episode_dict(obj):
            return [_episode_dict_to_steps(obj)]

        values = list(obj.values())
        if values and all(_is_episode_list(v) for v in values):
            return [list(v) for v in values]

        candidate_keys = ["obs", "observation", "state", "action", "reward", "done", "info"]
        present = [k for k in candidate_keys if k in obj]
        if present:
            lengths = []
            for key in present:
                value = obj[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    lengths.append(len(value))
            if lengths and len(set(lengths)) == 1:
                n = lengths[0]
                episode = []
                for i in range(n):
                    step = {}
                    for key in present:
                        value = obj[key]
                        step[key] = value[i]
                    episode.append(step)
                return [episode]

    raise ValueError(
        "Could not infer episode structure from the teacher rollout pickle. "
        "This exporter expects the save_rollouts() format from your distillation script: "
        "a list of episode dicts with base_observations, actions, rewards, and dones."
    )


# ------------------------------------------------------------
# Helpers: observation -> image
# ------------------------------------------------------------

def _get_from_dict(d: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _extract_obs(step: dict) -> Any:
    obs = _get_from_dict(step, ["obs", "observation", "state"])
    if isinstance(obs, dict):
        for key in ("image", "screen", "pixels", "frame", "observation"):
            if key in obs:
                return obs[key]
    return obs


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _normalize_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if 0.0 <= lo and hi <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale frame, got {gray.shape}")
    return np.repeat(gray[:, :, None], 3, axis=2)


def _chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    return np.transpose(arr, (1, 2, 0))


def _tile_rgb_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(frames), axis=1)


def _pick_frame(frames: Sequence[np.ndarray], frame_policy: str) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("No frames available to pick from")
    if frame_policy == "latest":
        return frames[-1]
    if frame_policy == "oldest":
        return frames[0]
    if frame_policy == "middle":
        return frames[len(frames) // 2]
    if frame_policy == "tile":
        return _tile_rgb_frames(frames)
    raise ValueError(f"Unsupported frame policy: {frame_policy}")


def _stacked_rgb_frames_from_chw(arr: np.ndarray) -> List[np.ndarray]:
    c, h, w = arr.shape
    if c % 3 != 0:
        raise ValueError(f"CHW array does not contain stacked RGB frames: {arr.shape}")
    num_frames = c // 3
    return [_chw_to_hwc(arr[3 * i : 3 * (i + 1)]) for i in range(num_frames)]


def _stacked_rgb_frames_from_hwc(arr: np.ndarray) -> List[np.ndarray]:
    h, w, c = arr.shape
    if c % 3 != 0:
        raise ValueError(f"HWC array does not contain stacked RGB frames: {arr.shape}")
    num_frames = c // 3
    return [arr[:, :, 3 * i : 3 * (i + 1)] for i in range(num_frames)]


def _obs_to_png_ready(obs: Any, frame_policy: str = "latest") -> np.ndarray:
    arr = _normalize_uint8_image(_to_numpy(obs))

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        return _gray_to_rgb(arr)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported observation shape: {arr.shape}")

    # HWC-like
    if arr.shape[0] >= 16 and arr.shape[1] >= 16:
        h, w, c = arr.shape

        if c == 1:
            return _gray_to_rgb(arr[:, :, 0])
        if c == 3:
            return arr
        if c > 3 and c % 3 == 0:
            frames = _stacked_rgb_frames_from_hwc(arr)
            return _pick_frame(frames, frame_policy)

        # grayscale stack fallback
        frames = [_gray_to_rgb(arr[:, :, i]) for i in range(c)]
        return _pick_frame(frames, frame_policy)

    # CHW-like
    if arr.shape[1] >= 16 and arr.shape[2] >= 16:
        c, h, w = arr.shape

        if c == 1:
            return _gray_to_rgb(arr[0])
        if c == 3:
            return _chw_to_hwc(arr)
        if c > 3 and c % 3 == 0:
            frames = _stacked_rgb_frames_from_chw(arr)
            return _pick_frame(frames, frame_policy)

        # grayscale stack fallback
        frames = [_gray_to_rgb(arr[i]) for i in range(c)]
        return _pick_frame(frames, frame_policy)

    raise ValueError(f"Could not interpret observation shape: {arr.shape}")


# ------------------------------------------------------------
# Helpers: action names / prompts / metadata
# ------------------------------------------------------------

def _parse_action_names(s: str) -> List[str]:
    names = [x.strip() for x in s.split(",") if x.strip()]
    if not names:
        raise ValueError("--action-names cannot be empty")
    return names


def _make_prompt(prompt_template: str, env_name: str, action_names: Sequence[str]) -> str:
    return prompt_template.format(env_name=env_name, action_names=", ".join(action_names))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, bool):
            return int(x)
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# ------------------------------------------------------------
# Dataset building
# ------------------------------------------------------------

def _episode_seed(step: dict, episode_idx: int, default_seed_base: int) -> int:
    info = step.get("info", {}) if isinstance(step.get("info", {}), dict) else {}
    return _safe_int(info.get("seed", step.get("seed", default_seed_base + episode_idx)), default_seed_base + episode_idx)


def _episode_frame_skip(step: dict, default_frame_skip: int) -> int:
    info = step.get("info", {}) if isinstance(step.get("info", {}), dict) else {}
    return _safe_int(info.get("frame_skip", default_frame_skip), default_frame_skip)


def _step_to_row(
    step: dict,
    episode_idx: int,
    t: int,
    split: str,
    image_rel_path: str,
    prompt_template: str,
    env_name: str,
    action_names: Sequence[str],
    default_frame_skip: int,
    default_seed_base: int,
) -> Dict[str, Any]:
    info = step.get("info", {}) if isinstance(step.get("info", {}), dict) else {}
    action_id = _safe_int(step.get("action", step.get("action_id", 0)), 0)
    frame_skip = _episode_frame_skip(step, default_frame_skip)
    seed = _episode_seed(step, episode_idx, default_seed_base)

    if 0 <= action_id < len(action_names):
        action_text = action_names[action_id]
    else:
        action_text = f"ACTION_{action_id}"

    row = {
        "image": image_rel_path,
        "split": split,
        "prompt": _make_prompt(prompt_template, env_name, action_names),
        "action_id": action_id,
        "action_text": action_text,
        "frame_skip": frame_skip,
        "reward": _safe_float(step.get("reward", 0.0), 0.0),
        "done": bool(step.get("done", False)),
        "episode_idx": int(episode_idx),
        "t": int(t),
        "seed": int(seed),
        "deterministic_policy_step": True,
        "info": json.dumps(_jsonable(info), ensure_ascii=False),
    }

    for key in (
        "lives",
        "decision_step_idx",
        "env_step_idx",
        "latency_mode",
        "applied_action",
        "queued_action",
        "scheduled_apply_step",
        "pending_queue_len",
    ):
        if key in info:
            value = info[key]
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(_jsonable(value), ensure_ascii=False)
            else:
                row[key] = value

    return row


def build_rows_and_images(
    episodes: Sequence[Sequence[dict]],
    output_dir: str,
    prompt_template: str,
    env_name: str,
    action_names: Sequence[str],
    val_ratio: float,
    seed: int,
    default_frame_skip: int,
    default_seed_base: int,
    max_episodes: int,
    frame_policy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    episode_indices = list(range(len(episodes)))
    if max_episodes > 0:
        episode_indices = episode_indices[:max_episodes]

    shuffled = episode_indices[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 and val_ratio > 0 else 0
    val_eps = set(shuffled[:n_val])

    images_root = Path(output_dir) / "images"
    train_img_root = images_root / "train"
    val_img_root = images_root / "validation"
    train_img_root.mkdir(parents=True, exist_ok=True)
    val_img_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    stats = {"episodes": 0, "rows": 0, "train_rows": 0, "validation_rows": 0}

    for ep_idx in episode_indices:
        episode = episodes[ep_idx]
        split = "validation" if ep_idx in val_eps else "train"
        ep_dir = (val_img_root if split == "validation" else train_img_root) / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        for t, step in enumerate(episode):
            obs = _extract_obs(step)
            if obs is None:
                raise ValueError(f"Missing obs/observation/state at episode {ep_idx}, step {t}")
            img = _obs_to_png_ready(obs, frame_policy=frame_policy)
            image_name = f"step_{t:06d}.png"
            image_path = ep_dir / image_name
            PILImage.fromarray(img).save(image_path)

            rel_image_path = os.path.relpath(image_path, output_dir)
            row = _step_to_row(
                step=step,
                episode_idx=ep_idx,
                t=t,
                split=split,
                image_rel_path=rel_image_path,
                prompt_template=prompt_template,
                env_name=env_name,
                action_names=action_names,
                default_frame_skip=default_frame_skip,
                default_seed_base=default_seed_base,
            )
            rows.append(row)
            stats["rows"] += 1
            stats[f"{split}_rows"] += 1

        stats["episodes"] += 1

    return rows, stats


def rows_to_dataset_dict(rows: Sequence[Dict[str, Any]], output_dir: str) -> DatasetDict:
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    def _materialize_image_paths(examples: List[Dict[str, Any]]) -> Dataset:
        cols: Dict[str, List[Any]] = {}
        for row in examples:
            for key, value in row.items():
                cols.setdefault(key, []).append(value)
        ds = Dataset.from_dict(cols)
        ds = ds.map(lambda ex: {"image": str((Path(output_dir) / ex["image"]).resolve())})
        ds = ds.cast_column("image", Image())
        return ds

    dsd = DatasetDict()
    dsd["train"] = _materialize_image_paths(train_rows)
    dsd["validation"] = _materialize_image_paths(val_rows)
    return dsd


# ------------------------------------------------------------
# Parquet export / hub upload
# ------------------------------------------------------------

def export_local_parquet(
    dataset_dict: DatasetDict,
    parquet_dir: str,
    train_num_shards: int,
    validation_num_shards: int,
) -> None:
    parquet_root = Path(parquet_dir)
    parquet_root.mkdir(parents=True, exist_ok=True)

    for split, num_shards in (("train", train_num_shards), ("validation", validation_num_shards)):
        ds = dataset_dict[split]
        split_dir = parquet_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        if num_shards <= 0:
            num_shards = max(1, math.ceil(len(ds) / 50_000))

        for shard_idx in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
            out_path = split_dir / f"{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(str(out_path))


def push_dataset(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: Optional[str],
    max_shard_size: Optional[str],
    train_num_shards: Optional[int],
    validation_num_shards: Optional[int],
    private: bool,
) -> None:
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "token": token,
        "private": private,
        "embed_external_files": True,
    }

    if max_shard_size:
        kwargs["max_shard_size"] = max_shard_size
    elif train_num_shards or validation_num_shards:
        kwargs["num_shards"] = {
            "train": int(train_num_shards or 1),
            "validation": int(validation_num_shards or 1),
        }

    dataset_dict.push_to_hub(**kwargs)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert VizDoom teacher rollout pickle to a HF image dataset with train/validation parquet splits."
    )
    ap.add_argument("--teacher-rollout-path", type=str, required=True, help="Path to vizdoom_basic_teacher_latency0_rollouts.pkl")
    ap.add_argument("--output-dir", type=str, required=True, help="Local working directory for extracted PNGs and optional parquet")
    ap.add_argument("--repo-id", type=str, default="", help="HF dataset repo, e.g. talha1503/vizdoom_basic_teacher_parquet")
    ap.add_argument("--token", type=str, default=None, help="HF token; optional if already logged in")
    ap.add_argument("--env-name", type=str, default="VizDoom Basic")
    ap.add_argument(
        "--action-names",
        type=str,
        default="NOOP,MOVE_LEFT,MOVE_RIGHT,ATTACK",
        help="Comma-separated action names in teacher action-id order",
    )
    ap.add_argument(
        "--prompt-template",
        type=str,
        default="You are playing {env_name}. Align with the monster, shoot it, and survive. Choose the next action: {action_names}.",
        help="Prompt template. Available fields: {env_name}, {action_names}",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--default-frame-skip", type=int, default=0, help="Used if frame_skip is missing from step info")
    ap.add_argument("--default-seed-base", type=int, default=0, help="Used if seed is missing from rollout info")
    ap.add_argument(
        "--frame-policy",
        type=str,
        default="latest",
        choices=["latest", "oldest", "middle", "tile"],
        help="How to choose a frame when the observation stores stacked frames.",
    )
    ap.add_argument("--max-episodes", type=int, default=-1, help="Optional cap for debugging")

    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-shard-size", type=str, default="500MB", help="HF parquet shard size, e.g. 500MB or 1GB")
    ap.add_argument("--train-num-shards", type=int, default=0, help="Explicit train shard count for hub push if max_shard_size is empty")
    ap.add_argument("--validation-num-shards", type=int, default=0, help="Explicit validation shard count for hub push if max_shard_size is empty")

    ap.add_argument("--write-local-parquet", action="store_true")
    ap.add_argument("--local-parquet-dir", type=str, default="")
    ap.add_argument("--local-train-num-shards", type=int, default=32)
    ap.add_argument("--local-validation-num-shards", type=int, default=8)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    action_names = _parse_action_names(args.action_names)
    obj = _load_pickle(args.teacher_rollout_path)
    episodes = _extract_episodes(obj)
    print(f"[info] loaded pickle from: {args.teacher_rollout_path}")
    print(f"[info] detected episodes: {len(episodes)}")
    print(f"[info] prompt template: {args.prompt_template}")
    print(f"[info] frame policy: {args.frame_policy}")

    rows, stats = build_rows_and_images(
        episodes=episodes,
        output_dir=args.output_dir,
        prompt_template=args.prompt_template,
        env_name=args.env_name,
        action_names=action_names,
        val_ratio=args.val_ratio,
        seed=args.seed,
        default_frame_skip=args.default_frame_skip,
        default_seed_base=args.default_seed_base,
        max_episodes=args.max_episodes,
        frame_policy=args.frame_policy,
    )

    print(json.dumps({**stats, "frame_policy": args.frame_policy}, indent=2))

    dataset_dict = rows_to_dataset_dict(rows, args.output_dir)
    print(dataset_dict)
    print(dataset_dict["train"].features)

    stats_path = Path(args.output_dir) / "conversion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({**stats, "frame_policy": args.frame_policy}, f, indent=2)

    if args.write_local_parquet:
        parquet_dir = args.local_parquet_dir or str(Path(args.output_dir) / "parquet")
        export_local_parquet(
            dataset_dict=dataset_dict,
            parquet_dir=parquet_dir,
            train_num_shards=args.local_train_num_shards,
            validation_num_shards=args.local_validation_num_shards,
        )
        print(f"[info] wrote local parquet shards to: {parquet_dir}")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id is required when using --push-to-hub")
        push_dataset(
            dataset_dict=dataset_dict,
            repo_id=args.repo_id,
            token=args.token,
            max_shard_size=(args.max_shard_size or None),
            train_num_shards=(args.train_num_shards or None),
            validation_num_shards=(args.validation_num_shards or None),
            private=args.private,
        )
        print(f"[info] pushed dataset to HF: {args.repo_id}")


if __name__ == "__main__":
    main()
