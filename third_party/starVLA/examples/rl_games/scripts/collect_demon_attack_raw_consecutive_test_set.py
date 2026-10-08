#!/usr/bin/env python
"""Collect a small Demon Attack raw-consecutive test set from an SF teacher.

This is intentionally separate from latency-sensitive-bench's canonical
dataset exporter. It rolls out the Demon Attack latency-0 teacher, reconstructs
consecutive raw RGB frames from the env raw-RGB stacks, selects five diverse
15-frame windows, and renders the current bullet-aware ghost-trail diagnostics
for each selected window.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

STARVLA_ROOT = Path(__file__).resolve().parents[3]
NU_ROOT = STARVLA_ROOT.parent
LATENCY_BENCH_ROOT = NU_ROOT / "latency-sensitive-bench"
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))
if str(LATENCY_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENCY_BENCH_ROOT))

from examples.rl_games.scripts.build_bullet_aware_ghost_trails_demon_attack import (
    collect_tracks,
    debug_tracks_image,
    make_sheet,
    render_tracks,
    track_summary,
)
from examples.rl_games.scripts.image_patching import DEMON_ATTACK_SHIP_RGB
from latency_bench.data.rollout_image_io import frames_from_env_raw_rgb_info
from latency_bench.data.sf_teacher_config import load_rollout_training_config
from latency_bench.data.sf_teacher_drivers import BatchedTeacherRolloutDriver, SerialTeacherRolloutDriver
from latency_bench.data.sf_teacher_runtime import (
    build_rollout_driver_runtime,
    rollout_step_flags,
    step_teacher_policy,
)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT_CANDIDATES = [
    LATENCY_BENCH_ROOT / "checkpoints_old" / "demon_attack_sf_same_latency_l0_seed0",
    LATENCY_BENCH_ROOT / "checkpoints" / "demon_attack_sf_same_latency_l0_seed0",
]

SCORE_ROW_END = 20
GROUND_ROW_START = 188


@dataclass
class RawFrameRecord:
    episode_idx: int
    decision_step: int
    raw_frame: int
    image: np.ndarray
    action: Any
    reward: float
    raw_reward: float


@dataclass
class EpisodeRollout:
    episode_idx: int
    seed: int
    frames: list[RawFrameRecord]
    decision_steps: int
    return_value: float
    raw_return: float
    terminated: bool


@dataclass
class WindowCandidate:
    episode_idx: int
    start_idx: int
    records: list[RawFrameRecord]
    features: np.ndarray
    score: float
    summary: dict[str, Any]


def resolve_checkpoint_root(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint root does not exist: {path}")
        return path
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path.resolve()
    checked = "\n".join(f"  - {path}" for path in DEFAULT_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "Demon Attack latency-0 checkpoint not found. Checked:\n"
        f"{checked}\n"
        "Download command from Standard Pipeline.md:\n"
        "  hf download latency-sensitive-bench/latency-checkpoints --repo-type model "
        '--include "checkpoints_old/demon_attack_sf_same_latency_l0_seed0/**" --local-dir ./'
    )


def _force_config_overrides(config: dict[str, Any], *, device: str, simulator: str | None) -> dict[str, Any]:
    config = dict(config)
    flat_cfg = dict(config.get("flat_cfg", {}))
    if simulator:
        flat_cfg["simulator"] = str(simulator)
    config["flat_cfg"] = flat_cfg
    if device:
        config["device_override"] = str(device)
    config["env_name"] = "demon_attack"
    return config


def _append_raw_stack_records(
    out: list[RawFrameRecord],
    *,
    episode_idx: int,
    decision_step: int,
    info: dict[str, Any],
    action: Any,
    reward: float,
) -> None:
    stack = frames_from_env_raw_rgb_info(info)
    if not stack:
        return
    end_raw_frame = int(info.get("raw_frame", info.get("env_step_idx", 0)))
    start_raw_frame = end_raw_frame - len(stack)
    last_seen = out[-1].raw_frame if out else -1
    raw_reward = float(info.get("raw_reward", reward))
    for offset, frame in enumerate(stack):
        raw_frame = start_raw_frame + offset
        if raw_frame < 0 or raw_frame <= last_seen:
            continue
        out.append(
            RawFrameRecord(
                episode_idx=int(episode_idx),
                decision_step=int(decision_step),
                raw_frame=int(raw_frame),
                image=np.array(frame, copy=True),
                action=action,
                reward=float(reward),
                raw_reward=raw_reward,
            )
        )


def collect_raw_rollouts(
    config: dict[str, Any],
    *,
    scratch_dir: Path,
    episodes_to_scan: int,
    max_decision_steps: int,
    deterministic: bool,
    raw_stack_frames: int | None,
) -> tuple[list[EpisodeRollout], dict[str, Any]]:
    if raw_stack_frames is not None:
        config = dict(config)
        flat_cfg = dict(config.get("flat_cfg", {}))
        flat_cfg["frame_stack"] = int(raw_stack_frames)
        config["flat_cfg"] = flat_cfg
    cfg, runtime, driver, checkpoint_frame_stack = build_rollout_driver_runtime(
        config,
        output_dir=scratch_dir,
        num_envs=1,
        export_env_raw_rgb_frames=True,
    )

    device = runtime["device"]
    rnn_states = torch.zeros([driver.slot_count, runtime["rnn_size"]], dtype=torch.float32, device=device)
    env_name = "demon_attack"
    rollouts: list[EpisodeRollout] = []

    try:
        for episode_idx in range(int(episodes_to_scan)):
            seed = int(getattr(cfg, "seed", 0)) + episode_idx
            if isinstance(driver, BatchedTeacherRolloutDriver):
                reset_results = driver.reset_slots({0: seed}, [True])
            else:
                reset_results = driver.reset_slots({0: seed})
            active_obs = [None for _ in range(driver.slot_count)]
            active_obs[0] = reset_results[0][0]
            rnn_states[0].zero_()

            records: list[RawFrameRecord] = []
            decision_step = 0
            episode_return = 0.0
            episode_raw_return = 0.0
            terminated = False
            while decision_step < int(max_decision_steps):
                raw_actions, new_rnn_states, step_results = step_teacher_policy(
                    runtime,
                    driver,
                    active_slots=[0],
                    active_obs=active_obs,
                    rnn_states=rnn_states,
                    deterministic=deterministic,
                )
                step_result = step_results[0]
                reward = float(step_result.reward)
                info = dict(step_result.info)
                episode_return += reward
                episode_raw_return += float(info.get("raw_reward", reward))
                _append_raw_stack_records(
                    records,
                    episode_idx=episode_idx,
                    decision_step=decision_step,
                    info=info,
                    action=raw_actions[0],
                    reward=reward,
                )
                rnn_states[0] = new_rnn_states[0]
                next_decision_step = decision_step + 1
                life_loss_boundary, episode_done = rollout_step_flags(
                    env_name,
                    step_result,
                    next_decision_step=next_decision_step,
                    max_steps_per_episode=int(max_decision_steps),
                )
                if episode_done:
                    terminated = True
                    break
                if life_loss_boundary:
                    active_obs[0], reset_info = driver.continue_after_life_loss(0)
                    rnn_states[0].zero_()
                    # Life loss reset frames are intentionally not appended as
                    # candidate raw motion windows.
                    _ = reset_info
                else:
                    active_obs[0] = step_result.next_obs
                decision_step = next_decision_step

            rollouts.append(
                EpisodeRollout(
                    episode_idx=episode_idx,
                    seed=seed,
                    frames=records,
                    decision_steps=decision_step + 1,
                    return_value=float(episode_return),
                    raw_return=float(episode_raw_return),
                    terminated=bool(terminated),
                )
            )
    finally:
        driver.close()

    metadata = {
        "checkpoint_train_step": int(runtime["checkpoint"]["train_step"]),
        "checkpoint_frame_stack": int(checkpoint_frame_stack),
        "env_fps": float(cfg.env_fps),
        "obs_fps": float(cfg.obs_fps),
        "obs_stride_raw_frames": int(round(float(cfg.env_fps) / float(cfg.obs_fps))),
        "device": str(device),
        "simulator": str(getattr(cfg, "simulator", "")),
    }
    return rollouts, metadata


def foreground_mask(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    not_black = np.any(frame != 0, axis=-1)
    rows = np.arange(H)[:, None]
    in_band = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    not_ship = ~np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    return not_black & in_band & not_ship


def component_stats(frame: np.ndarray) -> tuple[int, int, int]:
    import cv2

    fg = foreground_mask(frame)
    num, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    small = 0
    large = 0
    for lab in range(1, num):
        mask = labels == lab
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.where(mask)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if area <= 14 and w <= 6 and h <= 18:
            small += 1
        else:
            large += 1
    return small, large, int(fg.sum())


def ship_centroid(frame: np.ndarray) -> tuple[float, float] | None:
    mask = np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def candidate_from_records(
    records: list[RawFrameRecord],
    *,
    episode_idx: int,
    start_idx: int,
    window_len: int,
    episode_frame_count: int,
) -> WindowCandidate | None:
    window = records[start_idx : start_idx + window_len]
    if len(window) != window_len:
        return None
    raw_indices = [item.raw_frame for item in window]
    if any(b - a != 1 for a, b in zip(raw_indices, raw_indices[1:])):
        return None

    current = window[-1].image
    first = window[0].image
    small, large, fg_pixels = component_stats(current)
    motion = float(np.mean(np.abs(current.astype(np.int16) - first.astype(np.int16))))
    centroid = ship_centroid(current)
    ship_x = 0.5 if centroid is None else centroid[0] / current.shape[1]
    progress = start_idx / max(1, episode_frame_count - window_len)
    bullet_score = min(float(small), 6.0) / 6.0
    enemy_score = min(float(large), 4.0) / 4.0
    motion_score = min(motion / 8.0, 1.0)
    score = 2.0 * bullet_score + enemy_score + motion_score + min(fg_pixels / 250.0, 1.0)
    features = np.array([progress, ship_x, bullet_score, enemy_score, motion_score], dtype=np.float64)
    return WindowCandidate(
        episode_idx=episode_idx,
        start_idx=start_idx,
        records=window,
        features=features,
        score=float(score),
        summary={
            "raw_start": int(raw_indices[0]),
            "raw_end": int(raw_indices[-1]),
            "decision_start": int(window[0].decision_step),
            "decision_end": int(window[-1].decision_step),
            "small_components_current": int(small),
            "large_components_current": int(large),
            "foreground_pixels_current": int(fg_pixels),
            "motion_mean_abs_rgb": float(motion),
            "ship_centroid_current": None if centroid is None else [float(centroid[0]), float(centroid[1])],
            "score": float(score),
        },
    )


def build_candidates(rollouts: list[EpisodeRollout], *, window_len: int, stride: int) -> list[WindowCandidate]:
    candidates: list[WindowCandidate] = []
    for rollout in rollouts:
        max_start = len(rollout.frames) - int(window_len)
        for start_idx in range(0, max_start + 1, max(1, int(stride))):
            cand = candidate_from_records(
                rollout.frames,
                episode_idx=rollout.episode_idx,
                start_idx=start_idx,
                window_len=int(window_len),
                episode_frame_count=len(rollout.frames),
            )
            if cand is not None:
                candidates.append(cand)
    return candidates


def select_diverse_windows(candidates: list[WindowCandidate], *, count: int) -> list[WindowCandidate]:
    if len(candidates) < count:
        raise ValueError(f"only found {len(candidates)} candidate windows, need {count}")
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[WindowCandidate] = []
    while len(selected) < count:
        best = None
        best_value = None
        for cand in candidates:
            if cand in selected:
                continue
            if not selected:
                value = cand.score
            else:
                min_dist = min(float(np.linalg.norm(cand.features - other.features)) for other in selected)
                value = cand.score + 1.75 * min_dist
            if best_value is None or value > best_value:
                best = cand
                best_value = value
        if best is None:
            break
        selected.append(best)
    return sorted(selected, key=lambda c: (c.episode_idx, c.records[0].raw_frame))


def save_contact_sheet(frames: list[np.ndarray], out_path: Path, *, cols: int = 5) -> None:
    H, W = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * H, cols * W, 3), 255, dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = frame
    Image.fromarray(sheet).save(out_path)


def save_window_outputs(window: WindowCandidate, *, out_dir: Path, set_idx: int) -> dict[str, Any]:
    set_dir = out_dir / f"set_{set_idx}_ep{window.episode_idx}_raw{window.records[0].raw_frame}"
    set_dir.mkdir(parents=True, exist_ok=True)
    frames = [record.image for record in window.records]
    for i, record in enumerate(window.records):
        tag = "current" if i == len(window.records) - 1 else "raw"
        Image.fromarray(record.image).save(set_dir / f"frame_{i:02d}_{tag}_raw{record.raw_frame:06d}.png")
    save_contact_sheet(frames, set_dir / "contact_sheet_inputs.png")

    tracks = collect_tracks(frames)
    final = render_tracks(frames, tracks, render_all_history=True)
    bullets = render_tracks(
        frames,
        tracks,
        include_kinds={"player_bullet", "enemy_bullet"},
        render_all_history=True,
    )
    large = render_tracks(
        frames,
        tracks,
        include_kinds={"ship", "enemy"},
        render_all_history=True,
    )
    debug = debug_tracks_image(frames[-1], tracks)
    Image.fromarray(final).save(set_dir / "ghost_trail_composite.png")
    Image.fromarray(bullets).save(set_dir / "bullet_only_composite.png")
    Image.fromarray(large).save(set_dir / "ship_enemy_only_composite.png")
    Image.fromarray(debug).save(set_dir / "debug_tracks_3x.png")
    (set_dir / "track_summary.txt").write_text(track_summary(tracks))
    make_sheet(
        [
            ("current", frames[-1]),
            ("bullet-only", bullets),
            ("ship/enemy-only", large),
            ("combined", final),
        ],
        set_dir / "comparison_sheet_2x.png",
    )

    metadata = {
        "set_idx": int(set_idx),
        "episode_idx": int(window.episode_idx),
        "raw_frames": [int(record.raw_frame) for record in window.records],
        "decision_steps": [int(record.decision_step) for record in window.records],
        "actions": [str(record.action) for record in window.records],
        "rewards": [float(record.reward) for record in window.records],
        **window.summary,
    }
    (set_dir / "window_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    return {"name": set_dir.name, "path": str(set_dir), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--window-len", type=int, default=15)
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--episodes-to-scan", type=int, default=5)
    parser.add_argument("--max-decision-steps", type=int, default=900)
    parser.add_argument("--candidate-stride", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulator", default="", help="Optional Sample Factory simulator override, e.g. cpu.")
    parser.add_argument("--raw-stack-frames", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root or None)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (SCRIPT_DIR / f"ghost_trail_test_outputs_demon_attack_raw_consecutive_{int(args.window_len)}").resolve()
    )
    scratch_dir = out_dir / "_scratch_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    config, config_metadata = load_rollout_training_config(
        checkpoint_root,
        latency_config_path=None,
        device_override=args.device or None,
    )
    config = _force_config_overrides(
        config,
        device=str(args.device),
        simulator=str(args.simulator) if args.simulator else None,
    )
    raw_stack_frames = int(args.raw_stack_frames) if int(args.raw_stack_frames) > 0 else None

    rollouts, runtime_metadata = collect_raw_rollouts(
        config,
        scratch_dir=scratch_dir,
        episodes_to_scan=int(args.episodes_to_scan),
        max_decision_steps=int(args.max_decision_steps),
        deterministic=bool(args.deterministic),
        raw_stack_frames=raw_stack_frames,
    )
    candidates = build_candidates(
        rollouts,
        window_len=int(args.window_len),
        stride=int(args.candidate_stride),
    )
    selected = select_diverse_windows(candidates, count=int(args.num_windows))

    selected_metadata = [
        save_window_outputs(window, out_dir=out_dir, set_idx=i)
        for i, window in enumerate(selected)
    ]
    metadata = {
        "schema": "demon_attack_raw_consecutive_test_set_v1",
        "checkpoint_root": str(checkpoint_root),
        "config_metadata": config_metadata,
        "runtime_metadata": runtime_metadata,
        "window_len": int(args.window_len),
        "num_windows": int(args.num_windows),
        "episodes_to_scan": int(args.episodes_to_scan),
        "max_decision_steps": int(args.max_decision_steps),
        "candidate_stride": int(args.candidate_stride),
        "rollouts": [
            {
                "episode_idx": int(rollout.episode_idx),
                "seed": int(rollout.seed),
                "raw_frame_count": len(rollout.frames),
                "decision_steps": int(rollout.decision_steps),
                "return_value": float(rollout.return_value),
                "raw_return": float(rollout.raw_return),
                "terminated": bool(rollout.terminated),
            }
            for rollout in rollouts
        ],
        "selected_windows": selected_metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Done. Selected {len(selected_metadata)} windows -> {out_dir}")


if __name__ == "__main__":
    main()
