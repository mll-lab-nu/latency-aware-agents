#!/usr/bin/env python
"""Build same-endpoint Demon Attack ghost-trail ablations.

Each selected endpoint raw frame is rendered four ways:
  - continuous_15: t-14..t
  - continuous_30: t-29..t
  - continuous_60: t-59..t
  - stride4_15: t-56,t-52,...,t

The endpoint/current frame is identical across all variants for a fair visual
comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
    render_tracks,
    track_summary,
)
from examples.rl_games.scripts.collect_demon_attack_raw_consecutive_test_set import (
    DEFAULT_CHECKPOINT_CANDIDATES,
    EpisodeRollout,
    RawFrameRecord,
    _force_config_overrides,
    collect_raw_rollouts,
    resolve_checkpoint_root,
    save_contact_sheet,
)
from latency_bench.data.sf_teacher_config import load_rollout_training_config

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "ghost_test_trail_ablation_demon_attack_same_endpoints"

VARIANT_SPECS = {
    "continuous_15": {"mode": "continuous", "length": 15},
    "continuous_30": {"mode": "continuous", "length": 30},
    "continuous_60": {"mode": "continuous", "length": 60},
    "stride4_15": {"mode": "stride", "length": 15, "stride": 4},
}
REQUIRED_KINDS = ("ship", "enemy", "player_bullet", "enemy_bullet")


@dataclass
class EndpointCandidate:
    episode_idx: int
    endpoint_raw_frame: int
    endpoint_record_idx: int
    records_by_raw_frame: dict[int, RawFrameRecord]
    track_counts: dict[str, int]
    score: float
    features: np.ndarray


def records_for_variant(
    records_by_raw_frame: dict[int, RawFrameRecord],
    endpoint_raw_frame: int,
    variant_name: str,
) -> list[RawFrameRecord] | None:
    spec = VARIANT_SPECS[variant_name]
    if spec["mode"] == "continuous":
        length = int(spec["length"])
        raw_frames = range(endpoint_raw_frame - length + 1, endpoint_raw_frame + 1)
    else:
        length = int(spec["length"])
        stride = int(spec["stride"])
        start = endpoint_raw_frame - stride * (length - 1)
        raw_frames = range(start, endpoint_raw_frame + 1, stride)

    records = []
    for raw_frame in raw_frames:
        record = records_by_raw_frame.get(int(raw_frame))
        if record is None:
            return None
        records.append(record)
    return records


def track_counts_for_records(records: list[RawFrameRecord]) -> dict[str, int]:
    tracks = collect_tracks([record.image for record in records])
    counts = Counter(track.kind for track in tracks)
    return {kind: int(counts.get(kind, 0)) for kind in REQUIRED_KINDS}


def endpoint_candidate_from_rollout(
    rollout: EpisodeRollout,
    endpoint_record_idx: int,
) -> EndpointCandidate | None:
    endpoint = rollout.frames[endpoint_record_idx]
    records_by_raw_frame = {record.raw_frame: record for record in rollout.frames}
    variant_records = {
        name: records_for_variant(records_by_raw_frame, endpoint.raw_frame, name)
        for name in VARIANT_SPECS
    }
    if any(records is None for records in variant_records.values()):
        return None

    continuous_60 = variant_records["continuous_60"]
    assert continuous_60 is not None
    counts = track_counts_for_records(continuous_60)
    frames = [record.image for record in continuous_60]
    motion = float(np.mean(np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16))))
    progress = endpoint_record_idx / max(1, len(rollout.frames) - 1)
    score = (
        7.0 * min(counts["enemy_bullet"], 2)
        + 4.0 * min(counts["player_bullet"], 3)
        + 2.0 * min(counts["enemy"], 4)
        + 1.0 * min(counts["ship"], 1)
        + min(motion / 8.0, 1.0)
    )
    features = np.array(
        [
            float(rollout.episode_idx),
            float(progress),
            min(counts["enemy_bullet"], 2) / 2.0,
            min(counts["player_bullet"], 3) / 3.0,
            min(counts["enemy"], 4) / 4.0,
            min(motion / 8.0, 1.0),
        ],
        dtype=np.float64,
    )
    return EndpointCandidate(
        episode_idx=int(rollout.episode_idx),
        endpoint_raw_frame=int(endpoint.raw_frame),
        endpoint_record_idx=int(endpoint_record_idx),
        records_by_raw_frame=records_by_raw_frame,
        track_counts=counts,
        score=float(score),
        features=features,
    )


def build_endpoint_candidates(
    rollouts: list[EpisodeRollout],
    *,
    candidate_stride: int,
) -> list[EndpointCandidate]:
    candidates: list[EndpointCandidate] = []
    for rollout in rollouts:
        for endpoint_record_idx in range(59, len(rollout.frames), max(1, int(candidate_stride))):
            candidate = endpoint_candidate_from_rollout(rollout, endpoint_record_idx)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def select_diverse_endpoints(candidates: list[EndpointCandidate], *, count: int) -> list[EndpointCandidate]:
    if len(candidates) < count:
        raise ValueError(f"only found {len(candidates)} endpoint candidates, need {count}")

    selected: list[EndpointCandidate] = []
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    while len(selected) < count:
        selected_kinds = {
            kind
            for candidate in selected
            for kind, value in candidate.track_counts.items()
            if value > 0
        }
        selected_episodes = {candidate.episode_idx for candidate in selected}
        best = None
        best_value = None
        for candidate in candidates:
            if candidate in selected:
                continue
            coverage_bonus = 0.0
            if "enemy_bullet" not in selected_kinds and candidate.track_counts["enemy_bullet"] > 0:
                coverage_bonus += 30.0
            if "player_bullet" not in selected_kinds and candidate.track_counts["player_bullet"] > 0:
                coverage_bonus += 12.0
            if "enemy" not in selected_kinds and candidate.track_counts["enemy"] > 0:
                coverage_bonus += 8.0
            if "ship" not in selected_kinds and candidate.track_counts["ship"] > 0:
                coverage_bonus += 4.0
            episode_bonus = 2.5 if candidate.episode_idx not in selected_episodes else 0.0
            if selected:
                min_dist = min(float(np.linalg.norm(candidate.features - other.features)) for other in selected)
            else:
                min_dist = 0.0
            value = candidate.score + coverage_bonus + episode_bonus + 1.5 * min_dist
            if best_value is None or value > best_value:
                best = candidate
                best_value = value
        if best is None:
            break
        selected.append(best)

    selected_kinds = {
        kind
        for candidate in selected
        for kind, value in candidate.track_counts.items()
        if value > 0
    }
    missing = sorted(set(REQUIRED_KINDS) - selected_kinds)
    if missing:
        raise ValueError(
            "selected endpoints do not cover required tracked kinds: "
            f"{missing}. Increase --episodes-to-scan or lower --candidate-stride."
        )
    return sorted(selected, key=lambda item: (item.episode_idx, item.endpoint_raw_frame))


def make_labeled_sheet(images: list[tuple[str, np.ndarray]], out_path: Path, *, cols: int) -> None:
    tiles = []
    for label, arr in images:
        im = Image.fromarray(arr).convert("RGB").resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (im.width, im.height + 18), "white")
        tile.paste(im, (0, 18))
        ImageDraw.Draw(tile).text((3, 3), label, fill=(0, 0, 0))
        tiles.append(tile)

    w = max(tile.width for tile in tiles)
    h = max(tile.height for tile in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        sheet.paste(tile, (col * w, row * h))
    sheet.save(out_path)


def stack_images_with_labels(items: list[tuple[str, Path]], out_path: Path) -> None:
    tiles = []
    for label, path in items:
        im = Image.open(path).convert("RGB")
        tile = Image.new("RGB", (im.width, im.height + 24), "white")
        tile.paste(im, (0, 24))
        ImageDraw.Draw(tile).text((4, 6), label, fill=(0, 0, 0))
        tiles.append(tile)
    width = max(tile.width for tile in tiles)
    height = sum(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(out_path)


def save_variant_outputs(
    records: list[RawFrameRecord],
    *,
    variant_dir: Path,
    variant_name: str,
) -> dict[str, Any]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    frames = [record.image for record in records]
    for i, record in enumerate(records):
        tag = "current" if i == len(records) - 1 else "raw"
        Image.fromarray(record.image).save(variant_dir / f"frame_{i:02d}_{tag}_raw{record.raw_frame:06d}.png")
    save_contact_sheet(frames, variant_dir / "contact_sheet_inputs.png")

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
    Image.fromarray(final).save(variant_dir / "ghost_trail_composite.png")
    Image.fromarray(bullets).save(variant_dir / "bullet_only_composite.png")
    Image.fromarray(large).save(variant_dir / "ship_enemy_only_composite.png")
    Image.fromarray(debug).save(variant_dir / "debug_tracks_3x.png")
    (variant_dir / "track_summary.txt").write_text(track_summary(tracks))
    make_labeled_sheet(
        [
            ("current", frames[-1]),
            ("bullet-only", bullets),
            ("ship/enemy-only", large),
            ("combined", final),
        ],
        variant_dir / "comparison_sheet_2x.png",
        cols=2,
    )

    counts = Counter(track.kind for track in tracks)
    metadata = {
        "variant": variant_name,
        "raw_frames": [int(record.raw_frame) for record in records],
        "decision_steps": [int(record.decision_step) for record in records],
        "track_counts": {kind: int(counts.get(kind, 0)) for kind in REQUIRED_KINDS},
    }
    (variant_dir / "variant_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {**metadata, "path": str(variant_dir)}


def save_endpoint_outputs(candidate: EndpointCandidate, *, out_dir: Path, set_idx: int) -> dict[str, Any]:
    endpoint_dir = out_dir / f"endpoint_{set_idx}_ep{candidate.episode_idx}_raw{candidate.endpoint_raw_frame}"
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    variant_metadata = {}
    variant_images = {}
    current = None
    for variant_name in VARIANT_SPECS:
        records = records_for_variant(candidate.records_by_raw_frame, candidate.endpoint_raw_frame, variant_name)
        if records is None:
            raise RuntimeError(f"missing records for {variant_name} at raw {candidate.endpoint_raw_frame}")
        metadata = save_variant_outputs(
            records,
            variant_dir=endpoint_dir / variant_name,
            variant_name=variant_name,
        )
        variant_metadata[variant_name] = metadata
        variant_images[variant_name] = np.array(Image.open(endpoint_dir / variant_name / "ghost_trail_composite.png"))
        current = records[-1].image

    assert current is not None
    make_labeled_sheet(
        [
            ("current", current),
            ("continuous_15", variant_images["continuous_15"]),
            ("continuous_30", variant_images["continuous_30"]),
            ("continuous_60", variant_images["continuous_60"]),
        ],
        endpoint_dir / "length_ablation_sheet.png",
        cols=4,
    )
    make_labeled_sheet(
        [
            ("current", current),
            ("continuous_60", variant_images["continuous_60"]),
            ("stride4_15", variant_images["stride4_15"]),
        ],
        endpoint_dir / "sampling_ablation_sheet.png",
        cols=3,
    )
    make_labeled_sheet(
        [
            ("current", current),
            ("continuous_15", variant_images["continuous_15"]),
            ("continuous_30", variant_images["continuous_30"]),
            ("continuous_60", variant_images["continuous_60"]),
            ("stride4_15", variant_images["stride4_15"]),
        ],
        endpoint_dir / "all_variants_sheet.png",
        cols=5,
    )

    metadata = {
        "set_idx": int(set_idx),
        "episode_idx": int(candidate.episode_idx),
        "endpoint_raw_frame": int(candidate.endpoint_raw_frame),
        "selection_track_counts_continuous_60": candidate.track_counts,
        "selection_score": float(candidate.score),
        "variants": variant_metadata,
    }
    (endpoint_dir / "endpoint_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"name": endpoint_dir.name, "path": str(endpoint_dir), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-endpoints", type=int, default=5)
    parser.add_argument("--episodes-to-scan", type=int, default=8)
    parser.add_argument("--max-decision-steps", type=int, default=900)
    parser.add_argument("--candidate-stride", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulator", default="")
    parser.add_argument("--raw-stack-frames", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root or None)
    out_dir = Path(args.out_dir).expanduser().resolve()
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

    candidates = build_endpoint_candidates(rollouts, candidate_stride=int(args.candidate_stride))
    if not any(candidate.track_counts["enemy_bullet"] > 0 for candidate in candidates):
        raise ValueError(
            "no endpoint candidates with enemy_bullet tracks found. "
            "Increase --episodes-to-scan or lower --candidate-stride."
        )
    selected = select_diverse_endpoints(candidates, count=int(args.num_endpoints))
    endpoint_metadata = [
        save_endpoint_outputs(candidate, out_dir=out_dir, set_idx=i)
        for i, candidate in enumerate(selected)
    ]
    stack_images_with_labels(
        [(item["name"], Path(item["path"]) / "length_ablation_sheet.png") for item in endpoint_metadata],
        out_dir / "overview_length_ablation_sheet.png",
    )
    stack_images_with_labels(
        [(item["name"], Path(item["path"]) / "sampling_ablation_sheet.png") for item in endpoint_metadata],
        out_dir / "overview_sampling_ablation_sheet.png",
    )
    stack_images_with_labels(
        [(item["name"], Path(item["path"]) / "all_variants_sheet.png") for item in endpoint_metadata],
        out_dir / "overview_all_variants_sheet.png",
    )

    metadata = {
        "schema": "demon_attack_same_endpoint_ghost_trail_ablation_v1",
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_candidates": [str(path) for path in DEFAULT_CHECKPOINT_CANDIDATES],
        "config_metadata": config_metadata,
        "runtime_metadata": runtime_metadata,
        "num_endpoints": int(args.num_endpoints),
        "episodes_to_scan": int(args.episodes_to_scan),
        "max_decision_steps": int(args.max_decision_steps),
        "candidate_stride": int(args.candidate_stride),
        "variants": VARIANT_SPECS,
        "selected_endpoints": endpoint_metadata,
        "rollouts": [
            {
                "episode_idx": int(rollout.episode_idx),
                "seed": int(rollout.seed),
                "raw_frame_count": len(rollout.frames),
                "decision_steps": int(rollout.decision_steps),
                "return": float(rollout.return_value),
                "raw_return": float(rollout.raw_return),
            }
            for rollout in rollouts
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Done. Selected {len(endpoint_metadata)} endpoints -> {out_dir}")


if __name__ == "__main__":
    main()
