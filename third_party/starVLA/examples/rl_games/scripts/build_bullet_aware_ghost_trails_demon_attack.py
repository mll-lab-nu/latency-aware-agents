#!/usr/bin/env python
"""Build bullet-aware Demon Attack ghost trails from raw consecutive windows.

This experiment uses the same sample positions as the existing Demon Attack
ghost-trail outputs, but it reloads a raw consecutive frame window ending at
each position. That matters for bullets: ship-selected ghost frames can skip
short-lived bullet motion entirely.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.image_patching import DEMON_ATTACK_SHIP_RGB, _require_cv2

import cv2

DATASET_NAME = "latency-sensitive-bench/benchmark-datasets"
DATASET_SUBDIR = "demon_attack_200ep__demon_attack_fix_latency_0_200ep"
DATASET_REVISION = "5305590625c28a6922e73f46d18d52738762c906"

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "ghost_trail_test_outputs_demon_attack_bidiirection"
DEFAULT_OUT_DIR = SCRIPT_DIR / "ghost_trail_test_outputs_demon_attack_bullet_aware_all"

SCORE_ROW_END = 20
GROUND_ROW_START = 188
MAX_FOREGROUND_COMPONENT_AREA = 500


@dataclass
class Component:
    frame_idx: int
    centroid: tuple[float, float]
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    mean_color: np.ndarray


@dataclass
class Track:
    kind: str
    history: list[Component]


def load_episode_rows(episode_idx: int, cache_dir: str | None):
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_NAME, DATASET_SUBDIR,
        revision=DATASET_REVISION,
        split="train",
        verification_mode="no_checks",
        cache_dir=cache_dir,
        columns=["episode_idx", "decision_step", "image"],
    )
    ds = ds.filter(lambda ex: ex["episode_idx"] == episode_idx)
    ds = ds.sort("decision_step")
    return ds


def parse_sample_positions(sample_dir: Path) -> list[tuple[str, int]]:
    out = []
    for item in sorted(sample_dir.iterdir()):
        if not item.is_dir():
            continue
        match = re.search(r"_pos_(\d+)$", item.name)
        if match:
            out.append((item.name, int(match.group(1))))
    if not out:
        raise ValueError(f"no set_*_pos_* directories found in {sample_dir}")
    return out


def foreground_components(frame: np.ndarray, frame_idx: int) -> list[Component]:
    _require_cv2()
    H, W = frame.shape[:2]

    not_black = np.any(frame != 0, axis=-1)
    rows = np.arange(H)[:, None]
    in_band = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    not_ship = ~np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    fg = not_black & in_band & not_ship

    num, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    comps: list[Component] = []
    for lab in range(1, num):
        mask = labels == lab
        area = int(mask.sum())
        if area < 1 or area > MAX_FOREGROUND_COMPONENT_AREA:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        comps.append(
            Component(
                frame_idx=frame_idx,
                centroid=(float(xs.mean()), float(ys.mean())),
                mask=mask,
                area=area,
                bbox=(x0, y0, x1, y1),
                mean_color=frame[mask].mean(axis=0),
            )
        )
    return comps


def ship_track(frames: list[np.ndarray]) -> Track | None:
    history = []
    for frame_idx, frame in enumerate(frames):
        mask = np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        history.append(
            Component(
                frame_idx=frame_idx,
                centroid=(float(xs.mean()), float(ys.mean())),
                mask=mask,
                area=int(mask.sum()),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                mean_color=np.array(DEMON_ATTACK_SHIP_RGB, dtype=np.float32),
            )
        )
    return Track("ship", history) if history else None


def is_small_component(comp: Component) -> bool:
    x0, y0, x1, y1 = comp.bbox
    w = x1 - x0
    h = y1 - y0
    return comp.area <= 14 and w <= 6 and h <= 18


def _velocity(history: list[Component]) -> tuple[float, float]:
    if len(history) < 2:
        return (0.0, 0.0)
    x0, y0 = history[-2].centroid
    x1, y1 = history[-1].centroid
    return (x1 - x0, y1 - y0)


def build_tracks(
    frame_components: list[list[Component]],
    *,
    bullet_like: bool,
    max_gap: int,
) -> list[list[Component]]:
    tracks: list[list[Component]] = []
    active_last_idx: list[int] = []

    for frame_idx, comps in enumerate(frame_components):
        candidates = [c for c in comps if is_small_component(c) == bullet_like]
        used_tracks: set[int] = set()

        for comp in candidates:
            best_ti = None
            best_score = None
            cx, cy = comp.centroid
            for ti, history in enumerate(tracks):
                if ti in used_tracks:
                    continue
                gap = frame_idx - active_last_idx[ti]
                if gap < 1 or gap > max_gap:
                    continue

                lx, ly = history[-1].centroid
                dx = cx - lx
                dy = cy - ly
                if bullet_like:
                    if abs(dx) > 8 or abs(dy) > 24:
                        continue
                    if len(history) >= 2:
                        _, prev_dy = _velocity(history)
                        if abs(prev_dy) >= 1.0 and abs(dy) >= 1.0 and np.sign(prev_dy) != np.sign(dy):
                            continue
                    score = abs(dx) * 2.0 + abs(dy)
                else:
                    dist = float(np.hypot(dx, dy))
                    if dist > 22.0:
                        continue
                    score = dist

                if best_score is None or score < best_score:
                    best_ti = ti
                    best_score = score

            if best_ti is None:
                tracks.append([comp])
                active_last_idx.append(frame_idx)
            else:
                tracks[best_ti].append(comp)
                active_last_idx[best_ti] = frame_idx
                used_tracks.add(best_ti)

    return tracks


def classify_bullet_track(history: list[Component]) -> str | None:
    if len(history) < 2:
        return None
    x0, y0 = history[0].centroid
    x1, y1 = history[-1].centroid
    dy = y1 - y0
    dx = x1 - x0
    frame_span = max(1, history[-1].frame_idx - history[0].frame_idx)
    median_area = float(np.median([comp.area for comp in history]))
    if median_area > 8.0:
        return None
    if abs(dy) < 10.0:
        return None
    if abs(dy) < 1.2 * abs(dx):
        return None
    if abs(dy) / frame_span < 3.0:
        return None
    return "player_bullet" if dy < 0 else "enemy_bullet"


def truncate_by_direction(history: list[Component]) -> list[Component]:
    if len(history) < 3:
        return history

    centroids = np.array([h.centroid for h in history], dtype=np.float64)
    ref_vec = centroids[-1] - centroids[-2]
    if not np.any(ref_vec):
        return history[-2:]

    kept = [len(history) - 2]
    for i in range(len(history) - 3, -1, -1):
        v = centroids[i + 1] - centroids[i]
        if float(np.dot(v, ref_vec)) <= 0:
            break
        kept.append(i)
    kept.sort()
    kept.append(len(history) - 1)
    return [history[i] for i in kept]


def alpha_for_frame(frame_idx: int, n: int, *, gamma: float, min_alpha: int, max_alpha: int) -> float:
    x = (frame_idx + 1) / n
    return (min_alpha + (max_alpha - min_alpha) * (x**gamma)) / 255.0


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    _require_cv2()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1) > 0


def contour_mask(mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) <= 8:
        return mask
    _require_cv2()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=1) > 0
    edge = mask & ~eroded
    return edge if edge.any() else mask


def render_tracks(
    frames: list[np.ndarray],
    tracks: list[Track],
    *,
    include_kinds: set[str] | None = None,
    render_all_history: bool = False,
) -> np.ndarray:
    n = len(frames)
    H, W = frames[0].shape[:2]
    current_idx = n - 1
    base = frames[-1].astype(np.float32)
    result = base.copy()
    rows = np.arange(H)[:, None]
    in_playfield = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    current_foreground = np.any(frames[-1] != 0, axis=-1) & in_playfield

    if render_all_history:
        params = {
            "ship": {"gamma": 2.3, "min_alpha": 0, "max_alpha": 72, "dilate": 0, "linger": n},
            "enemy": {"gamma": 2.3, "min_alpha": 0, "max_alpha": 78, "dilate": 0, "linger": n},
            "player_bullet": {"gamma": 1.8, "min_alpha": 4, "max_alpha": 120, "dilate": 1, "linger": n},
            "enemy_bullet": {"gamma": 1.8, "min_alpha": 4, "max_alpha": 120, "dilate": 1, "linger": n},
        }
    else:
        params = {
            "ship": {"gamma": 1.4, "min_alpha": 0, "max_alpha": 150, "dilate": 0, "linger": 0},
            "enemy": {"gamma": 1.8, "min_alpha": 0, "max_alpha": 130, "dilate": 0, "linger": 1},
            "player_bullet": {"gamma": 0.9, "min_alpha": 35, "max_alpha": 235, "dilate": 1, "linger": 3},
            "enemy_bullet": {"gamma": 0.9, "min_alpha": 35, "max_alpha": 235, "dilate": 1, "linger": 3},
        }

    draw_ops = []
    for track in tracks:
        if include_kinds is not None and track.kind not in include_kinds:
            continue
        if not track.history:
            continue
        p = params[track.kind]
        if current_idx - track.history[-1].frame_idx > p["linger"]:
            continue

        history = track.history if (render_all_history or "bullet" in track.kind) else truncate_by_direction(track.history)
        ghosts = history[:-1] if history[-1].frame_idx == current_idx else history
        for comp in ghosts:
            draw_ops.append((comp.frame_idx, track.kind, comp, p))

    for _, kind, comp, p in sorted(draw_ops, key=lambda item: item[0]):
        alpha = alpha_for_frame(
            comp.frame_idx,
            n,
            gamma=p["gamma"],
            min_alpha=p["min_alpha"],
            max_alpha=p["max_alpha"],
        )
        mask = dilate_mask(comp.mask, int(p["dilate"])) if "bullet" in kind else contour_mask(comp.mask)
        mask = mask & ~current_foreground
        if not mask.any() or alpha <= 0.0:
            continue
        if "bullet" in kind:
            color = comp.mean_color.astype(np.float32)
            result[mask] = result[mask] * (1.0 - alpha) + color * alpha
        else:
            source = frames[comp.frame_idx].astype(np.float32)
            result[mask] = result[mask] * (1.0 - alpha) + source[mask] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def collect_tracks(frames: list[np.ndarray]) -> list[Track]:
    frame_components = [foreground_components(fr, i) for i, fr in enumerate(frames)]
    tracks: list[Track] = []

    st = ship_track(frames)
    if st is not None:
        tracks.append(st)

    small_tracks = build_tracks(frame_components, bullet_like=True, max_gap=1)
    for history in small_tracks:
        kind = classify_bullet_track(history)
        if kind is not None:
            tracks.append(Track(kind, history))

    large_tracks = build_tracks(frame_components, bullet_like=False, max_gap=2)
    current_idx = len(frames) - 1
    for history in large_tracks:
        if len(history) >= 3 and current_idx - history[-1].frame_idx <= 1:
            tracks.append(Track("enemy", history))

    return tracks


def track_summary(tracks: list[Track]) -> str:
    counts = {"ship": 0, "enemy": 0, "player_bullet": 0, "enemy_bullet": 0}
    lines = []
    for track in tracks:
        counts[track.kind] += 1
        first = track.history[0]
        last = track.history[-1]
        x0, y0 = first.centroid
        x1, y1 = last.centroid
        lines.append(
            f"{track.kind:14s} len={len(track.history):2d} frames={first.frame_idx:02d}->{last.frame_idx:02d} "
            f"centroid=({x0:.1f},{y0:.1f})->({x1:.1f},{y1:.1f})"
        )
    header = " ".join(f"{k}={v}" for k, v in counts.items())
    return header + "\n" + "\n".join(lines) + "\n"


def debug_tracks_image(base: np.ndarray, tracks: list[Track]) -> np.ndarray:
    colors = {
        "ship": (255, 80, 255),
        "enemy": (120, 180, 255),
        "player_bullet": (80, 255, 120),
        "enemy_bullet": (255, 180, 80),
    }
    im = Image.fromarray(base).convert("RGB").resize((base.shape[1] * 3, base.shape[0] * 3), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    for track in tracks:
        pts = [(int(round(x * 3)), int(round(y * 3))) for x, y in [c.centroid for c in track.history]]
        if len(pts) >= 2:
            draw.line(pts, fill=colors[track.kind], width=2)
        for x, y in pts:
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=colors[track.kind])
    return np.array(im)


def make_sheet(images: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    tiles = []
    for label, arr in images:
        im = Image.fromarray(arr).convert("RGB").resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (im.width, im.height + 18), "white")
        tile.paste(im, (0, 18))
        ImageDraw.Draw(tile).text((3, 3), label, fill=(0, 0, 0))
        tiles.append(tile)

    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 2)
        sheet.paste(tile, (c * w, r * h))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--window-len", type=int, default=16)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    out_root = Path(args.out_dir)
    samples = parse_sample_positions(sample_dir)

    print(f"Loading episode {args.episode_idx} from {DATASET_NAME} ({DATASET_SUBDIR})...")
    ds = load_episode_rows(args.episode_idx, args.cache_dir)
    print(f"Episode {args.episode_idx} has {len(ds)} rows.")

    out_root.mkdir(parents=True, exist_ok=True)
    for set_name, pos in samples:
        start = max(0, pos - args.window_len + 1)
        rows = ds.select(list(range(start, pos + 1)))
        steps = list(rows["decision_step"])
        frames = [np.array(img.convert("RGB")) for img in rows["image"]]
        tracks = collect_tracks(frames)

        out_dir = out_root / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (step, frame) in enumerate(zip(steps, frames)):
            tag = "current" if i == len(frames) - 1 else "raw"
            Image.fromarray(frame).save(out_dir / f"frame_{i:02d}_{tag}_step{step}.png")

        final = render_tracks(frames, tracks)
        bullets = render_tracks(frames, tracks, include_kinds={"player_bullet", "enemy_bullet"})
        large = render_tracks(frames, tracks, include_kinds={"ship", "enemy"})
        debug = debug_tracks_image(frames[-1], tracks)

        Image.fromarray(final).save(out_dir / "ghost_trail_composite.png")
        Image.fromarray(bullets).save(out_dir / "bullet_only_composite.png")
        Image.fromarray(large).save(out_dir / "ship_enemy_only_composite.png")
        Image.fromarray(debug).save(out_dir / "debug_tracks_3x.png")
        (out_dir / "track_summary.txt").write_text(track_summary(tracks))
        make_sheet(
            [
                ("current", frames[-1]),
                ("bullet-only", bullets),
                ("ship/enemy-only", large),
                ("combined", final),
            ],
            out_dir / "comparison_sheet_2x.png",
        )
        print(f"{set_name}: raw_frames={len(frames)} tracks={len(tracks)} -> {out_dir}")

    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
