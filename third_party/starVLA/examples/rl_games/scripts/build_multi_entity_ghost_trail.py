#!/usr/bin/env python
"""Unidirectional ghost trails for *every* entity in Demon Attack (player
ship, player bullets, enemy ships, enemy bullets) -- not just the ship.

Reuses the exact same per-set candidate frames already saved by
test_image_patching_demon_attack.py under
ghost_trail_test_outputs_demon_attack_bidiirection/set_*/.

Why this can't just reuse N more exact-color segmenters like the ship's:
enemy color cycles per wave/episode (empirically: blue in one episode, olive
in another, for the same sprite shape), so a hardcoded RGB would silently
stop matching on a different wave. Instead, entities other than the ship are
found via background subtraction: every pixel that isn't pure black, isn't
inside the fixed score-text band (rows < SCORE_ROW_END) or the fixed
ground/water/lives-icon band (rows >= GROUND_ROW_START), and isn't the
ship's own color, is a foreground candidate. Connected components on that
mask give per-frame blobs (one blob can be a whole enemy, a bullet, a piece
of an enemy, or a one-off explosion particle).

Blobs are linked across frames into per-entity tracks with greedy
nearest-centroid matching (gated by MAX_MATCH_DIST). Tracks shorter than
MIN_TRACK_LEN frames are dropped -- this is what filters out one-off
explosion debris, which doesn't persist as a single coherent blob the way a
real moving sprite does. Surviving tracks that reach the current frame each
get the same unidirectional truncation as the ship's trail, generalized from
a 1D sign check to a 2D one: walking backward from the current frame, keep a
step only while its displacement vector has a non-negative dot product with
the reference (current) displacement vector -- i.e. points in the same
general direction, not just the same horizontal sign -- and stop at the
first reversal. A static reference (entity not currently moving) only
continues through other exactly-static steps.

All surviving, truncated tracks (ship included) are accumulated into one
shared alpha/color buffer -- exactly like build_ghost_trail_image does for
the ship alone -- so overlapping ghosts from different entities (e.g. a
bullet passing near the ship) blend the same way a single entity's
overlapping ghosts do, rather than separate sequential composites painting
over each other.

Usage:
    python examples/rl_games/scripts/build_multi_entity_ghost_trail.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.image_patching import DEMON_ATTACK_SHIP_RGB, _require_cv2

import cv2

BIDIRECTION_DIR = Path(__file__).parent / "ghost_trail_test_outputs_demon_attack_bidiirection"
OUT_DIR = Path(__file__).parent / "ghost_trail_outputs_demon_attack_all"

# Empirically measured (see analysis in conversation): these rows are static
# background/HUD in every sampled frame, regardless of episode -- score text
# always starts at row 7, and the ground/water/lives-icon band always
# occupies rows 188-209 of this 210-row frame.
SCORE_ROW_END = 20
GROUND_ROW_START = 188

MIN_TRACK_LEN = 3  # frames a blob must persist in to count as a real entity, not explosion debris
MAX_MATCH_DIST = 25.0  # px gate for greedy nearest-centroid matching between frames
MIN_COMPONENT_AREA = 2  # px, drops stray single-pixel noise

GAMMA = 0.7
MIN_ALPHA = 35
MAX_ALPHA = 255


def foreground_components(frame: np.ndarray) -> list[tuple[tuple[float, float], np.ndarray]]:
    """Return [(centroid_xy, mask), ...] for every non-ship foreground blob."""
    _require_cv2()
    H, W = frame.shape[:2]

    not_black = np.any(frame != 0, axis=-1)
    rows = np.arange(H)[:, None]
    in_band = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    not_ship = ~np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    fg = not_black & in_band & not_ship

    # Dilate only to merge a single sprite's disconnected sub-parts (e.g. an
    # enemy's body + its accent dot) into one blob for labeling; the actual
    # per-component mask used for rendering is intersected back with the
    # un-dilated `fg` so ghosts keep crisp original edges.
    fg_u8 = fg.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(fg_u8, k, iterations=1)

    num, labels = cv2.connectedComponents(dilated, connectivity=8)
    out = []
    for lab in range(1, num):
        comp_mask = (labels == lab) & fg
        if comp_mask.sum() < MIN_COMPONENT_AREA:
            continue
        ys, xs = np.where(comp_mask)
        centroid = (float(xs.mean()), float(ys.mean()))
        out.append((centroid, comp_mask))
    return out


def build_tracks(
    frame_components: list[list[tuple[tuple[float, float], np.ndarray]]],
) -> list[list[tuple[int, tuple[float, float], np.ndarray]]]:
    """Greedy nearest-centroid matching across frames -> list of tracks.

    Each track is a chronological list of (frame_idx, centroid, mask).
    """
    tracks: list[dict] = []
    for frame_idx, comps in enumerate(frame_components):
        used = set()
        for centroid, mask in comps:
            best_ti, best_dist = None, None
            for ti, tr in enumerate(tracks):
                if ti in used:
                    continue
                d = np.hypot(tr["last"][0] - centroid[0], tr["last"][1] - centroid[1])
                if d <= MAX_MATCH_DIST and (best_dist is None or d < best_dist):
                    best_ti, best_dist = ti, d
            if best_ti is not None:
                tracks[best_ti]["history"].append((frame_idx, centroid, mask))
                tracks[best_ti]["last"] = centroid
                used.add(best_ti)
            else:
                tracks.append({"history": [(frame_idx, centroid, mask)], "last": centroid})
    return [t["history"] for t in tracks]


def ship_track(frames: list[np.ndarray]) -> list[tuple[int, tuple[float, float], np.ndarray]]:
    history = []
    for frame_idx, frame in enumerate(frames):
        mask = np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        history.append((frame_idx, (float(xs.mean()), float(ys.mean())), mask))
    return history


def _sign_vec(v: np.ndarray) -> bool:
    return bool(np.any(v != 0))


def truncate_by_direction(
    history: list[tuple[int, tuple[float, float], np.ndarray]],
) -> list[tuple[int, tuple[float, float], np.ndarray]]:
    """Drop entries older than the first direction reversal, walking backward
    from the most recent entry. `history` is chronological; the reference
    direction is the displacement between its last two entries."""
    if len(history) < 2:
        return history

    centroids = np.array([h[1] for h in history], dtype=np.float64)
    ref_vec = centroids[-1] - centroids[-2]
    ref_moving = _sign_vec(ref_vec)

    kept = [len(history) - 2]
    for i in range(len(history) - 3, -1, -1):
        v = centroids[i + 1] - centroids[i]
        if ref_moving:
            if float(np.dot(v, ref_vec)) <= 0:
                break
        else:
            if _sign_vec(v):
                break
        kept.append(i)

    kept.sort()
    kept.append(len(history) - 1)
    return [history[i] for i in kept]


def composite_all_entities(
    frames: list[np.ndarray],
    *,
    gamma: float = GAMMA,
    min_alpha: int = MIN_ALPHA,
    max_alpha: int = MAX_ALPHA,
) -> np.ndarray:
    n = len(frames)
    H, W = frames[0].shape[:2]
    current_idx = n - 1

    frame_components = [foreground_components(fr) for fr in frames]
    tracks = [ship_track(frames)] + build_tracks(frame_components)

    alpha_acc = np.zeros((H, W), dtype=np.float32)
    color_acc = np.zeros((H, W, 3), dtype=np.float32)

    for history in tracks:
        if len(history) < MIN_TRACK_LEN:
            continue
        if history[-1][0] != current_idx:
            continue  # entity isn't present right now; nothing to trail into

        kept = truncate_by_direction(history)
        for frame_idx, _, mask in kept[:-1]:  # exclude the current-frame entry itself
            # Normalize against the current frame too, so the immediately
            # previous ghost is still below the max opacity ceiling.
            x = (frame_idx + 1) / n
            alpha_value = (min_alpha + (max_alpha - min_alpha) * (x**gamma)) / 255.0
            alpha_acc[mask] += alpha_value
            color_acc[mask] += alpha_value * frames[frame_idx][mask].astype(np.float32)

    base = frames[-1].astype(np.float32)
    has_ghost = alpha_acc > 1e-6
    final_alpha = np.clip(alpha_acc, 0.0, 1.0)
    blended_color = np.zeros((H, W, 3), dtype=np.float32)
    blended_color[has_ghost] = color_acc[has_ghost] / alpha_acc[has_ghost, None]

    result = base.copy()
    result[has_ghost] = base[has_ghost] * (1.0 - final_alpha[has_ghost, None]) + blended_color[has_ghost] * final_alpha[
        has_ghost, None
    ]
    return np.clip(result, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(BIDIRECTION_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--min-alpha", type=int, default=MIN_ALPHA)
    parser.add_argument("--max-alpha", type=int, default=MAX_ALPHA)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.out_dir)

    if not (0 <= args.min_alpha <= args.max_alpha <= 255):
        raise ValueError("expected 0 <= min_alpha <= max_alpha <= 255")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"expected bidirectional outputs at {input_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    for set_dir in sorted(input_dir.iterdir()):
        if not set_dir.is_dir():
            continue

        frame_paths = sorted(set_dir.glob("frame_*.png"))
        if not frame_paths:
            continue
        frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

        out_dir = out_root / set_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in set_dir.glob("frame_*.png"):
            Image.fromarray(np.array(Image.open(p).convert("RGB"))).save(out_dir / p.name)

        composite = composite_all_entities(
            frames,
            gamma=args.gamma,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
        )
        Image.fromarray(composite).save(out_dir / "ghost_trail_composite.png")

        print(f"{set_dir.name}: {len(frames)} frames -> {out_dir}")

    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
