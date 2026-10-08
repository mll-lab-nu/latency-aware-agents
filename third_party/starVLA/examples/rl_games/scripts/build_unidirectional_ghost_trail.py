#!/usr/bin/env python
"""Build unidirectional ghost-trail composites from the existing bidirectional
test outputs, for a side-by-side comparison.

Reuses the exact same per-set candidate frames already saved by
test_image_patching_demon_attack.py under
ghost_trail_test_outputs_demon_attack_bidiirection/set_*/ (those are already
the bidirectionally-selected ghosts + current frame). For each set, this
further prunes the ghost sequence so the trail only extends backward through
frames that move in the same direction the ship is moving *right now* (sign
of the most recent ghost -> current displacement, with "no movement" also
treated as its own direction). The first backward step whose direction sign
differs from that reference truncates the trail there, instead of folding
back over itself the way the bidirectional trail does on a reversal.

Usage:
    python examples/rl_games/scripts/build_unidirectional_ghost_trail.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.image_patching import (
    DEMON_ATTACK_SHIP_RGB,
    build_ghost_trail_image,
    make_exact_color_segmenter,
)

BIDIRECTION_DIR = Path(__file__).parent / "ghost_trail_test_outputs_demon_attack_bidiirection"
UNIDIRECTION_DIR = Path(__file__).parent / "ghost_trail_test_outputs_demon_attack_unidirection"

segment_ship = make_exact_color_segmenter(DEMON_ATTACK_SHIP_RGB)


def _ship_centroid_x(frame: np.ndarray) -> float:
    mask = segment_ship(frame)
    if mask is None or not mask.any():
        raise ValueError("could not segment ship in frame")
    xs = np.where(mask)[1]
    return float(xs.mean())


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def truncate_to_unidirectional(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Drop ghosts older than the first direction reversal, walking backward
    from the current frame.

    `frames` must be chronologically ordered, oldest first, with frames[-1]
    the current frame.
    """
    n = len(frames)
    if n < 2:
        return frames

    centroids = [_ship_centroid_x(fr) for fr in frames]
    ref_sign = _sign(centroids[-1] - centroids[-2])

    kept = [n - 2]  # the most recent ghost always matches the reference by construction
    for i in range(n - 3, -1, -1):
        if _sign(centroids[i + 1] - centroids[i]) != ref_sign:
            break
        kept.append(i)

    kept.sort()
    return [frames[i] for i in kept] + [frames[-1]]


def main() -> None:
    if not BIDIRECTION_DIR.is_dir():
        raise FileNotFoundError(f"expected bidirectional outputs at {BIDIRECTION_DIR}")

    UNIDIRECTION_DIR.mkdir(parents=True, exist_ok=True)

    for set_dir in sorted(BIDIRECTION_DIR.iterdir()):
        if not set_dir.is_dir():
            continue

        frame_paths = sorted(set_dir.glob("frame_*.png"))
        if not frame_paths:
            continue
        frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

        out_dir = UNIDIRECTION_DIR / set_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in set_dir.glob("*.png"):
            shutil.copy2(p, out_dir / p.name)

        kept_frames = truncate_to_unidirectional(frames)
        composite = build_ghost_trail_image(kept_frames, segment_fn=segment_ship, occlusion_fn=None)
        Image.fromarray(composite).save(out_dir / "ghost_trail_composite.png")

        print(
            f"{set_dir.name}: {len(frames) - 1} ghosts -> {len(kept_frames) - 1} kept (unidirectional) -> {out_dir}"
        )

    print(f"Done. Outputs in {UNIDIRECTION_DIR}")


if __name__ == "__main__":
    main()
