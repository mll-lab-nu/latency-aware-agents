"""Ghost-trail image patching for RL-games frames.

Builds a single composite image where the most recent frame is the
authoritative background (always correct), and up to N earlier frames
contribute fading "ghost" cutouts of a tracked object at its past on-screen
positions -- giving a model implicit access to recent motion/velocity that a
single frame can't show.

The core functions (`select_ghost_frames`, `build_ghost_trail_image`) are
game-agnostic: pass a `segment_fn` that extracts the object to trail, and
(optionally) an `occlusion_fn` plus `steps`/`scroll_px_per_step` for games
where the camera follows the object and the background scrolls.

Flappy Bird (the first game this was built for) needs both extras: its world
scrolls (pipes move left over time) while the bird stays at a roughly fixed
screen x, so a ghost's old (x, y) position can be occluded by a pipe that has
since scrolled into that exact spot, and on-screen position alone
understates the bird's true motion through the level. Ghosts are clipped to
the *current* frame's pipe/ground mask before compositing (recomputed fresh
every time, so it stays correct for every timestep, not just frames where
the background happens to be unchanged) and shifted by the measured scroll
rate to reproject them into the current frame's reference.

Demon Attack (static camera, no scroll) needs neither: on-screen position
already is the true motion, and there's nothing that should occlude the
trail, so it can use `occlusion_fn=None` and omit `steps`.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def _natural_key(p: str):
    # Sort frame2.png before frame10.png
    base = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def load_frames_from_paths(image_paths: Sequence[str]) -> List[np.ndarray]:
    """Load a sequence of image file paths into naturally-sorted RGB arrays."""
    paths = sorted(image_paths, key=_natural_key)
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for ghost-trail patching (pip install opencv-python)")


def _make_pipe_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Mask of pipe (and bush) colored pixels.

    Cast to int16 before +/- comparisons to avoid uint8 wraparound.
    """
    _require_cv2()
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Pipes/bushes are yellow-green-ish; sky is ~H=90+, excluded by the upper bound.
    pipe_hsv = cv2.inRange(
        hsv,
        np.array([30, 30, 40], np.uint8),
        np.array([75, 255, 255], np.uint8),
    ).astype(bool)

    Ri = frame_rgb[..., 0].astype(np.int16)
    Gi = frame_rgb[..., 1].astype(np.int16)
    Bi = frame_rgb[..., 2].astype(np.int16)
    # Require real saturation in addition to "G is somewhat higher than R/B" --
    # otherwise near-white/cream pixels (e.g. a bird sprite's eye highlight,
    # RGB ~(215, 230, 204)) can satisfy the G-dominance check by a wide enough
    # margin to false-positive as "pipe", punching a hole in the bird mask.
    # Real pipe greens are heavily desaturated from white (e.g. (85, 128, 34)).
    saturation = hsv[..., 1].astype(np.int16)
    pipe_rgb = (Gi > 70) & ((Gi - Ri) > 12) & ((Gi - Bi) > 12) & (saturation > 40)

    pipe = pipe_hsv | pipe_rgb

    # Dilate so pipe edges/caps don't leak into the bird mask.
    pipe_u8 = pipe.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    pipe_u8 = cv2.dilate(pipe_u8, k, iterations=1)
    return pipe_u8.astype(bool)


def _make_ground_mask(frame_rgb: np.ndarray, ground_fraction: float = 0.22) -> np.ndarray:
    """Mask of the ground band (bushes/stripe/dirt) at the bottom of the frame.

    This renderer always places the ground at the same fixed bottom slice of
    the frame, so a fixed-row cutoff is more robust here than a color
    heuristic: it can't false-positive on sky/clouds, and it covers the dirt
    strip that isn't green enough for the pipe-color heuristic to catch.
    Measured empirically: the bush/stripe/dirt band starts at ~78% of frame
    height, so a 22% bottom slice covers it with a small safety margin.
    """
    H, W = frame_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    ground_start = int(H * (1.0 - ground_fraction))
    mask[ground_start:, :] = True
    return mask


def _make_occlusion_mask(frame_rgb: np.ndarray, ground_fraction: float = 0.22) -> np.ndarray:
    """Everything in the *current* frame that should occlude a ghost.

    Recomputed fresh per-frame from the current frame alone, so it stays
    correct regardless of how far pipes have scrolled since a ghost's source
    frame was captured.
    """
    return _make_pipe_mask(frame_rgb) | _make_ground_mask(frame_rgb, ground_fraction)


def _segment_bird(
    frame_rgb: np.ndarray,
    occlusion_mask: Optional[np.ndarray] = None,
    sky_patch: int = 40,
    sky_tol: int = 6,
) -> Optional[np.ndarray]:
    """Segment the bird sprite via color, without using motion.

    Builds a core (orange body + white belly), expands it to include the dark
    outline, excludes pipes/ground, then picks the best connected component
    near the bird's typical x. Finally strips any pixels matching the frame's
    dominant sky color, which removes stray blue/cloud leakage from dilation.
    """
    _require_cv2()
    H, W = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    orange = cv2.inRange(
        hsv,
        np.array([5, 70, 70], np.uint8),
        np.array([40, 255, 255], np.uint8),
    ).astype(bool)
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 210], np.uint8),
        np.array([180, 70, 255], np.uint8),
    ).astype(bool)

    occlusion = occlusion_mask if occlusion_mask is not None else _make_occlusion_mask(frame_rgb)
    core = (orange | white) & (~occlusion)

    core_u8 = core.astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_u8 = cv2.dilate(core_u8, k5, iterations=2)

    dark = hsv[..., 2] < 120  # outline pixels
    cand = core_u8.astype(bool) & (~occlusion) & (dark | orange | white)

    cand_u8 = cand.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, k3, iterations=1)
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE, k3, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    if num <= 1:
        return None

    target_x = int(W * 0.35)  # bird is usually around here
    best = None
    best_score = None
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if not (80 <= area <= 9000):
            continue
        comp = labels == lab
        core_cnt = int((core & comp).sum())
        if core_cnt < 30:
            continue
        cx, _ = cents[lab]
        dist = abs(cx - target_x)
        # Prefer lots of core pixels, then closer x, then larger area.
        score = (-core_cnt, dist, -area)
        if best is None or score < best_score:
            best, best_score = lab, score

    if best is None:
        return None

    mask = labels == best

    # Remove pixels matching the frame's dominant sky color.
    p = min(sky_patch, H, W)
    patch = frame_rgb[:p, :p].reshape(-1, 3)
    colors, counts = np.unique(patch, axis=0, return_counts=True)
    sky_rgb = colors[int(np.argmax(counts))].astype(np.int16)
    Ri = frame_rgb[..., 0].astype(np.int16)
    Gi = frame_rgb[..., 1].astype(np.int16)
    Bi = frame_rgb[..., 2].astype(np.int16)
    d = np.abs(Ri - sky_rgb[0]) + np.abs(Gi - sky_rgb[1]) + np.abs(Bi - sky_rgb[2])
    mask = mask & ~(d <= sky_tol)

    # Keep only the largest remaining component (removes tiny stray dots).
    mask_u8 = mask.astype(np.uint8) * 255
    num2, lab2, stats2, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num2 > 1:
        best2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        mask = lab2 == best2

    return mask


def make_exact_color_segmenter(target_rgb: "tuple[int, int, int]", tolerance: int = 0):
    """Build a `segment_fn(frame_rgb, occlusion_mask=None) -> Optional[mask]`
    that matches pixels within L1 `tolerance` of `target_rgb`.

    Use for sprites that render as an exact, unchanging solid color (e.g.
    classic Atari sprites like Demon Attack's ship) -- far more robust than
    HSV/morphology heuristics when the color truly doesn't vary, and much
    simpler than what Flappy Bird's anti-aliased sprite needed.
    """
    target = np.array(target_rgb, dtype=np.int16)

    def _segment(frame_rgb: np.ndarray, occlusion_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        diff = np.abs(frame_rgb.astype(np.int16) - target).sum(axis=-1)
        mask = diff <= tolerance
        if occlusion_mask is not None:
            mask = mask & ~occlusion_mask
        return mask if mask.any() else None

    return _segment


# Empirically measured: Demon Attack's player ship renders as this exact RGB,
# always exactly 44 pixels, with zero variation observed across episodes
# 0, 1, 2, 5, and 10 (latency-sensitive-bench/benchmark-datasets, latency 0).
DEMON_ATTACK_SHIP_RGB = (184, 70, 162)


def _object_centroid(
    frame_rgb: np.ndarray,
    segment_fn=_segment_bird,
    occlusion_mask: Optional[np.ndarray] = None,
) -> Optional[tuple]:
    mask = segment_fn(frame_rgb, occlusion_mask)
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


# Empirically measured (constrained cross-correlation search over multiple
# points in a flappy_200ep episode, latency 0): the world scrolls left at an
# exact, constant 4.0 px per raw decision_step, with zero observed variance.
# This holds because the bird's screen-x is fixed by the camera and the pipes
# scroll at a fixed game speed -- it is specific to this renderer/dataset.
PIPE_SCROLL_PX_PER_STEP = 4.0


def _shift_horizontal(image: np.ndarray, shift_x: float) -> np.ndarray:
    """Shift `image` left by `shift_x` px, filling the revealed area with
    zeros (no wraparound). Works for uint8 RGB frames and uint8 0/255 masks."""
    if abs(shift_x) < 1e-6:
        return image
    _require_cv2()
    H, W = image.shape[:2]
    M = np.array([[1, 0, -float(shift_x)], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def select_ghost_frames(
    history: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    trail_len: int = 7,
    min_pixel_gap: float = 16.0,
    scroll_px_per_step: float = PIPE_SCROLL_PX_PER_STEP,
    segment_fn=_segment_bird,
) -> "tuple[List[np.ndarray], List[int]]":
    """Pick up to `trail_len` ghost frames from `history` that actually look
    different from each other, accounting for both the tracked object's own
    on-screen motion AND any world scroll (e.g. Flappy Bird's bird barely
    moves on-screen, so vertical-only distance understates how visually
    separated two frames will actually end up once re-projected into the
    current frame -- see `build_ghost_trail_image`). Pass `scroll_px_per_step
    =0` for static-background games (e.g. Demon Attack), where on-screen
    position already is the true motion.

    `history` and `steps` are chronologically ordered and the same length,
    with `history[-1]`/`steps[-1]` the current frame. Walks backward and only
    keeps a candidate once its *effective* position (its own centroid plus
    scroll-implied horizontal offset from the previously kept position) has
    moved at least `min_pixel_gap`. Without this, a fixed "previous N rows"
    window frequently lands entirely within one sprite-height of motion (e.g.
    near the apex of a Flappy Bird flap, or while an agent holds still), so
    ghosts would overlap almost completely.

    `segment_fn` defaults to the Flappy Bird bird segmenter; pass a different
    one (e.g. from :func:`make_exact_color_segmenter`) for other games.

    Returns `(ghost_frames, ghost_steps)`, oldest-first, ready to pass (with
    the current frame/step appended) to :func:`build_ghost_trail_image`.
    """
    if len(history) < 2:
        return [], []

    current_step = steps[-1]
    ref_centroid = _object_centroid(history[-1], segment_fn)
    ref_step = current_step
    selected: List[np.ndarray] = []
    selected_steps: List[int] = []
    for fr, step in zip(reversed(history[:-1]), reversed(steps[:-1])):
        centroid = _object_centroid(fr, segment_fn)
        if centroid is None:
            continue
        if ref_centroid is not None:
            # Combine the scroll-implied offset (the part of motion hidden
            # from on-screen position when the camera follows the object,
            # e.g. Flappy Bird) with the object's own measured horizontal
            # displacement (the *only* signal when there's no scroll, e.g.
            # Demon Attack's ship, which moves on-screen with zero camera
            # compensation). For Flappy this just adds a small, real jitter
            # term; for Demon Attack the scroll term is 0 and this reduces to
            # plain on-screen distance.
            dx = scroll_px_per_step * (ref_step - step) + (centroid[0] - ref_centroid[0])
            dy = centroid[1] - ref_centroid[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < min_pixel_gap:
                continue
        selected.append(fr)
        selected_steps.append(step)
        ref_centroid = centroid
        ref_step = step
        if len(selected) >= trail_len:
            break

    selected.reverse()  # oldest first
    selected_steps.reverse()
    return selected, selected_steps


def build_ghost_trail_image(
    frames: Sequence[np.ndarray],
    steps: Optional[Sequence[int]] = None,
    *,
    gamma: float = 0.7,
    min_alpha: int = 35,
    ground_fraction: float = 0.22,
    scroll_px_per_step: float = PIPE_SCROLL_PX_PER_STEP,
    segment_fn=_segment_bird,
    occlusion_fn=_make_occlusion_mask,
) -> np.ndarray:
    """Composite a ghost trail onto the LAST frame in `frames`.

    Args:
        frames: chronologically ordered RGB frames. ``frames[-1]`` is the
            current frame, used untouched as the background. ``frames[:-1]``
            are the trail, oldest first, each rendered as a fading cutout of
            whatever `segment_fn` extracts, in its original color, clipped
            to `occlusion_fn`'s mask of the current frame (if any).
        steps: optional decision_step (or other monotonic raw-frame counter)
            for each entry in `frames`, same length and order. When given,
            each ghost is shifted left by ``scroll_px_per_step * (steps[-1] -
            steps[i])`` before compositing -- for games where the camera
            follows the tracked object so its on-screen position barely
            moves (e.g. Flappy Bird), without this a ghost only shows local
            bob, not the world distance actually covered since that frame.
            Pass `scroll_px_per_step=0` (or omit `steps`) for static-camera
            games (e.g. Demon Attack), where on-screen position already is
            the true motion.
        gamma: <1 makes older ghosts more visible (less aggressive fade).
        min_alpha: opacity floor (0-255) for the oldest ghost.
        ground_fraction: bottom fraction of the frame treated as ground,
            forwarded to `occlusion_fn` (ignored if `occlusion_fn` is None or
            doesn't accept it).
        segment_fn: `(frame_rgb, occlusion_mask=None) -> Optional[mask]`.
            Defaults to the Flappy Bird bird segmenter; pass a different one
            (e.g. from :func:`make_exact_color_segmenter`) for other games.
        occlusion_fn: `(frame_rgb, ground_fraction) -> mask` of current-frame
            pixels that should hide any ghost drawn under them (e.g. Flappy
            Bird's pipes/ground). Pass `None` for games with nothing that
            should occlude the trail (e.g. Demon Attack's static background).

    Returns:
        An RGB array the same shape as ``frames[-1]``.
    """
    if not frames:
        raise ValueError("frames must contain at least the current frame")

    base = frames[-1]
    H, W = base.shape[:2]

    ghost_frames = frames[:-1]
    if not ghost_frames:
        return base.copy()

    occlusion = occlusion_fn(base, ground_fraction) if occlusion_fn is not None else np.zeros((H, W), dtype=bool)
    current_step = steps[-1] if steps is not None else None

    # Accumulate every ghost's contribution per pixel instead of sequentially
    # alpha-compositing one on top of the next. Sequential "over" compositing
    # lets a more opaque, newer ghost paint over almost all of an older one
    # wherever they overlap (the bird moves little relative to its own sprite
    # size between consecutive samples, so overlap is heavy) -- that erases
    # most of the trail. Accumulation instead makes overlapping ghosts build
    # up into a denser, more visible shadow (closer to a real motion-blur
    # trail), while isolated ghosts still show through faintly on their own.
    L = len(ghost_frames)
    alpha_acc = np.zeros((H, W), dtype=np.float32)
    color_acc = np.zeros((H, W, 3), dtype=np.float32)

    for idx, fr in enumerate(ghost_frames):
        mask = segment_fn(fr)
        if mask is None or not mask.any():
            continue

        if current_step is not None:
            shift_x = scroll_px_per_step * (current_step - steps[idx])
            mask = _shift_horizontal(mask.astype(np.uint8) * 255, shift_x) > 127
            fr = _shift_horizontal(fr, shift_x)
            if not mask.any():
                continue

        # Clip the ghost to areas the current frame doesn't already occlude,
        # so it never appears to float through an obstacle that has since
        # moved/scrolled into its old position.
        mask = mask & (~occlusion)
        if not mask.any():
            continue

        x = (idx + 1) / L  # (0, 1], oldest -> smallest, newest ghost -> 1.0
        alpha_value = (min_alpha + (255 - min_alpha) * (x ** gamma)) / 255.0

        alpha_acc[mask] += alpha_value
        color_acc[mask] += alpha_value * fr[mask].astype(np.float32)

    has_ghost = alpha_acc > 1e-6
    final_alpha = np.clip(alpha_acc, 0.0, 1.0)
    blended_color = np.zeros((H, W, 3), dtype=np.float32)
    blended_color[has_ghost] = color_acc[has_ghost] / alpha_acc[has_ghost, None]

    result = base.astype(np.float32).copy()
    result[has_ghost] = (
        base[has_ghost].astype(np.float32) * (1.0 - final_alpha[has_ghost, None])
        + blended_color[has_ghost] * final_alpha[has_ghost, None]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def isolate_and_overlay_ghost_trail(image_paths: Sequence[str], out_path: str, **kwargs) -> str:
    """Convenience wrapper: load files in natural order, build the ghost trail
    for the last frame, and save it to `out_path`.

    Extra kwargs are forwarded to :func:`build_ghost_trail_image`.
    """
    frames = load_frames_from_paths(image_paths)
    result = build_ghost_trail_image(frames, **kwargs)
    Image.fromarray(result).save(out_path)
    return out_path


# Example:
# out = isolate_and_overlay_ghost_trail(
#     image_paths=sorted(glob("frames/*.png")),
#     out_path="ghost.png",
# )
# print("saved:", out)
