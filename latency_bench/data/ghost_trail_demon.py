"""Bullet-aware Demon Attack ghost-trail image transform for dataset export.

Ported verbatim (logic + parameters) from
``starVLA/examples/rl_games/scripts/build_bullet_aware_ghost_trails_demon_attack.py``
so the exported dataset trails are pixel-faithful to the validated
``continuous_30`` ablation (rendered there with ``render_all_history=True``).

The transform composites a causal ghost trail of the moving entities (player
ship, demons, and bullets) onto the current frame, using a window of the last
``history_frames`` consecutive raw frames. It is dataset-export only; the eval
side is intentionally not wired here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    cv2 = None


# Demon Attack raw ALE screen geometry / palette (native 210x160 RGB).
DEMON_ATTACK_SHIP_RGB = (184, 70, 162)
SCORE_ROW_END = 20
GROUND_ROW_START = 188
MAX_FOREGROUND_COMPONENT_AREA = 500


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for demon_attack_ghost_trail image export")


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


def ship_track(frames: Sequence[np.ndarray]) -> Track | None:
    history: list[Component] = []
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
    return (min_alpha + (max_alpha - min_alpha) * (x ** gamma)) / 255.0


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
    frames: Sequence[np.ndarray],
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


def collect_tracks(frames: Sequence[np.ndarray]) -> list[Track]:
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


@dataclass
class _FrameCache:
    """Per-frame data pre-computed once and shared across all sliding windows."""
    fg_comps: list[tuple]  # (centroid, mask, area, bbox, mean_color) — no frame_idx
    ship: tuple | None     # (mask, centroid, area, bbox) or None if absent


def _cache_frame(frame: np.ndarray) -> _FrameCache:
    """Pre-compute foreground components and ship presence for one raw frame.

    Equivalent to calling ``foreground_components`` + the ship pass of
    ``ship_track``, but without committing to a window-relative frame index so
    the result can be reused for every window that contains this frame.
    """
    _require_cv2()
    frame = np.asarray(frame, dtype=np.uint8)
    H, W = frame.shape[:2]

    not_black = np.any(frame != 0, axis=-1)
    rows = np.arange(H)[:, None]
    in_band = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    not_ship = ~np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    fg = not_black & in_band & not_ship
    num, labels = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    fg_comps: list[tuple] = []
    for lab in range(1, num):
        mask = labels == lab
        area = int(mask.sum())
        if area < 1 or area > MAX_FOREGROUND_COMPONENT_AREA:
            continue
        ys, xs = np.where(mask)
        fg_comps.append((
            (float(xs.mean()), float(ys.mean())),
            mask,
            area,
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            frame[mask].mean(axis=0),
        ))

    ship_mask = np.all(frame == np.array(DEMON_ATTACK_SHIP_RGB), axis=-1)
    if ship_mask.any():
        ys, xs = np.where(ship_mask)
        ship: tuple | None = (
            ship_mask,
            (float(xs.mean()), float(ys.mean())),
            int(ship_mask.sum()),
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        )
    else:
        ship = None

    return _FrameCache(fg_comps=fg_comps, ship=ship)


def _tracks_from_cache(cache_window: list[_FrameCache]) -> list[Track]:
    """Build tracks from pre-computed per-frame caches with window-relative indices.

    Drop-in replacement for ``collect_tracks`` that avoids re-running
    ``foreground_components`` and the ship scan for frames already cached.
    """
    frame_components = [
        [
            Component(frame_idx=i, centroid=c, mask=m, area=a, bbox=b, mean_color=mc)
            for c, m, a, b, mc in pc.fg_comps
        ]
        for i, pc in enumerate(cache_window)
    ]
    tracks: list[Track] = []

    ship_history: list[Component] = []
    for i, pc in enumerate(cache_window):
        if pc.ship is not None:
            s_mask, s_centroid, s_area, s_bbox = pc.ship
            ship_history.append(Component(
                frame_idx=i,
                centroid=s_centroid,
                mask=s_mask,
                area=s_area,
                bbox=s_bbox,
                mean_color=np.array(DEMON_ATTACK_SHIP_RGB, dtype=np.float32),
            ))
    if ship_history:
        tracks.append(Track("ship", ship_history))

    small_tracks = build_tracks(frame_components, bullet_like=True, max_gap=1)
    for history in small_tracks:
        kind = classify_bullet_track(history)
        if kind is not None:
            tracks.append(Track(kind, history))

    large_tracks = build_tracks(frame_components, bullet_like=False, max_gap=2)
    current_idx = len(cache_window) - 1
    for history in large_tracks:
        if len(history) >= 3 and current_idx - history[-1].frame_idx <= 1:
            tracks.append(Track("enemy", history))

    return tracks


def build_demon_attack_ghost_trail_window(
    frames: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    config,
) -> np.ndarray:
    """Build one demon-attack ghost-trail image from a causal raw-frame window.

    ``frames`` is chronological with the current frame last. ``steps`` is unused
    by the renderer (entities are tracked by positional frame index within the
    window) but kept for signature parity with the flappy builder. Matches the
    validated ``continuous_30`` ablation by rendering with
    ``render_all_history=True``.
    """
    del steps  # entities are tracked positionally within the window
    if len(frames) == 0:
        raise ValueError("frames must contain at least the current frame")
    window = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if len(window) == 1:
        return window[-1].copy()
    tracks = collect_tracks(window)
    return render_tracks(window, tracks, render_all_history=True)


def build_demon_attack_ghost_trail_episode_frames(
    frames: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    config,
) -> dict[int, np.ndarray]:
    """Build one ghost-trail image per decision step from a continuous raw window.

    For each decision step the window is the last ``config.history_frames``
    consecutive raw frames (inclusive of the current frame); for the first
    ``< history_frames`` steps all available past frames in the episode are used.

    Foreground components and ship data are pre-computed once per frame and
    reused across all windows, reducing ``foreground_components`` calls from
    O(N * window_len) to O(N).
    """
    if len(frames) != len(steps):
        raise ValueError("frames and steps must have the same length")
    window_len = max(1, int(config.history_frames))

    uint8_frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    frame_cache = [_cache_frame(f) for f in uint8_frames]

    out: dict[int, np.ndarray] = {}
    for idx, step in enumerate(steps):
        start = max(0, idx - window_len + 1)
        window_frames = uint8_frames[start : idx + 1]
        if len(window_frames) == 1:
            out[int(step)] = window_frames[-1].copy()
            continue
        tracks = _tracks_from_cache(frame_cache[start : idx + 1])
        out[int(step)] = render_tracks(window_frames, tracks, render_all_history=True)
    return out
