"""Causal Flappy and Demon Attack motion trails shared by collection and inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

import cv2


PIPE_SCROLL_PX_PER_STEP = 4.0

SegmentFn = Callable[[np.ndarray], np.ndarray | None]


@dataclass(frozen=True)
class GhostTrailConfig:
    """Trail settings: Flappy counts past decision frames; Demon counts total raw frames."""

    image_transform: str = "raw_rgb"
    history_frames: int = 5
    gamma: float = 1.3
    min_alpha: int = 35
    ground_fraction: float = 0.22
    scroll_px_per_step: float = PIPE_SCROLL_PX_PER_STEP


def _shift_horizontal(image: np.ndarray, shift_x: float) -> np.ndarray:
    """Shift image left by ``shift_x`` pixels without wraparound."""
    if abs(float(shift_x)) < 1e-6:
        return image
    height, width = image.shape[:2]
    matrix = np.array([[1, 0, -float(shift_x)], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _make_pipe_mask(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    pipe_hsv = cv2.inRange(
        hsv,
        np.array([30, 30, 40], np.uint8),
        np.array([75, 255, 255], np.uint8),
    ).astype(bool)

    red = frame_rgb[..., 0].astype(np.int16)
    green = frame_rgb[..., 1].astype(np.int16)
    blue = frame_rgb[..., 2].astype(np.int16)
    saturation = hsv[..., 1].astype(np.int16)
    pipe_rgb = (green > 70) & ((green - red) > 12) & ((green - blue) > 12) & (saturation > 40)
    pipe = pipe_hsv | pipe_rgb

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(pipe.astype(np.uint8) * 255, kernel, iterations=1).astype(bool)


def _make_ground_mask(frame_rgb: np.ndarray, ground_fraction: float) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    mask[int(height * (1.0 - float(ground_fraction))) :, :] = True
    return mask


def _make_occlusion_mask(frame_rgb: np.ndarray, ground_fraction: float) -> np.ndarray:
    return _make_pipe_mask(frame_rgb) | _make_ground_mask(frame_rgb, ground_fraction)


def _segment_flappy_bird(frame_rgb: np.ndarray, *, ground_fraction: float = 0.22) -> np.ndarray | None:
    """Segment the Flappy bird sprite without motion or future frames."""
    height, width = frame_rgb.shape[:2]
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

    occlusion = _make_occlusion_mask(frame_rgb, ground_fraction)
    core = (orange | white) & (~occlusion)
    core_u8 = core.astype(np.uint8) * 255
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_u8 = cv2.dilate(core_u8, kernel5, iterations=2)

    dark = hsv[..., 2] < 120
    candidate = core_u8.astype(bool) & (~occlusion) & (dark | orange | white)
    candidate_u8 = candidate.astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel3, iterations=1)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel3, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    if num_labels <= 1:
        return None

    target_x = int(width * 0.35)
    best_label = None
    best_score = None
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (80 <= area <= 9000):
            continue
        component = labels == label
        core_count = int((core & component).sum())
        if core_count < 30:
            continue
        center_x, _ = centroids[label]
        score = (-core_count, abs(center_x - target_x), -area)
        if best_label is None or score < best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return None

    mask = labels == best_label
    patch_size = min(40, height, width)
    patch = frame_rgb[:patch_size, :patch_size].reshape(-1, 3)
    colors, counts = np.unique(patch, axis=0, return_counts=True)
    sky_rgb = colors[int(np.argmax(counts))].astype(np.int16)
    diff = np.abs(frame_rgb.astype(np.int16) - sky_rgb).sum(axis=-1)
    mask = mask & ~(diff <= 6)

    mask_u8 = mask.astype(np.uint8) * 255
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels2 > 1:
        best_label2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        mask = labels2 == best_label2
    return mask


def build_ghost_trail_image(
    frames: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    segment_fn: SegmentFn,
    gamma: float,
    min_alpha: int,
    scroll_px_per_step: float,
    occlusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Composite a causal ghost trail onto the current frame.

    ``frames`` must be chronological and include the current frame last. The
    output is still one RGB image, with all ghost content derived only from
    frames at or before the current decision step.
    """

    base = np.asarray(frames[-1], dtype=np.uint8)
    ghost_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[:-1]]
    if not ghost_frames:
        return base.copy()

    height, width = base.shape[:2]
    occlusion = occlusion_mask if occlusion_mask is not None else np.zeros((height, width), dtype=bool)
    current_step = int(steps[-1])
    alpha_acc = np.zeros((height, width), dtype=np.float32)
    color_acc = np.zeros((height, width, 3), dtype=np.float32)
    ghost_count = len(ghost_frames)

    for idx, frame in enumerate(ghost_frames):
        mask = segment_fn(frame)
        if mask is None or not mask.any():
            continue

        shift_x = float(scroll_px_per_step) * float(current_step - int(steps[idx]))
        shifted_mask = _shift_horizontal(mask.astype(np.uint8) * 255, shift_x) > 127
        shifted_frame = _shift_horizontal(frame, shift_x)
        shifted_mask = shifted_mask & (~occlusion)
        if not shifted_mask.any():
            continue

        x = float(idx + 1) / float(ghost_count)
        alpha_value = (int(min_alpha) + (255 - int(min_alpha)) * (x ** float(gamma))) / 255.0
        alpha_acc[shifted_mask] += alpha_value
        color_acc[shifted_mask] += alpha_value * shifted_frame[shifted_mask].astype(np.float32)

    has_ghost = alpha_acc > 1e-6
    if not has_ghost.any():
        return base.copy()

    final_alpha = np.clip(alpha_acc, 0.0, 1.0)
    blended_color = np.zeros((height, width, 3), dtype=np.float32)
    blended_color[has_ghost] = color_acc[has_ghost] / alpha_acc[has_ghost, None]

    result = base.astype(np.float32).copy()
    result[has_ghost] = (
        base[has_ghost].astype(np.float32) * (1.0 - final_alpha[has_ghost, None])
        + blended_color[has_ghost] * final_alpha[has_ghost, None]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def build_flappy_ghost_trail_episode_frames(
    frames: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    config: GhostTrailConfig,
) -> dict[int, np.ndarray]:
    """Build one ghost-trail image per Flappy decision step before row filtering."""
    history_frames = config.history_frames
    out: dict[int, np.ndarray] = {}
    for idx, step in enumerate(steps):
        start = max(0, idx - history_frames)
        out[int(step)] = build_flappy_ghost_trail_window(
            frames[start : idx + 1],
            steps[start : idx + 1],
            config=config,
        )
    return out


def build_flappy_ghost_trail_window(
    frames: Sequence[np.ndarray],
    steps: Sequence[int],
    *,
    config: GhostTrailConfig,
) -> np.ndarray:
    """Build one live/eval Flappy ghost-trail image from a causal frame window."""
    current = np.asarray(frames[-1], dtype=np.uint8)
    occlusion = _make_occlusion_mask(current, float(config.ground_fraction))
    return build_ghost_trail_image(
        frames,
        steps,
        segment_fn=lambda frame: _segment_flappy_bird(frame, ground_fraction=float(config.ground_fraction)),
        gamma=float(config.gamma),
        min_alpha=int(config.min_alpha),
        scroll_px_per_step=float(config.scroll_px_per_step),
        occlusion_mask=occlusion,
    )


# Native Demon Attack playfield and ship palette.
DEMON_ATTACK_SHIP_RGB = (184, 70, 162)
SCORE_ROW_END = 20
GROUND_ROW_START = 188
MAX_FOREGROUND_COMPONENT_AREA = 500


@dataclass
class _Component:
    frame_idx: int
    centroid: tuple[float, float]
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    mean_color: np.ndarray


@dataclass
class _Track:
    kind: str
    history: list[_Component]


def _is_small_component(comp: _Component) -> bool:
    x0, y0, x1, y1 = comp.bbox
    w = x1 - x0
    h = y1 - y0
    return comp.area <= 14 and w <= 6 and h <= 18


def _build_tracks(
    frame_components: list[list[_Component]],
    *,
    bullet_like: bool,
    max_gap: int,
) -> list[list[_Component]]:
    tracks: list[list[_Component]] = []
    active_last_idx: list[int] = []

    for frame_idx, comps in enumerate(frame_components):
        candidates = [c for c in comps if _is_small_component(c) == bullet_like]
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
                        prev_dy = history[-1].centroid[1] - history[-2].centroid[1]
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


def _classify_bullet_track(history: list[_Component]) -> str | None:
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


def _alpha_for_frame(frame_idx: int, n: int, *, gamma: float, min_alpha: int, max_alpha: int) -> float:
    x = (frame_idx + 1) / n
    return (min_alpha + (max_alpha - min_alpha) * (x ** gamma)) / 255.0


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1) > 0


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) <= 8:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=1) > 0
    edge = mask & ~eroded
    return edge if edge.any() else mask


def _render_tracks(
    frames: Sequence[np.ndarray],
    tracks: list[_Track],
) -> np.ndarray:
    n = len(frames)
    H, W = frames[0].shape[:2]
    current_idx = n - 1
    base = frames[-1].astype(np.float32)
    result = base.copy()
    rows = np.arange(H)[:, None]
    in_playfield = np.broadcast_to((rows >= SCORE_ROW_END) & (rows < GROUND_ROW_START), (H, W))
    current_foreground = np.any(frames[-1] != 0, axis=-1) & in_playfield

    params = {
        "ship": {"gamma": 2.3, "min_alpha": 0, "max_alpha": 72},
        "enemy": {"gamma": 2.3, "min_alpha": 0, "max_alpha": 78},
        "player_bullet": {"gamma": 1.8, "min_alpha": 4, "max_alpha": 120},
        "enemy_bullet": {"gamma": 1.8, "min_alpha": 4, "max_alpha": 120},
    }

    draw_ops = []
    for track in tracks:
        p = params[track.kind]

        history = track.history
        ghosts = history[:-1] if history[-1].frame_idx == current_idx else history
        for comp in ghosts:
            draw_ops.append((comp.frame_idx, track.kind, comp, p))

    for _, kind, comp, p in sorted(draw_ops, key=lambda item: item[0]):
        alpha = _alpha_for_frame(
            comp.frame_idx,
            n,
            gamma=p["gamma"],
            min_alpha=p["min_alpha"],
            max_alpha=p["max_alpha"],
        )
        mask = _dilate_mask(comp.mask) if "bullet" in kind else _contour_mask(comp.mask)
        mask = mask & ~current_foreground
        if not mask.any():
            continue
        if "bullet" in kind:
            color = comp.mean_color.astype(np.float32)
            result[mask] = result[mask] * (1.0 - alpha) + color * alpha
        else:
            source = frames[comp.frame_idx].astype(np.float32)
            result[mask] = result[mask] * (1.0 - alpha) + source[mask] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


@dataclass
class _FrameCache:
    """Per-frame data pre-computed once and shared across all sliding windows."""
    fg_comps: list[tuple]  # (centroid, mask, area, bbox, mean_color) — no frame_idx
    ship: tuple | None     # (mask, centroid, area, bbox) or None if absent


def _cache_frame(frame: np.ndarray) -> _FrameCache:
    """Extract one raw frame independently of its position in a sliding window."""
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


def _tracks_from_cache(cache_window: list[_FrameCache]) -> list[_Track]:
    """Associate cached entities using window-relative frame indices."""
    frame_components = [
        [
            _Component(frame_idx=i, centroid=c, mask=m, area=a, bbox=b, mean_color=mc)
            for c, m, a, b, mc in pc.fg_comps
        ]
        for i, pc in enumerate(cache_window)
    ]
    tracks: list[_Track] = []

    ship_history: list[_Component] = []
    for i, pc in enumerate(cache_window):
        if pc.ship is not None:
            s_mask, s_centroid, s_area, s_bbox = pc.ship
            ship_history.append(_Component(
                frame_idx=i,
                centroid=s_centroid,
                mask=s_mask,
                area=s_area,
                bbox=s_bbox,
                mean_color=np.array(DEMON_ATTACK_SHIP_RGB, dtype=np.float32),
            ))
    if ship_history:
        tracks.append(_Track("ship", ship_history))

    small_tracks = _build_tracks(frame_components, bullet_like=True, max_gap=1)
    for history in small_tracks:
        kind = _classify_bullet_track(history)
        if kind is not None:
            tracks.append(_Track(kind, history))

    large_tracks = _build_tracks(frame_components, bullet_like=False, max_gap=2)
    current_idx = len(cache_window) - 1
    for history in large_tracks:
        if len(history) >= 3 and current_idx - history[-1].frame_idx <= 1:
            tracks.append(_Track("enemy", history))

    return tracks


class DemonGhostTrailStream:
    """Cache raw-frame features for consecutive Demon Attack trail windows."""

    def __init__(self, history_frames: int) -> None:
        self.raw_frames: deque[np.ndarray] = deque(maxlen=history_frames)
        self.frame_cache: deque[_FrameCache] = deque(maxlen=history_frames)


    def append(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        self.raw_frames.append(frame)
        self.frame_cache.append(_cache_frame(frame))

    def render(self) -> np.ndarray:
        window = list(self.raw_frames)
        if len(window) == 1:
            return window[-1].copy()
        tracks = _tracks_from_cache(list(self.frame_cache))
        return _render_tracks(window, tracks)


def build_demon_attack_ghost_trail_window(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Render the continuous raw-frame window used by Demon Attack inference."""
    if len(frames) == 1:
        return np.asarray(frames[0], dtype=np.uint8).copy()
    stream = DemonGhostTrailStream(len(frames))
    for frame in frames:
        stream.append(frame)
    return stream.render()


def build_demon_attack_ghost_trail_episode_frames(
    frames: Sequence[np.ndarray], steps: Sequence[int], *, config: GhostTrailConfig,
) -> dict[int, np.ndarray]:
    """Render each episode frame with a bounded cache, including the current frame."""
    stream = DemonGhostTrailStream(config.history_frames)
    result = {}
    for step, frame in zip(steps, frames, strict=True):
        stream.append(frame)
        result[step] = stream.render()
    return result
