"""Causal ghost-trail image transforms for rollout dataset export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    cv2 = None


PIPE_SCROLL_PX_PER_STEP = 4.0

SegmentFn = Callable[[np.ndarray], np.ndarray | None]


@dataclass(frozen=True)
class GhostTrailConfig:
    """Configuration for causal ghost-trail rollout images."""

    image_transform: str = "raw_rgb"
    history_frames: int = 5
    gamma: float = 1.3
    min_alpha: int = 35
    ground_fraction: float = 0.22
    scroll_px_per_step: float = PIPE_SCROLL_PX_PER_STEP

    def normalized_transform(self) -> str:
        return str(self.image_transform).strip().lower()

    def metadata(self) -> dict:
        return asdict(self)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for flappy_ghost_trail image export")


def _shift_horizontal(image: np.ndarray, shift_x: float) -> np.ndarray:
    """Shift image left by ``shift_x`` pixels without wraparound."""
    if abs(float(shift_x)) < 1e-6:
        return image
    _require_cv2()
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
    _require_cv2()
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
    _require_cv2()
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
    if len(frames) != len(steps):
        raise ValueError("frames and steps must have the same length")
    if not frames:
        raise ValueError("frames must contain at least the current frame")

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
    if len(frames) != len(steps):
        raise ValueError("frames and steps must have the same length")
    history_frames = max(0, int(config.history_frames))
    out: dict[int, np.ndarray] = {}
    for idx, step in enumerate(steps):
        start = max(0, idx - history_frames)
        window_frames = frames[start : idx + 1]
        window_steps = steps[start : idx + 1]
        current = np.asarray(window_frames[-1], dtype=np.uint8)
        occlusion = _make_occlusion_mask(current, float(config.ground_fraction))
        out[int(step)] = build_ghost_trail_image(
            window_frames,
            window_steps,
            segment_fn=lambda frame: _segment_flappy_bird(frame, ground_fraction=float(config.ground_fraction)),
            gamma=float(config.gamma),
            min_alpha=int(config.min_alpha),
            scroll_px_per_step=float(config.scroll_px_per_step),
            occlusion_mask=occlusion,
        )
    return out


def build_episode_image_frames(
    *,
    env_name: str,
    rows: Sequence[dict],
    frames: Sequence[np.ndarray],
    config: GhostTrailConfig,
) -> dict[int, np.ndarray]:
    """Build the image frame mapping used by rollout row writing."""
    if len(rows) != len(frames):
        raise ValueError("rows and frames must have the same length")
    transform = config.normalized_transform()
    steps = [int(row["decision_step"]) for row in rows]
    if transform == "raw_rgb":
        return {int(step): frame for step, frame in zip(steps, frames)}
    if transform == "flappy_ghost_trail":
        if str(env_name) != "flappy":
            raise ValueError("image_transform=flappy_ghost_trail is only supported for env_name=flappy")
        return build_flappy_ghost_trail_episode_frames(frames, steps, config=config)
    raise ValueError(f"Unsupported image_transform={config.image_transform!r}")
