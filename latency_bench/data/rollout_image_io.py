"""Image extraction and writing helpers for rollout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.data.background_writer import BackgroundWriter

from latency_bench.utils.array import to_numpy
from latency_bench.data.ghost_trail import (
    GhostTrailConfig, build_flappy_ghost_trail_episode_frames,
    build_demon_attack_ghost_trail_episode_frames,
)


def frames_from_env_raw_rgb_info(info: Mapping[str, Any]) -> list[np.ndarray]:
    """Extract raw RGB frame stack from environment info.

    Args:
        info: Environment info mapping containing raw RGB frames.

    Returns:
        Frames ordered oldest to newest.
    """
    raw_stack = to_numpy(info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])
    return [frame for frame in raw_stack]


def frame_from_env_raw_rgb_info(info: Mapping[str, Any]) -> np.ndarray:
    """Extract the current raw RGB frame used for a decision row.

    Args:
        info: Environment info mapping containing raw RGB frames.

    Returns:
        Latest frame in the raw RGB stack.
    """
    return frames_from_env_raw_rgb_info(info)[-1]


def save_image_frame(frame: np.ndarray, image_path: Path, export_image_size: tuple[int, int] | None) -> None:
    """Save one raw RGB frame to disk.

    Args:
        frame: RGB image array.
        image_path: Destination image path.
        export_image_size: Optional ``(width, height)`` resize target.
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame)
    if export_image_size is not None:
        image = image.resize(export_image_size, Image.Resampling.BILINEAR)
    image.save(image_path)


class RolloutImageWriter:
    """Write rollout images synchronously or with a bounded thread pool.

    Args:
        max_workers: Number of image writer threads.
        row_buffer_size: Maximum pending writes before waiting.
        export_image_size: Optional ``(width, height)`` resize target.
    """
    def __init__(self, *, max_workers: int, row_buffer_size: int, export_image_size: tuple[int, int] | None):
        """Create an image writer for one rollout collection run.

        Args:
            max_workers: Number of image writer threads.
            row_buffer_size: Maximum pending writes before waiting.
            export_image_size: Optional ``(width, height)`` resize target.
        """
        self.writer = BackgroundWriter(max_workers=max_workers, max_pending=row_buffer_size) if max_workers != 0 else None
        self.export_image_size = export_image_size

    def write(self, frame: np.ndarray, image_path: Path) -> str:
        """Write one image frame and return its dataset path.

        Args:
            frame: RGB image array.
            image_path: Destination image path.

        Returns:
            Image path string stored in the rollout row.
        """
        if self.writer is None:
            save_image_frame(frame, image_path, self.export_image_size)
            return str(image_path)
        self.writer.submit(
            save_image_frame,
            np.array(frame, copy=True),
            image_path,
            self.export_image_size,
        )
        return str(image_path)

    def flush(self) -> None:
        """Wait for all pending image writes to finish."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Flush pending writes and close the thread pool."""
        if self.writer is not None:
            self.writer.close()


def build_episode_image_frames(
    *,
    env_name: str,
    rows: Sequence[dict],
    frames: Sequence[np.ndarray],
    config: GhostTrailConfig,
) -> dict[int, np.ndarray]:
    """Build the image frame mapping used by rollout row writing."""
    transform = config.image_transform
    steps = [int(row["decision_step"]) for row in rows]
    if transform == "raw_rgb":
        return {int(step): frame for step, frame in zip(steps, frames)}
    if transform == "flappy_ghost_trail":
        if str(env_name) != "flappy":
            raise ValueError("image_transform=flappy_ghost_trail is only supported for env_name=flappy")
        return build_flappy_ghost_trail_episode_frames(frames, steps, config=config)
    if transform == "demon_attack_ghost_trail":
        if str(env_name) != "demon_attack":
            raise ValueError("image_transform=demon_attack_ghost_trail is only supported for env_name=demon_attack")
        return build_demon_attack_ghost_trail_episode_frames(frames, steps, config=config)
    raise ValueError(f"Unsupported image_transform={config.image_transform!r}")
