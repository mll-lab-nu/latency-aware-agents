"""Image extraction and writing helpers for rollout datasets."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to NumPy arrays.

    Args:
        value: Tensor, array, or array-like value.

    Returns:
        NumPy array view or copy.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def frames_from_env_raw_rgb_info(info: Mapping[str, Any]) -> list[np.ndarray]:
    """Extract raw RGB frame stack from environment info.

    Args:
        info: Environment info mapping containing raw RGB frames.

    Returns:
        Frames ordered oldest to newest.
    """
    raw_stack = _to_numpy(info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])
    return [frame for frame in raw_stack]


def frame_from_env_raw_rgb_info(info: Mapping[str, Any]) -> np.ndarray:
    """Extract the current raw RGB frame used for a decision row.

    Args:
        info: Environment info mapping containing raw RGB frames.

    Returns:
        Latest frame in the raw RGB stack.
    """
    return frames_from_env_raw_rgb_info(info)[-1]


def _save_image_frame(frame: np.ndarray, image_path: Path, export_image_size: tuple[int, int] | None) -> None:
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
        self.executor = ThreadPoolExecutor(max_workers=int(max_workers)) if int(max_workers) > 0 else None
        self.pending_futures: set[Future[None]] = set()
        self.row_buffer_size = int(row_buffer_size)
        self.export_image_size = export_image_size

    def write(self, frame: np.ndarray, image_path: Path) -> str:
        """Write one image frame and return its dataset path.

        Args:
            frame: RGB image array.
            image_path: Destination image path.

        Returns:
            Image path string stored in the rollout row.
        """
        if self.executor is None:
            _save_image_frame(frame, image_path, self.export_image_size)
            return str(image_path)
        future = self.executor.submit(
            _save_image_frame,
            np.array(frame, copy=True),
            image_path,
            self.export_image_size,
        )
        self.pending_futures.add(future)
        if len(self.pending_futures) >= self.row_buffer_size:
            done, pending = wait(self.pending_futures, return_when=FIRST_COMPLETED)
            for item in done:
                item.result()
            self.pending_futures = set(pending)
        return str(image_path)

    def flush(self) -> None:
        """Wait for all pending image writes to finish."""
        for future in self.pending_futures:
            future.result()
        self.pending_futures = set()

    def close(self) -> None:
        """Flush pending writes and close the thread pool."""
        try:
            self.flush()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
