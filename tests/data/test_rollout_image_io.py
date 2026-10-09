from __future__ import annotations
from pathlib import Path

import numpy as np


def test_write_filtered_episode_images_preserves_empty_filtered_episode():
    from latency_bench.data.rollout_image_io import RolloutImageWriter
    from latency_bench.data.rollout_episode import write_filtered_episode_images

    image_writer = RolloutImageWriter(max_workers=0, row_buffer_size=1, export_image_size=None)

    rows = write_filtered_episode_images(image_writer, [], {}, context_window=5)

    assert rows == []


def test_write_filtered_episode_images_context_zero_keeps_only_current_image(tmp_path):
    from latency_bench.data.rollout_image_io import RolloutImageWriter
    from latency_bench.data.rollout_episode import write_filtered_episode_images

    image_writer = RolloutImageWriter(max_workers=0, row_buffer_size=1, export_image_size=None)
    rows = [{"image": str(tmp_path / "frame.png"), "decision_step": 0}]
    frames = {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    written_rows = write_filtered_episode_images(
        image_writer,
        rows,
        frames,
        context_window=0,
    )

    assert "context_images" not in written_rows[0]
    assert Path(written_rows[0]["image"]).is_file()


def test_write_filtered_episode_images_resets_context_after_decision_step_gap(tmp_path):
    from PIL import Image
    from latency_bench.data.rollout_image_io import RolloutImageWriter
    from latency_bench.data.rollout_episode import write_filtered_episode_images

    image_writer = RolloutImageWriter(max_workers=0, row_buffer_size=1, export_image_size=None)
    rows = [
        {
            "image": str(tmp_path / "images" / "train" / "episode_000000" / f"decision_step_{step:06d}.png"),
            "decision_step": step,
        }
        for step in [0, 1, 4, 5]
    ]
    frames_by_decision_step = {
        step: np.full((2, 2, 3), step * 10, dtype=np.uint8)
        for step in [0, 1, 4, 5]
    }

    written_rows = write_filtered_episode_images(
        image_writer,
        rows,
        frames_by_decision_step,
        context_window=4,
    )

    context_values = [
        [
            np.asarray(Image.open(path))[0, 0, 0].item()
            for path in row["context_images"]
        ]
        for row in written_rows
    ]
    current_values = [
        np.asarray(Image.open(row["image"]))[0, 0, 0].item()
        for row in written_rows
    ]

    assert context_values == [
        [0, 0, 0],
        [0, 0, 0],
        [40, 40, 40],
        [40, 40, 40],
    ]
    assert current_values == [0, 10, 40, 50]
