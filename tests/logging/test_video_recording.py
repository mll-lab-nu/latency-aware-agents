from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from latency_bench.core.types import EpisodeMetrics


def test_stratified_video_selection_uses_episode_return_rank_bins():
    from latency_bench.logging.video import select_episode_return_stratified

    metrics = [
        EpisodeMetrics(episode_id=episode_id, episode_return=float(episode_id), survival_steps=1)
        for episode_id in range(10)
    ]

    selections = select_episode_return_stratified(metrics, num_bins=10, seed=0)

    assert [selection.episode_id for selection in selections] == list(range(10))
    assert [selection.bin_index for selection in selections] == list(range(10))
    assert [selection.episode_return for selection in selections] == [float(value) for value in range(10)]


def test_stratified_video_selection_skips_empty_rank_bins():
    from latency_bench.logging.video import select_episode_return_stratified

    metrics = [
        EpisodeMetrics(episode_id=4, episode_return=40.0, survival_steps=1),
        EpisodeMetrics(episode_id=2, episode_return=20.0, survival_steps=1),
        EpisodeMetrics(episode_id=8, episode_return=80.0, survival_steps=1),
    ]

    selections = select_episode_return_stratified(metrics, num_bins=10, seed=0)

    assert [selection.episode_id for selection in selections] == [2, 4, 8]
    assert [selection.bin_index for selection in selections] == [0, 3, 6]
    assert [selection.episode_return for selection in selections] == [20.0, 40.0, 80.0]


def test_episode_video_recorder_writes_mp4_and_manifest(tmp_path: Path):
    from latency_bench.logging.video import EpisodeVideoRecorder

    recorder = EpisodeVideoRecorder(output_dir=tmp_path, fps=30.0)
    recorder.start_episode(
        episode_id=7,
        metadata={"bin_index": 2, "episode_return": 12.5},
    )
    for value in range(3):
        recorder.record_frame(np.full((12, 16, 3), value * 20, dtype=np.uint8))
    recorder.close_episode({})

    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(manifest) == 1
    assert manifest[0]["episode_id"] == 7
    assert manifest[0]["bin_index"] == 2
    assert manifest[0]["episode_return"] == 12.5
    assert manifest[0]["fps"] == 30.0
    assert manifest[0]["frame_count"] == 3
    video_path = tmp_path / manifest[0]["video_path"]
    assert video_path.is_file()
    probe_output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,nb_frames",
            "-of",
            "compact=p=0:nk=1",
            str(video_path),
        ],
        text=True,
    ).strip()
    assert probe_output == "h264|yuv420p|3"


def test_episode_video_recorder_requires_explicit_episode_metadata(tmp_path: Path):
    from latency_bench.logging.video import EpisodeVideoRecorder

    recorder = EpisodeVideoRecorder(output_dir=tmp_path, fps=30.0)

    try:
        recorder.start_episode(episode_id=1)
    except TypeError:
        pass
    else:
        raise AssertionError("start_episode must require explicit metadata")
