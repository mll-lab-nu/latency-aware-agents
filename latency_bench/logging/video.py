from __future__ import annotations

import json
import queue
import random
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_bench.core.types import EpisodeMetrics


@dataclass(frozen=True)
class VideoSelection:
    episode_id: int
    bin_index: int
    episode_return: float


def select_episode_return_stratified(
    metrics: list[EpisodeMetrics],
    *,
    num_bins: int,
    seed: int,
) -> list[VideoSelection]:
    """Pick representative episodes from rank bins sorted by episode_return.

    The score is intentionally fixed to EpisodeMetrics.episode_return. The bins
    are contiguous rank/distribution bins after sorting, not fixed numeric score
    ranges such as [0, 10), [10, 20), ...
    """
    ordered = sorted(metrics, key=lambda item: (item.episode_return, item.episode_id))
    bins: list[list[EpisodeMetrics]] = [[] for _ in range(num_bins)]
    total = len(ordered)
    for rank, metric in enumerate(ordered):
        bin_index = rank * num_bins // total
        bins[bin_index].append(metric)

    rng = random.Random(seed)
    selections = []
    for bin_index, bin_metrics in enumerate(bins):
        if not bin_metrics:
            continue
        metric = rng.choice(bin_metrics)
        selections.append(
            VideoSelection(
                episode_id=metric.episode_id,
                bin_index=bin_index,
                episode_return=metric.episode_return,
            )
        )
    return selections


class AsyncVideoWriter:
    def __init__(self, path: str | Path, *, fps: float):
        self.path = Path(path)
        self.fps = fps
        self.frame_count = 0
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frame: Any) -> None:
        self.frame_count += 1
        self._queue.put(frame.copy())

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        process = None
        command = None
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if process is None:
                    height, width = frame.shape[:2]
                    command = [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-vcodec",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(self.fps),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ]
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                process.stdin.write(frame.tobytes())
            if process is not None:
                process.stdin.close()
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command, stderr=process.stderr.read())
        except Exception as exc:
            self._error = exc
            if process is not None:
                process.kill()
                process.wait()


class EpisodeVideoRecorder:
    def __init__(self, output_dir: str | Path, *, fps: float):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: AsyncVideoWriter | None = None
        self._episode_id: int | None = None
        self._metadata: dict[str, Any] = {}

    def start_episode(self, *, episode_id: int, metadata: dict[str, Any]) -> None:
        self._episode_id = episode_id
        self._metadata = metadata.copy()
        name_parts = []
        if "bin_index" in self._metadata:
            name_parts.append(f"bin_{self._metadata['bin_index']:02d}")
        name_parts.append(f"episode_{self._episode_id:06d}")
        path = self.output_dir / ("_".join(name_parts) + ".mp4")
        self._metadata["video_path"] = path.name
        self._writer = AsyncVideoWriter(path, fps=self.fps)

    def record_frame(self, frame: Any) -> None:
        self._writer.write(frame)

    def close_episode(self, metadata: dict[str, Any]) -> None:
        self._metadata.update(metadata)
        self._writer.close()
        row = {
            "episode_id": self._episode_id,
            "fps": self.fps,
            "frame_count": self._writer.frame_count,
            **self._metadata,
        }
        with (self.output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._writer = None
        self._episode_id = None
        self._metadata = {}
