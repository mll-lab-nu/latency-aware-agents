"""Streaming video encoding and atomic numeric/video episode commits."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def encode_video(frames, path: Path, *, fps, executable):
    frame = frames[0]
    process = subprocess.Popen(
        [executable, "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size",
         f"{frame.shape[1]}x{frame.shape[0]}", "-framerate", str(fps),
         "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        try:
            process.stdin.close()
        finally:
            return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def commit_episode(target: Path, payload, video_path):
    """Expose a complete NPZ/MP4 pair with a single directory rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / "episode.npz", **payload)
        shutil.copyfile(video_path, staging / "episode.mp4")
        staging.replace(target)
    return target
