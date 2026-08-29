"""Small bridge from canonical rollout exports to Isaac-GR00T data files."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from latency_bench.data.action_layouts import G1_SONIC_GR00T_ACTION_SPEC
from latency_bench.data.exporters.starvla_lerobot import (
    convert_canonical_rollout_to_lerobot,
)
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
    G1_GR00T_ACTION_HORIZON,
)

OFFICIAL_GR00T_EMBODIMENT_TAG = "NEW_EMBODIMENT"


def _official_gr00t_modality_json(state_dim: int) -> dict[str, Any]:
    """Return the LeRobot modality metadata consumed by Isaac-GR00T."""
    return {
        "state": {
            "state": {
                "start": 0,
                "end": state_dim,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "action": {
                "start": 0,
                "end": G1_GR00T_ACTION_DIM,
                "dtype": "float32",
                "absolute": True,
                "original_key": "action",
            }
        },
        "video": {"ego_view": {"original_key": "observation.image"}},
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def _write_official_gr00t_modality(dataset_dir: Path | str, state_dim: int) -> Path:
    """Write the official modality metadata into one LeRobot dataset split."""
    path = Path(dataset_dir) / "meta" / "modality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_official_gr00t_modality_json(state_dim), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_official_gr00t_videos(dataset_dir: Path) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = info["fps"]
    feature = info["features"]["observation.image"]
    height, width, channels = feature["shape"]
    for episode_index in range(info["total_episodes"]):
        chunk = episode_index // info["chunks_size"]
        episode_path = dataset_dir / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        video_path = dataset_dir / info["video_path"].format(
            episode_chunk=chunk,
            video_key="observation.image",
            episode_index=episode_index,
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        images = pq.read_table(episode_path, columns=["observation.image"])[
            "observation.image"
        ].to_pylist()
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
        for image in images:
            with Image.open(io.BytesIO(image["bytes"])) as frame:
                process.stdin.write(frame.convert("RGB").tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)

    info["total_videos"] = info["total_episodes"]
    feature["dtype"] = "video"
    feature["info"] = {
        "video.height": height,
        "video.width": width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": channels,
        "has_audio": False,
    }
    del feature["video_info"]
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def convert_canonical_rollout_to_official_gr00t(
    raw_export_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Convert a canonical G1 rollout and write both train/val metadata."""
    raw_metadata = json.loads(
        Path(raw_export_dir).joinpath("metadata.json").read_text(encoding="utf-8")
    )
    if (
        raw_metadata["action_layout"] != G1_SONIC_GR00T_ACTION_SPEC.layout
        or tuple(raw_metadata["action_labels"])
        != G1_SONIC_GR00T_ACTION_SPEC.labels
    ):
        raise ValueError("official GR00T requires the g1_sonic_gr00t_v2 action layout")
    manifest = convert_canonical_rollout_to_lerobot(
        raw_export_dir,
        output_dir,
        force=force,
        action_carrier="native",
        normalize_state=False,
        write_starvla_handoff=False,
    )
    if manifest["action_dim"] != G1_GR00T_ACTION_DIM:
        raise ValueError(
            f"official GR00T requires a {G1_GR00T_ACTION_DIM}D action, "
            f"got {manifest['action_dim']}"
        )
    output_dir = Path(output_dir)
    validation_dir = output_dir.with_name(f"{output_dir.name}__val")
    _write_official_gr00t_videos(output_dir)
    _write_official_gr00t_videos(validation_dir)
    _write_official_gr00t_modality(output_dir, manifest["state_dim"])
    _write_official_gr00t_modality(validation_dir, manifest["state_dim"])
    manifest = {
        **manifest,
        "official_embodiment_tag": OFFICIAL_GR00T_EMBODIMENT_TAG,
        "official_action_horizon": G1_GR00T_ACTION_HORIZON,
        "official_modality_config": "latency_bench/policy/official_gr00t_modality.py",
    }
    output_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    validation_manifest_path = validation_dir / "manifest.json"
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    validation_manifest.update(
        {
            "official_embodiment_tag": OFFICIAL_GR00T_EMBODIMENT_TAG,
            "official_action_horizon": G1_GR00T_ACTION_HORIZON,
            "official_modality_config": "latency_bench/policy/official_gr00t_modality.py",
        }
    )
    validation_manifest_path.write_text(
        json.dumps(validation_manifest, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "OFFICIAL_GR00T_EMBODIMENT_TAG",
    "convert_canonical_rollout_to_official_gr00t",
]
