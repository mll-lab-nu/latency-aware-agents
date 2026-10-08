from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModalityWindowIndices:
    observation_indices: list[int]
    language_indices: list[int]
    state_indices: list[int]
    action_indices: list[int]


def _cfg_get(data_cfg: Mapping[str, Any] | Any | None, key: str) -> Any:
    if data_cfg is None:
        return None
    getter = getattr(data_cfg, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(data_cfg, key, None)


def _int_list(value: Sequence[int] | None, label: str) -> list[int]:
    if value is None:
        raise ValueError(f"Missing {label}")
    out = [int(item) for item in value]
    if not out:
        raise ValueError(f"{label} must not be empty")
    return out


def resolve_modality_indices(
    default_observation_indices: Sequence[int],
    default_state_indices: Sequence[int],
    default_action_indices: Sequence[int],
    data_cfg: Mapping[str, Any] | Any | None,
) -> ModalityWindowIndices:
    observation_override = _cfg_get(data_cfg, "observation_indices")
    language_override = _cfg_get(data_cfg, "language_indices")
    state_override = _cfg_get(data_cfg, "state_indices")
    action_override = _cfg_get(data_cfg, "action_indices")

    observation_indices = _int_list(
        observation_override if observation_override is not None else default_observation_indices,
        "observation_indices",
    )
    language_indices = _int_list(
        language_override if language_override is not None else ([0] if observation_override is not None else observation_indices),
        "language_indices",
    )
    state_indices = _int_list(
        state_override if state_override is not None else ([0] if observation_override is not None else default_state_indices),
        "state_indices",
    )
    action_indices = _int_list(
        action_override if action_override is not None else default_action_indices,
        "action_indices",
    )

    return ModalityWindowIndices(
        observation_indices=observation_indices,
        language_indices=language_indices,
        state_indices=state_indices,
        action_indices=action_indices,
    )


def pack_image_sequence(
    frames: np.ndarray,
    pack_image_sequence: bool,
    image_sequence_length: int,
    resize_size: tuple[int, int],
) -> list[Image.Image]:
    if frames.ndim != 4:
        raise ValueError(f"Expected image frames shaped [T,H,W,C], got shape={frames.shape}")
    if image_sequence_length <= 0:
        raise ValueError(f"image_sequence_length must be positive, got {image_sequence_length}")

    selected_frames = frames if pack_image_sequence else frames[:1]
    expected_length = image_sequence_length if pack_image_sequence else 1
    if selected_frames.shape[0] != expected_length:
        raise ValueError(
            f"Expected {expected_length} image frame(s), got {selected_frames.shape[0]} "
            f"with pack_image_sequence={pack_image_sequence}"
        )

    return [Image.fromarray(frame).resize(resize_size) for frame in selected_frames]


def decode_image_entry(entry: Any, dataset_path: Path | None) -> np.ndarray:
    if isinstance(entry, np.ndarray):
        if entry.ndim == 3:
            return entry
        raise ValueError(f"Expected image array shaped [H,W,C], got shape={entry.shape}")
    if isinstance(entry, Image.Image):
        return np.asarray(entry.convert("RGB"))
    if isinstance(entry, (bytes, bytearray, memoryview)):
        return np.asarray(Image.open(io.BytesIO(bytes(entry))).convert("RGB"))
    if isinstance(entry, dict):
        image_bytes = entry.get("bytes")
        image_path = entry.get("path")
        if image_bytes is not None:
            return np.asarray(Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB"))
        if image_path is not None:
            path = Path(str(image_path))
            if not path.is_absolute():
                if dataset_path is None:
                    raise ValueError(f"Relative image path requires dataset_path: {image_path}")
                path = dataset_path / path
            return np.asarray(Image.open(path).convert("RGB"))
        raise ValueError("Image entry dict must contain `bytes` or `path`")
    raise TypeError(f"Unsupported image entry type: {type(entry)}")


def decode_prepacked_image_sequence(
    entry: Any,
    image_sequence_length: int,
    dataset_path: Path | None,
) -> np.ndarray:
    if image_sequence_length <= 0:
        raise ValueError(f"image_sequence_length must be positive, got {image_sequence_length}")
    if isinstance(entry, np.ndarray):
        if entry.ndim == 4:
            if entry.shape[0] != image_sequence_length:
                raise ValueError(f"Expected {image_sequence_length} image frames, got {entry.shape[0]}")
            return entry
        values = entry.tolist()
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray, memoryview)):
        values = list(entry)
    else:
        raise TypeError(f"Unsupported image sequence entry type: {type(entry)}")

    if len(values) != image_sequence_length:
        raise ValueError(f"Expected {image_sequence_length} image frames, got {len(values)}")
    return np.stack([decode_image_entry(item, dataset_path) for item in values])


def decode_context_image_sequence(
    context_entry: Any,
    current_entry: Any,
    image_sequence_length: int,
    dataset_path: Path | None,
) -> np.ndarray:
    if image_sequence_length < 2:
        raise ValueError(f"image_sequence_length must be at least 2, got {image_sequence_length}")
    if isinstance(context_entry, np.ndarray):
        if context_entry.ndim == 4:
            context_values = list(context_entry)
        else:
            context_values = context_entry.tolist()
    elif isinstance(context_entry, Sequence) and not isinstance(context_entry, (str, bytes, bytearray, memoryview)):
        context_values = list(context_entry)
    else:
        raise TypeError(f"Unsupported context image sequence entry type: {type(context_entry)}")

    expected_context_images = image_sequence_length - 1
    if len(context_values) != expected_context_images:
        raise ValueError(f"Expected {expected_context_images} context image frames, got {len(context_values)}")
    return np.stack([decode_image_entry(item, dataset_path) for item in [*context_values, current_entry]])
