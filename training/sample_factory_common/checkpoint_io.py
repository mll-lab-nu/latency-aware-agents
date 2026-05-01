from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def trusted_torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    load_path = str(Path(path).expanduser())
    try:
        return torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(load_path, map_location=device)


def load_sample_factory_checkpoint_from_paths(checkpoints: list[str] | list[Path], device: torch.device) -> dict[str, Any]:
    if not checkpoints:
        raise RuntimeError("No checkpoints found to load")
    last_error: Exception | None = None
    for checkpoint_path in reversed([str(Path(item).expanduser()) for item in checkpoints]):
        try:
            return trusted_torch_load(checkpoint_path, device)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError("Could not load checkpoint") from last_error
    raise RuntimeError("Could not load checkpoint")
