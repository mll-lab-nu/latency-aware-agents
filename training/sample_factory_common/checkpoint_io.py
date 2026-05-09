from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED = False


def trusted_torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    load_path = str(Path(path).expanduser())
    try:
        return torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(load_path, map_location=device)


def patch_sample_factory_checkpoint_loader() -> None:
    global _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED
    if _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED:
        return

    import sample_factory.algo.learning.learner as sf_learner

    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            sf_learner.log.warning("No checkpoints found")
            return None

        latest_checkpoint = checkpoints[-1]
        num_attempts = 3
        for attempt in range(num_attempts):
            try:
                sf_learner.log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                return trusted_torch_load(latest_checkpoint, device)
            except Exception:
                sf_learner.log.exception(f"Could not load from checkpoint, attempt {attempt}")
        return None

    sf_learner.Learner.load_checkpoint = staticmethod(load_checkpoint)
    _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED = True


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
