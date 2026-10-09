from __future__ import annotations

import torch
from typing import Any, Sequence

import numpy as np


def batch_indices(indices: Sequence[int] | np.ndarray | torch.Tensor | None, *, batch_size: int) -> list[int]:
    if indices is None:
        return list(range(batch_size))
    if isinstance(indices, torch.Tensor):
        return [int(item) for item in indices.reshape(-1).detach().cpu().tolist()]
    return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]


def seed_values(seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
    if seeds is None:
        return [None] * count
    if isinstance(seeds, (int, np.integer)):
        return [int(seeds)] * count
    return [None if item is None else int(item) for item in list(seeds)]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def maybe_set_cli_override(cfg: Any, key: str, value: Any) -> None:
    setattr(cfg, key, value)
    if not hasattr(cfg, "cli_args") or cfg.cli_args is None:
        cfg.cli_args = {}
    cfg.cli_args[key] = value


def env_instance_seed(
    *,
    base_seed: int,
    env_id: int,
) -> int:
    # Preserve the historical one-env-per-worker stream while making it independent of worker layout.
    return base_seed + 1046 * env_id
