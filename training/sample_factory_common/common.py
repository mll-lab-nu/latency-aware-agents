from __future__ import annotations

import random
from typing import Any

import numpy as np


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def list_to_csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


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
    worker_index: int = 0,
    vector_index: int = 0,
    env_id: int = 0,
) -> int:
    return int(base_seed) + 1009 * int(worker_index) + 9173 * int(vector_index) + 37 * int(env_id)


def seed_everything(seed: int) -> int:
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    try:
        import torch

        torch.manual_seed(normalized_seed)
    except ImportError:
        pass
    return normalized_seed
