from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def experiment_dir(train_dir: str | os.PathLike[str], experiment_name: str) -> Path:
    name = str(experiment_name).strip()
    if not name:
        raise ValueError("experiment.name must be a non-empty string")
    return Path(train_dir).expanduser() / name


def checkpoint_dir(
    train_dir: str | os.PathLike[str],
    experiment_name: str,
    *,
    policy_id: int = 0,
) -> Path:
    return experiment_dir(train_dir, experiment_name) / f"checkpoint_p{int(policy_id)}"


def format_checkpoint_name(*, train_step: int, env_steps: int) -> str:
    return f"checkpoint_{int(train_step):09d}_{int(env_steps):012d}.pth"


def find_checkpoint(
    train_dir: str | os.PathLike[str],
    experiment_name: str,
    *,
    kind: str = "latest",
    policy_id: int = 0,
) -> Path:
    normalized_kind = str(kind).strip().lower()
    if normalized_kind == "latest":
        pattern = "checkpoint_*.pth"
    elif normalized_kind == "best":
        pattern = "best_*.pth"
    else:
        raise ValueError("checkpoint kind must be 'latest' or 'best'")

    root = checkpoint_dir(train_dir, experiment_name, policy_id=policy_id)
    checkpoints = sorted(root.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No {normalized_kind} checkpoint matching {pattern!r} under {root}"
        )
    return checkpoints[-1]


def read_checkpoint_backend_type(path: str | os.PathLike[str]) -> str | None:
    checkpoint = _torch_load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    backend_type: Any = checkpoint.get("backend_type")
    if backend_type is None:
        return None
    return str(backend_type)


def _torch_load(path: str | os.PathLike[str], *, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
