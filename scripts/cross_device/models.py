"""Shared task models for the cross-device protocol.

The models intentionally stay small: they describe planned work and runtime
selection, while execution policy remains in the planner and runner modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved runtime environment state for filtering and safety checks."""
    requested: str
    detected_raw: str | None
    matched_alias: str | None
    matched_conda_env: str | None
    selected_alias: str | None
    allow_all: bool
    allow_mismatch: bool


@dataclass
class ProtocolTask:
    """Concrete unit of protocol work after row/device/seed expansion."""
    stage: str
    experiment_id: str
    experiment_ids: list[str]
    panel: str
    row: dict[str, Any]
    target_device: dict[str, Any]
    replay_device: dict[str, Any] | None
    seed: int
    seed_set: list[int]
    n_episodes: int
    n_samples: int | None
    checkpoint_kind: str
    output_dir: Path
    generated_config_path: Path
    target_profile_path: Path | None
    profile_output_dir: Path | None
    runtime_env_alias: str | None
    runtime_conda_env: str | None
    smoke: bool = False
    smoke_max_steps: int | None = None


@dataclass
class PlannedTask:
    """A protocol task plus the planner decision about whether it can run."""
    task: ProtocolTask
    runnable: bool
    skip_reason: str | None = None
