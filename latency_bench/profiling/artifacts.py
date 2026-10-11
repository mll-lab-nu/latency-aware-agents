"""Define local coordinates for profiles and measurement artifacts."""

from __future__ import annotations

from pathlib import Path


def profile_ref(model_id: str, gpu_class: str, workload_id: str, instance_id: str) -> Path:
    return Path(model_id) / gpu_class / workload_id / instance_id
