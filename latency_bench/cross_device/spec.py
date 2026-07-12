"""Cross-device configuration loading and measurement hardware detection."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from latency_bench.core.config import load_config


def load_workload(path: str | Path) -> dict[str, Any]:
    return load_config(path)


def detect_hardware(spec: dict[str, Any], requested: str = "auto") -> dict[str, Any]:
    """Resolve the hardware entry for this invocation.

    Args:
        spec: Loaded protocol spec containing device aliases.
        requested: Explicit device id/alias, or `auto` to inspect CUDA/nvidia-smi.

    Returns:
        A dict with the raw detected/requested string and matched device entry.
    """
    devices = spec["devices"]
    if requested != "auto":
        device = normalize_hardware_name(requested, devices)
        if device is None:
            _raise_unknown_hardware(requested, devices)
        return {"raw": requested, "device": device}

    candidates: list[str] = []
    if torch.cuda.is_available():
        candidates.append(str(torch.cuda.get_device_name(0)))
    smi_name = _nvidia_smi_device_name()
    if smi_name:
        candidates.append(smi_name)
    env_name = os.environ.get("CROSS_DEVICE_HARDWARE")
    if env_name:
        candidates.append(env_name)
    for candidate in candidates:
        device = normalize_hardware_name(candidate, devices)
        if device is not None:
            return {"raw": candidate, "device": device}
    _raise_unknown_hardware("; ".join(candidates) if candidates else "<none>", devices)
def normalize_hardware_name(name: str, devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match raw GPU text to a device entry from the protocol spec.

    Args:
        name: Raw text from CLI, `nvidia-smi`, or torch.
        devices: Device entries from the protocol spec.

    Returns:
        The matched device entry, or None when no alias overlaps.
    """
    candidates = _hardware_name_candidates(name)
    for normalized in candidates:
        for device in devices:
            names = [device["id"], device["display_name"], *device["aliases"]]
            if any(_normalize_name(item) == normalized for item in names):
                return device
    for normalized in candidates:
        if not normalized:
            continue
        for device in devices:
            names = [device["id"], device["display_name"], *device["aliases"]]
            aliases = [_normalize_name(item) for item in names]
            if any(_hardware_names_overlap(normalized, alias) for alias in aliases):
                return device
    return None


def _hardware_name_candidates(name: str) -> list[str]:
    raw = str(name)
    parts = [raw, *raw.replace(";", "\n").splitlines()]
    candidates: list[str] = []
    for part in parts:
        normalized = _normalize_name(part)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _hardware_names_overlap(detected: str, known: str) -> bool:
    if len(detected) < 4 or len(known) < 4:
        return False
    return detected in known or known in detected


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _raise_unknown_hardware(detected: str, devices: list[dict[str, Any]]) -> None:
    aliases = []
    for device in devices:
        aliases.extend([device["id"], device["display_name"]])
        aliases.extend(device["aliases"])
    raise SystemExit(
        "Could not match hardware. "
        f"detected={detected!r}. Known aliases: {', '.join(sorted(set(aliases)))}"
    )


def _nvidia_smi_device_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None
