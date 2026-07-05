"""Protocol spec loading plus hardware/runtime detection for cross-device runs.

The spec defines the protocol rows once at top level; experiments only carry
their panel label and, for cross-device prediction, the replay pairs. This
module also maps the current machine and Python environment onto spec entries.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def load_protocol_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Protocol spec must be a mapping: {spec_path}")
    return spec


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_")


def profile_dir(
    profiles_dir: Path,
    *,
    model_type: str,
    env_name: str,
    device_id: str,
    instance_id: str,
) -> Path:
    return Path(profiles_dir) / slugify(model_type) / slugify(env_name) / str(device_id) / str(instance_id)


def profile_path(
    profiles_dir: Path,
    *,
    model_type: str,
    env_name: str,
    device_id: str,
    instance_id: str,
) -> Path:
    return profile_dir(
        profiles_dir,
        model_type=model_type,
        env_name=env_name,
        device_id=device_id,
        instance_id=instance_id,
    ) / "profile.json"


def detect_hardware(spec: dict[str, Any], requested: str = "auto") -> dict[str, Any]:
    """Resolve the hardware entry for this invocation.

    Args:
        spec: Loaded protocol spec containing device aliases.
        requested: Explicit device id/alias, or `auto` to inspect CUDA/nvidia-smi.

    Returns:
        A dict with the raw detected/requested string and matched device entry.
    """
    devices = _devices(spec)
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


def detect_runtime_env_raw() -> str | None:
    """Return the current Python environment name, or None when undetectable."""
    override = os.environ.get("CROSS_DEVICE_RUNTIME_ENV")
    if override:
        return override
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return conda_env
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).name
    executable = Path(sys.executable)
    parts = executable.parts
    for marker in ("envs", "env"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def runtime_env_mismatch(spec: dict[str, Any]) -> str | None:
    """Return a mismatch message, or None when the detected env satisfies the spec.

    The spec declares a single `runtime_env` with `name` and `conda_env`; the
    detected environment may match either spelling.
    """
    required = spec["runtime_env"]
    detected = detect_runtime_env_raw()
    accepted = {_normalize_name(required["name"]), _normalize_name(required["conda_env"])}
    if detected and _normalize_name(detected) in accepted:
        return None
    return (
        f"runtime env mismatch: requires {required['name']} ({required['conda_env']}), "
        f"detected {detected or 'unknown'}. Activate the right conda env or set CROSS_DEVICE_RUNTIME_ENV."
    )


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
            names = [device.get("id"), device.get("display_name"), *device.get("aliases", [])]
            if any(_normalize_name(item) == normalized for item in names if item is not None):
                return device
    for normalized in candidates:
        if not normalized:
            continue
        for device in devices:
            names = [device.get("id"), device.get("display_name"), *device.get("aliases", [])]
            aliases = [_normalize_name(item) for item in names if item is not None]
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
        aliases.extend([str(device.get("id")), str(device.get("display_name"))])
        aliases.extend(str(alias) for alias in device.get("aliases", []))
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


def _devices(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [device for device in spec.get("devices", []) if isinstance(device, dict)]
