"""Runtime and hardware detection for cross-device experiments.

This module maps the current machine and Python environment to protocol spec
aliases. It does not build experiment plans or execute runs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from scripts.cross_device.models import RuntimeSelection


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


def detect_runtime_env(spec: dict[str, Any], requested: str = "auto", allow_mismatch: bool = False) -> RuntimeSelection:
    """Resolve the current Python environment to a protocol runtime alias.

    Args:
        spec: Loaded protocol spec containing `runtime_envs`.
        requested: Runtime selector from CLI: `auto`, `all`, alias, or conda env.
        allow_mismatch: Whether execution may proceed despite env mismatch.

    Returns:
        Runtime selection metadata used for filtering and skip reasons.
    """
    detected_raw = _detect_runtime_env_raw()
    matched_alias = _resolve_runtime_alias(spec, detected_raw) if detected_raw else None
    matched_conda_env = _runtime_conda_env(spec, matched_alias) if matched_alias else None
    if requested == "auto":
        selected_alias = matched_alias
        allow_all = False
    elif requested == "all":
        selected_alias = None
        allow_all = True
    else:
        selected_alias = _resolve_runtime_alias(spec, requested)
        if selected_alias is None:
            _raise_unknown_runtime_env(spec, requested)
        allow_all = False
    return RuntimeSelection(
        requested=requested,
        detected_raw=detected_raw,
        matched_alias=matched_alias,
        matched_conda_env=matched_conda_env,
        selected_alias=selected_alias,
        allow_all=allow_all,
        allow_mismatch=allow_mismatch,
    )


def _detect_runtime_env_raw() -> str | None:
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
    for part in reversed(parts):
        if part == "latency":
            return part
    return None


def _runtime_envs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    envs = spec.get("runtime_envs", {})
    return envs if isinstance(envs, dict) else {}


def _resolve_runtime_alias(spec: dict[str, Any], value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_name(value)
    for alias, env in _runtime_envs(spec).items():
        names = [alias, env.get("conda_env")]
        if any(_normalize_name(name) == normalized for name in names if name):
            return str(alias)
    return None


def _runtime_conda_env(spec: dict[str, Any], alias: str | None) -> str | None:
    if alias is None:
        return None
    env = _runtime_envs(spec).get(alias)
    if not isinstance(env, dict):
        return None
    value = env.get("conda_env")
    return None if value is None else str(value)


def _raise_unknown_runtime_env(spec: dict[str, Any], value: str) -> None:
    known = []
    for alias, env in _runtime_envs(spec).items():
        known.append(str(alias))
        if env.get("conda_env"):
            known.append(str(env["conda_env"]))
    raise SystemExit(f"Unknown --runtime-env {value!r}. Known runtime envs: {', '.join(sorted(known))}")


def _validate_runtime_safety(args: argparse.Namespace, runtime: RuntimeSelection) -> None:
    if runtime.requested == "all" and not args.dry_run and not runtime.allow_mismatch:
        raise SystemExit("--runtime-env all is only allowed with --dry-run unless --allow-runtime-env-mismatch is set")
    if runtime.requested == "auto" and runtime.matched_alias is None and not args.dry_run and not runtime.allow_mismatch:
        raise SystemExit(
            "Could not detect a compatible runtime env. Set CROSS_DEVICE_RUNTIME_ENV, "
            "activate the right conda env, or use --allow-runtime-env-mismatch."
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


def _stage_hardware(spec: dict[str, Any], args: argparse.Namespace, stage: str) -> dict[str, Any]:
    if stage == "sim_cross":
        return detect_hardware(spec, args.replay_hardware)
    return detect_hardware(spec, args.hardware)


def _devices(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [device for device in spec.get("devices", []) if isinstance(device, dict)]
