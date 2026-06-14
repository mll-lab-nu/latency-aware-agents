"""Collect a best-effort hardware fingerprint for latency profiles.

The cross-device protocol records more than the GPU model so that latency
profiles can be interpreted later: GPU power cap and VRAM, CPU core counts, RAM,
and OS/kernel all influence measured inference latency. Every probe is wrapped so
a missing tool (no `nvidia-smi`, no `/proc`) degrades to a partial result instead
of raising.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any


# GPU fields queried from nvidia-smi, in the order requested below.
_NVIDIA_SMI_FIELDS = (
    "name",
    "memory.total",
    "power.limit",
    "power.default_limit",
    "driver_version",
    "pcie.link.gen.max",
    "clocks.max.sm",
    "compute_cap",
)


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Return a best-effort hardware fingerprint for the current machine.

    Returns:
        A nested dict with `gpu`, `cpu`, `memory`, `system`, and `captured_at`
        keys. Any probe that fails contributes an empty section rather than
        raising, so the result is always JSON-serializable.
    """
    return {
        "gpu": _collect_gpu(),
        "cpu": _collect_cpu(),
        "memory": _collect_memory(),
        "system": _collect_system(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _collect_gpu() -> dict[str, Any]:
    gpu: dict[str, Any] = {}
    gpu.update(_nvidia_smi_gpu())
    torch_info = _torch_gpu()
    for key, value in torch_info.items():
        gpu.setdefault(key, value)
    return gpu


def _nvidia_smi_gpu() -> dict[str, Any]:
    output = _run(["nvidia-smi", f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}", "--format=csv,noheader"])
    if not output:
        return {}
    first_line = output.splitlines()[0]
    values = [item.strip() for item in first_line.split(",")]
    if len(values) != len(_NVIDIA_SMI_FIELDS):
        return {}
    return {field: value for field, value in zip(_NVIDIA_SMI_FIELDS, values)}


def _torch_gpu() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    info: dict[str, Any] = {}
    try:
        if torch.cuda.is_available():
            info["device_count"] = int(torch.cuda.device_count())
            info["torch_device_name"] = str(torch.cuda.get_device_name(0))
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        return info
    return info


def _collect_cpu() -> dict[str, Any]:
    cpu: dict[str, Any] = {"logical_cores": os.cpu_count()}
    lscpu = _parse_lscpu()
    if lscpu:
        for key in ("Model name", "Socket(s)", "Core(s) per socket", "CPU(s)", "CPU max MHz", "Thread(s) per core"):
            if key in lscpu:
                cpu[_slug(key)] = lscpu[key]
    else:
        model = _cpuinfo_model()
        if model:
            cpu["model_name"] = model
    return cpu


def _parse_lscpu() -> dict[str, str]:
    output = _run(["lscpu"])
    if not output:
        return {}
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed


def _cpuinfo_model() -> str | None:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _collect_memory() -> dict[str, Any]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kib = int(parts[1])
                    return {"total_kib": kib, "total_gib": round(kib / (1024 * 1024), 2)}
    except Exception:
        return {}
    return {}


def _collect_system() -> dict[str, Any]:
    system: dict[str, Any] = {}
    try:
        system["hostname"] = socket.gethostname()
    except Exception:
        pass
    try:
        system["platform"] = platform.platform()
    except Exception:
        pass
    try:
        system["python_version"] = platform.python_version()
    except Exception:
        pass
    return system


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    return result.stdout.strip() or None


def _slug(key: str) -> str:
    return (
        key.lower()
        .replace("(s)", "s")
        .replace(" ", "_")
        .replace("/", "_")
    )
