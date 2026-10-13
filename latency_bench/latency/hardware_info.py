"""Collect a best-effort hardware fingerprint for latency profiles.

The cross-device protocol records more than the GPU model so that latency
profiles can be interpreted later: GPU power cap and VRAM, CPU core counts, RAM,
and OS/kernel all influence measured inference latency. Every probe is wrapped so
a missing tool (no `nvidia-smi`, no `/proc`) degrades to a partial result instead
of raising.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Callable



# Preserve published deployment names; other models use their reported name.
_GPU_NAME_ALIASES = {
    'NVIDIA A100-SXM4-80GB': 'a100-sxm4-80gb',
    'NVIDIA A100-SXM4-40GB': 'a100-sxm4-40gb',
    'NVIDIA A100-PCIE-80GB': 'a100-pcie-80gb',
    'NVIDIA A100-PCIE-40GB': 'a100-pcie-40gb',
    'L40S': 'l40s',
    'NVIDIA L40S': 'l40s',
    'RTX 3090': 'rtx3090',
    'NVIDIA GeForce RTX 3090': 'rtx3090',
    'RTX 4090': 'rtx4090',
    'NVIDIA GeForce RTX 4090': 'rtx4090',
    'RTX 5090': 'rtx5090',
    'NVIDIA GeForce RTX 5090': 'rtx5090',
    'NVIDIA RTX PRO 6000 Blackwell Workstation Edition': 'rtx-pro-6000-blackwell-ws',
    'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'rtx-pro-6000-blackwell-server',
    'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition': 'rtx-pro-6000-blackwell-maxq-ws',
    'NVIDIA RPBZZZ6': 'rpbzzz6',
}

# GPU fields queried from nvidia-smi, in the order requested below.
_NVIDIA_SMI_FIELDS = (
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "power.limit",
    "power.default_limit",
    "driver_version",
    "pci.bus_id",
    "pcie.link.gen.max",
    "pcie.link.gen.current",
    "pcie.link.width.max",
    "pcie.link.width.current",
    "clocks.max.sm",
    "clocks.current.sm",
    "clocks.current.memory",
    "compute_cap",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
)

_PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "flash_attn",
    "xformers",
    "triton",
    "bitsandbytes",
    "numpy",
    "Pillow",
    "opencv-python",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "peft",
    "timm",
    "einops",
    "sentencepiece",
)

_RUNTIME_ENV_KEYS = (
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDNN_V8_API_ENABLED",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Return a best-effort hardware fingerprint for the current machine.

    Returns:
        A nested dict with `gpu`, `cpu`, `memory`, `system`, and `captured_at`
        keys. Any probe that fails contributes an empty section rather than
        raising, so the result is always JSON-serializable.
    """
    gpus = _collect_gpus()
    topology = _nvidia_smi_topology()
    return {
        "gpu": _collect_gpu_from_gpus(gpus, topology),
        "gpus": gpus,
        "cpu": _collect_cpu(),
        "memory": _collect_memory(),
        "system": _collect_system(),
        "runtime": _collect_runtime(),
        "torch": _collect_torch(),
        "packages": _collect_packages(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _collect_gpus() -> list[dict[str, Any]]:
    gpus = _nvidia_smi_gpus()
    torch_info = _torch_gpu()
    for index, gpu in enumerate(gpus):
        gpu["index"] = index
        for key, value in torch_info.items():
            gpu.setdefault(key, value)
    return gpus


def _collect_gpu_from_gpus(
    gpus: list[dict[str, Any]], topology: str | None = None
) -> dict[str, Any]:
    gpu: dict[str, Any] = dict(gpus[0]) if gpus else {}
    if topology:
        gpu["topology"] = topology
    return gpu


def describe_participating_gpus(
    *,
    gpus: list[dict[str, Any]],
    topology: str | None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Return the stable, public descriptor for the GPUs used by this process.

    ``gpus`` is the complete nvidia-smi inventory. CUDA's logical order is
    retained as ``cuda_index`` for worker mapping, while the public list is
    canonicalized by UUID so an input ordering change cannot create a new
    deployment identity.
    """
    selected = _select_visible_gpus(
        gpus,
        (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is None
            else cuda_visible_devices
        ),
    )
    canonical = sorted(selected, key=lambda item: item["gpu"]["uuid"])
    deployment_gpus = [
        {
            "uuid": item["gpu"]["uuid"],
            "name": item["gpu"]["name"],
            "cuda_index": item["cuda_index"],
        }
        for item in canonical
    ]
    topology_links = _normalize_gpu_topology(
        topology,
        selected_indices={item["physical_index"] for item in selected},
        gpus=gpus,
    )
    names = Counter(
        _GPU_NAME_ALIASES[gpu["name"]] if gpu["name"] in _GPU_NAME_ALIASES
        else re.sub(r"[^a-z0-9]+", "-", gpu["name"].lower()).strip("-")
        for gpu in deployment_gpus
    )
    gpu_class = "+".join(f"{count}x-{name}" for name, count in sorted(names.items()))
    return {
        "gpu_class": gpu_class,
        "gpus": deployment_gpus,
        "topology_links": topology_links,
    }


def describe_inference_deployment(
    fingerprint: dict[str, Any], devices: list[str],
) -> dict[str, Any]:
    """Bind worker slots to the selected CPU or physical CUDA devices."""
    cuda_indices = []
    for device in devices:
        if device == "cpu":
            continue
        if device == "cuda":
            cuda_indices.append(0)
        elif re.fullmatch(r"cuda:\d+", device):
            cuda_indices.append(int(device.split(":")[1]))
        else:
            raise ValueError(f"Unsupported profiling device: {device!r}; use cpu or cuda:N")
    if cuda_indices:
        visible = _select_visible_gpus(fingerprint["gpus"], os.environ.get("CUDA_VISIBLE_DEVICES"))
        selected = {index: visible[index]["gpu"]["uuid"] for index in cuda_indices}
        deployment = describe_participating_gpus(
            gpus=fingerprint["gpus"], topology=fingerprint["gpu"].get("topology"),
            cuda_visible_devices=",".join(selected.values()),
        )
        logical_indices = {uuid: index for index, uuid in selected.items()}
        for gpu in deployment["gpus"]:
            gpu["cuda_index"] = logical_indices[gpu["uuid"]]
        deployment["instance_id"] = instance_id_from_gpu_deployment(deployment)
    else:
        selected = {}
        deployment = {"gpu_class": "cpu", "gpus": [], "topology_links": []}
    if "cpu" in devices:
        # CPU profiles need a host identity, since there is no participating GPU UUID.
        identity = {"hostname": fingerprint["system"]["hostname"],
                    "cpu": fingerprint["cpu"], "gpus": sorted(selected.values())}
        # Frequency changes during a run and must not change its storage coordinate.
        identity["cpu"] = {key: value for key, value in identity["cpu"].items() if key != "cpu_mhz"}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        deployment["instance_id"] = "instance_" + hashlib.sha256(encoded).hexdigest()[:16]
        if selected:
            deployment["gpu_class"] += "+cpu"
    deployment["workers"] = [
        {"slot": slot, "device": device}
        for slot, device in enumerate(devices)
    ]
    return deployment


def public_hardware_projection(
    fingerprint: dict[str, Any],
    *,
    cuda_visible_devices: str | None = None,
    deployment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish device names and topology without host names or GPU UUIDs."""
    if deployment is None:
        deployment = describe_participating_gpus(
            gpus=fingerprint["gpus"], topology=fingerprint["gpu"].get("topology"),
            cuda_visible_devices=cuda_visible_devices,
        )
    slots = {gpu["uuid"]: gpu["cuda_index"] for gpu in deployment["gpus"]}
    public = {
        "gpu_class": deployment["gpu_class"],
        "gpus": [
            {"slot": gpu["cuda_index"], "name": gpu["name"]}
            for gpu in sorted(deployment["gpus"], key=lambda gpu: gpu["cuda_index"])
        ],
        "topology_links": [
            {"slot_a": slots[link["gpu_a"]], "slot_b": slots[link["gpu_b"]], "link": link["link"]}
            for link in deployment["topology_links"]
        ],
        "torch": fingerprint["torch"],
    }
    if deployment["gpus"]:
        public["driver_version"] = fingerprint["gpu"]["driver_version"]
    if "workers" in deployment:
        public["workers"] = [{"slot": w["slot"], "device": w["device"]} for w in deployment["workers"]]
        public["cpu"] = fingerprint["cpu"]
        public["memory"] = fingerprint["memory"]
    return public


def _select_visible_gpus(
    gpus: list[dict[str, Any]],
    cuda_visible_devices: str | None,
) -> list[dict[str, Any]]:
    if cuda_visible_devices is None:
        tokens = [str(index) for index in range(len(gpus))]
    else:
        tokens = [token.strip() for token in cuda_visible_devices.split(",")]
    if tokens == [""] or tokens == ["-1"]:
        return []
    by_uuid = {gpu["uuid"]: (index, gpu) for index, gpu in enumerate(gpus)}
    selected: list[dict[str, Any]] = []
    for cuda_index, token in enumerate(tokens):
        if token.startswith("GPU-"):
            physical_index, gpu = by_uuid[token]
        else:
            physical_index = int(token)
            gpu = gpus[physical_index]
        selected.append(
            {
                "physical_index": physical_index,
                "cuda_index": cuda_index,
                "gpu": gpu,
            }
        )
    return selected


_GPU_TOPOLOGY_LABEL = re.compile(r"^GPU(\d+)$")


def _normalize_gpu_topology(
    topology: str | None,
    *,
    selected_indices: set[int],
    gpus: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not topology:
        return []
    lines = [line.split() for line in topology.splitlines() if line.split()]
    header = {
        match.group(1): column + 1
        for column, token in enumerate(lines[0])
        if (match := _GPU_TOPOLOGY_LABEL.match(token))
    }
    rows = {
        int(tokens[0][3:]): tokens
        for tokens in lines[1:]
        if tokens and _GPU_TOPOLOGY_LABEL.match(tokens[0])
    }
    links: list[dict[str, str]] = []
    for left in sorted(selected_indices):
        for right in sorted(selected_indices):
            if left >= right:
                continue
            right_column = header[str(right)]
            links.append(
                {
                    "gpu_a": gpus[left]["uuid"],
                    "gpu_b": gpus[right]["uuid"],
                    "link": rows[left][right_column],
                }
            )
    return sorted(links, key=lambda link: (link["gpu_a"], link["gpu_b"]))


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    output = _run(["nvidia-smi", f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}", "--format=csv,noheader"])
    if not output:
        return []
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(_NVIDIA_SMI_FIELDS):
            return []
        gpus.append({field: value for field, value in zip(_NVIDIA_SMI_FIELDS, values)})
    return gpus


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


def safe_set(target: dict[str, Any], key: str, getter: Callable[[], Any]) -> None:
    """Set `target[key]` from `getter()`, skipping the key if the probe raises.

    Best-effort hardware/runtime probes are unavailable on many hosts (no GPU, older
    torch, missing attribute). Callers want "record it if you can" without a try/except
    around every field.
    """
    try:
        target[key] = getter()
    except Exception:
        pass


def torch_backend_flags(torch: Any) -> dict[str, Any]:
    """SDP / TF32 / cuDNN backend toggles as a flat dict (each field best-effort)."""
    backends: dict[str, Any] = {}
    safe_set(backends, "cuda_flash_sdp_enabled", lambda: bool(torch.backends.cuda.flash_sdp_enabled()))
    safe_set(backends, "cuda_mem_efficient_sdp_enabled", lambda: bool(torch.backends.cuda.mem_efficient_sdp_enabled()))
    safe_set(backends, "cuda_math_sdp_enabled", lambda: bool(torch.backends.cuda.math_sdp_enabled()))
    safe_set(backends, "cuda_cudnn_sdp_enabled", lambda: bool(torch.backends.cuda.cudnn_sdp_enabled()))
    safe_set(backends, "cuda_matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32))
    safe_set(backends, "cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32))
    safe_set(backends, "cudnn_benchmark", lambda: bool(torch.backends.cudnn.benchmark))
    return backends


def _collect_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "backends": torch_backend_flags(torch),
    }
    safe_set(info, "float32_matmul_precision", torch.get_float32_matmul_precision)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["current_device"] = int(torch.cuda.current_device())
            info["current_device_name"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))
            info["device_properties"] = _torch_device_properties(torch)
    except Exception:
        pass
    return info


def _torch_device_properties(torch: Any) -> list[dict[str, Any]]:
    properties: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        properties.append(
            {
                "index": index,
                "name": str(props.name),
                "total_memory": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
                "multi_processor_count": int(props.multi_processor_count),
            }
        )
    return properties


def _collect_cpu() -> dict[str, Any]:
    cpu: dict[str, Any] = {"logical_cores": os.cpu_count()}
    lscpu = _parse_lscpu()
    if lscpu:
        for key in (
            "Model name",
            "Socket(s)",
            "Core(s) per socket",
            "CPU(s)",
            "CPU max MHz",
            "CPU MHz",
            "Thread(s) per core",
            "NUMA node(s)",
            "L3 cache",
        ):
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
    try:
        system["libc"] = " ".join(item for item in platform.libc_ver() if item)
    except Exception:
        pass
    os_release = _os_release()
    if os_release:
        system["os_release"] = os_release
    return system


def _collect_runtime() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "cpu_affinity": _cpu_affinity(),
        "cgroup": _collect_cgroup(),
        "env": {
            key: os.environ[key]
            for key in _RUNTIME_ENV_KEYS
            if key in os.environ
        },
    }


def _cpu_affinity() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def _collect_cgroup() -> dict[str, str]:
    cgroup = {}
    for key, path in (
        ("cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("cpuset_cpus_effective", "/sys/fs/cgroup/cpuset.cpus.effective"),
    ):
        try:
            cgroup[key] = _read_text(path).strip()
        except Exception:
            pass
    return cgroup


def _collect_packages() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _nvidia_smi_topology() -> str | None:
    return _run(["nvidia-smi", "topo", "-m"])


def _os_release() -> dict[str, str]:
    path = "/etc/os-release"
    data: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, _, value = line.strip().partition("=")
                data[key] = value.strip('"')
    except Exception:
        return {}
    return data


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    return result.stdout.strip() or None


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _slug(key: str) -> str:
    return (
        key.lower()
        .replace("(s)", "s")
        .replace(" ", "_")
        .replace("/", "_")
    )


def instance_id_from_gpu_deployment(deployment: dict[str, Any]) -> str:
    """Identify the participating physical GPUs and topology, independent of ordering."""
    topology_links = [
        {
            "gpu_a": min(link["gpu_a"], link["gpu_b"]),
            "gpu_b": max(link["gpu_a"], link["gpu_b"]),
            "link": link["link"],
        }
        for link in deployment["topology_links"]
    ]
    identity = {
        "gpus": sorted(gpu["uuid"] for gpu in deployment["gpus"]),
        "topology_links": sorted(
            topology_links, key=lambda link: (link["gpu_a"], link["gpu_b"])
        ),
    }
    encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"instance_{digest[:16]}"
