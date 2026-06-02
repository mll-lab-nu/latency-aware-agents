from __future__ import annotations

import os
import site
import subprocess
import sys
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension


def ensure_compatible_cuda_extension_toolchain() -> Path | None:
    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability()
    target_sm = f"sm_{major}{minor}"
    for nvcc_path in _candidate_nvcc_paths():
        if _nvcc_supports_sm(nvcc_path, target_sm):
            cuda_home = nvcc_path.parent.parent
            _activate_cuda_home(cuda_home)
            return cuda_home
    return None


def _candidate_nvcc_paths() -> Iterator[Path]:
    seen: set[Path] = set()
    for nvcc_path in _raw_candidate_nvcc_paths():
        resolved = nvcc_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if nvcc_path.is_file() and os.access(nvcc_path, os.X_OK):
            yield nvcc_path


def _raw_candidate_nvcc_paths() -> Iterator[Path]:
    for cuda_home in (os.environ.get("CUDA_HOME"), cpp_extension.CUDA_HOME):
        if cuda_home:
            yield Path(cuda_home) / "bin" / "nvcc"

    yield from _system_nvcc_paths()
    yield from _env_nvcc_paths()


def _system_nvcc_paths() -> Iterator[Path]:
    yield Path("/usr/local/cuda/bin/nvcc")
    local_root = Path("/usr/local")
    if local_root.is_dir():
        yield from sorted(local_root.glob("cuda-*/bin/nvcc"))


def _env_nvcc_paths() -> Iterator[Path]:
    env_roots = _unique_paths(
        path
        for path in (
            os.environ.get("CONDA_PREFIX"),
            os.environ.get("VIRTUAL_ENV"),
            sys.prefix,
        )
        if path
    )
    for env_root in env_roots:
        yield env_root / "bin" / "nvcc"
        yield from sorted(env_root.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))

    for site_packages in _site_package_paths():
        yield from sorted((site_packages / "nvidia").glob("cu*/bin/nvcc"))


def _site_package_paths() -> Iterator[Path]:
    yield from _unique_paths(site.getsitepackages())


def _unique_paths(paths: Iterable[str | Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in paths:
        item = Path(path)
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield item


def _nvcc_supports_sm(nvcc_path: Path, target_sm: str) -> bool:
    result = subprocess.run(
        [str(nvcc_path), "--list-gpu-code"],
        capture_output=True,
        text=True,
        check=False,
    )
    return target_sm in result.stdout.replace(",", " ").split()


def _activate_cuda_home(cuda_home: Path) -> None:
    os.environ["CUDA_HOME"] = str(cuda_home)
    _prepend_paths("PATH", [cuda_home / "bin"])
    library_paths = _cuda_runtime_library_paths(cuda_home)
    _prepend_paths("LD_LIBRARY_PATH", library_paths)
    _prepend_paths("LIBRARY_PATH", library_paths)
    cpp_extension.CUDA_HOME = str(cuda_home)


def _cuda_runtime_library_paths(cuda_home: Path) -> list[Path]:
    cuda_lib_dirs = [path for path in (cuda_home / "lib64", cuda_home / "lib") if path.is_dir()]
    if any((path / "libcudart.so").is_file() for path in cuda_lib_dirs):
        return cuda_lib_dirs

    versioned_cudarts = sorted(
        cudart
        for cuda_lib_dir in cuda_lib_dirs
        for cudart in cuda_lib_dir.glob("libcudart.so.*")
        if cudart.is_file()
    )
    if versioned_cudarts:
        compat_dir = Path(__file__).resolve().parents[2] / ".cache" / "cuda_link_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        compat_link = compat_dir / "libcudart.so"
        tmp_link = compat_link.with_name(f".{compat_link.name}.{os.getpid()}.{uuid.uuid4().hex}")
        tmp_link.symlink_to(versioned_cudarts[-1])
        os.replace(tmp_link, compat_link)
        return [compat_dir, *cuda_lib_dirs]

    return cuda_lib_dirs


def _prepend_paths(name: str, paths: Iterable[Path]) -> None:
    prefixes = [str(path) for path in paths if path.is_dir()]
    existing = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    os.environ[name] = os.pathsep.join(prefixes + [part for part in existing if part not in prefixes])
