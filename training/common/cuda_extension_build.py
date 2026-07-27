from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch.utils.cpp_extension as cpp_extension


CUDA_ARCH_LIST = "9.0;12.0"


def ensure_compatible_cuda_extension_toolchain() -> Path:
    cuda_home = Path(os.environ["CUDA_HOME"])
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    _activate_cuda_home(cuda_home)
    return cuda_home


def _activate_cuda_home(cuda_home: Path) -> None:
    os.environ["CUDA_HOME"] = str(cuda_home)
    _prepend_paths("PATH", [cuda_home / "bin"])
    target_root = cuda_home / "targets" / "x86_64-linux"
    _prepend_paths("CPATH", [cuda_home / "include", target_root / "include"])
    library_paths = [
        cuda_home / "lib",
        cuda_home / "lib64",
        target_root / "lib",
        target_root / "lib" / "stubs",
    ]
    _prepend_paths("LD_LIBRARY_PATH", library_paths)
    _prepend_paths("LIBRARY_PATH", library_paths)
    cpp_extension.CUDA_HOME = str(cuda_home)


def _prepend_paths(name: str, paths: Iterable[Path]) -> None:
    prefixes = [str(path) for path in paths if path.is_dir()]
    existing = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    os.environ[name] = os.pathsep.join(prefixes + [part for part in existing if part not in prefixes])
