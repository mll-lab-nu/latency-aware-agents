from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

from torch.utils.cpp_extension import CUDA_HOME, load


@lru_cache(maxsize=1)
def load_cule_atari_extension() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    cule_root = repo_root / "third_party" / "cule"
    extra_include_paths = [
        str(cule_root),
        str(cule_root / "third_party" / "agency"),
        str(Path(sys.prefix) / "include"),
    ]
    if CUDA_HOME is not None:
        cccl_include = Path(CUDA_HOME) / "include" / "cccl"
        if cccl_include.exists():
            extra_include_paths.append(str(cccl_include))
    return load(
        name="demon_attack_cule_atari",
        sources=[
            str(cule_root / "torchcule" / "frontend.cpp"),
            str(cule_root / "torchcule" / "backend.cu"),
        ],
        extra_include_paths=extra_include_paths,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-DCULE_ONLY_4K_COMPILE", "-std=c++17"],
        extra_ldflags=["-lgomp", "-lz"],
        verbose=False,
    )
