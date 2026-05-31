from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from training.common.cuda_extension_build import ensure_compatible_cuda_extension_toolchain


@lru_cache(maxsize=1)
def load_deadly_corridor_cuda_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("deadly_corridor simulator=gpu requires CUDA, but torch.cuda.is_available() is false")
    ensure_compatible_cuda_extension_toolchain()
    root = Path(__file__).resolve().parent / "cuda_kernels"
    return load(
        name="deadly_corridor_cuda_kernels",
        sources=[
            str(root / "deadly_corridor_cuda.cpp"),
            str(root / "deadly_corridor_cuda_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
