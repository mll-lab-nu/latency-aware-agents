from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_demon_attack_cuda_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("demon_attack simulator=gpu requires CUDA, but torch.cuda.is_available() is false")
    root = Path(__file__).resolve().parent / "cuda_kernels"
    return load(
        name="demon_attack_cuda_kernels",
        sources=[
            str(root / "demon_attack_cuda.cpp"),
            str(root / "demon_attack_cuda_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
