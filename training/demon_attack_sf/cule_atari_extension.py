from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_cule_atari_extension() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    cule_root = repo_root / "third_party" / "cule"
    return load(
        name="demon_attack_cule_atari",
        sources=[
            str(cule_root / "torchcule" / "frontend.cpp"),
            str(cule_root / "torchcule" / "backend.cu"),
        ],
        extra_include_paths=[
            str(cule_root),
            str(cule_root / "third_party" / "agency"),
            str(cule_root / "third_party" / "pybind11" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++14"],
        extra_cuda_cflags=["-O3", "-DCULE_ONLY_4K_COMPILE", "-std=c++14"],
        extra_ldflags=["-lgomp", "-lz"],
        verbose=False,
    )
