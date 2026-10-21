"""Seed the Python, NumPy and PyTorch generators used by benchmark runtimes."""

from __future__ import annotations

import random
from importlib.util import find_spec

import numpy as np

# CPU custom components do not require the optional Torch runtime.
if find_spec("torch") is None:
    torch = None
else:
    import torch


def seed_everything(seed: int) -> int:
    """Reset CPU and CUDA random streams without changing determinism settings."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed
