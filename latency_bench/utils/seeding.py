"""Seed the Python, NumPy and PyTorch generators used by benchmark runtimes."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Reset CPU and CUDA random streams without changing determinism settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
