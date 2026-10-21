"""Array helpers shared across the codebase."""

from __future__ import annotations

from typing import Any
from importlib.util import find_spec

import numpy as np

if find_spec("torch") is None:
    torch = None
else:
    import torch


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to NumPy arrays.

    Args:
        value: Tensor, array, or array-like value.

    Returns:
        NumPy array view or copy.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def looks_chw(arr: np.ndarray) -> bool:
    """Return True when ``arr`` looks channel-first (C, H, W) with C in {1, 3}."""
    return arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}
