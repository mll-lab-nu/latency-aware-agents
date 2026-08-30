from __future__ import annotations

import numpy as np


def min_max_normalize_state(
    state: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
) -> np.ndarray:
    state_range = state_max - state_min
    scaled = np.divide(
        state - state_min,
        state_range,
        out=np.zeros_like(state, dtype=np.float32),
        where=state_range != 0,
    )
    return np.clip(np.where(state_range == 0, 0.0, 2.0 * scaled - 1.0), -1.0, 1.0)
