"""Everything under tests/training/gpu/ requires a CUDA GPU (and often a built
CUDA/cule extension). Auto-mark the whole subtree as `gpu` so it is deselected by
the default `-m "not gpu"` filter; run it explicitly with `pytest -m gpu` on a box
that has the hardware. Individual modules keep their own importorskip/skip guards
so they degrade gracefully when the optional deps are missing."""

from __future__ import annotations

from pathlib import Path

import pytest

_GPU_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    # A subdir conftest's hook runs for the whole session, so scope by path:
    # only mark items that live under this gpu/ directory.
    for item in items:
        try:
            item.path.relative_to(_GPU_DIR)
        except ValueError:
            continue
        item.add_marker(pytest.mark.gpu)
