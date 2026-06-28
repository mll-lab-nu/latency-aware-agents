from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests live in mirrored subdirectories (tests/executors/, tests/latency/, ...)
# but shared support modules (helpers.py, executor_mode_helpers.py) stay at the
# tests/ root. Keep tests/ on sys.path so `from helpers import ...` resolves from
# any subdirectory regardless of pytest's per-file import dir.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

