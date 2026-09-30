from __future__ import annotations

import time


def monotonic_ms() -> float:
    """Return the monotonic ``perf_counter`` reading in milliseconds."""
    return time.perf_counter() * 1000.0
