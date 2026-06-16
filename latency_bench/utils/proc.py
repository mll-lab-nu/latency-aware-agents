"""Process resource-usage helpers."""

from __future__ import annotations

import resource


def cpu_seconds() -> float:
    """Total CPU seconds (user + system) for this process and its children."""
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )
