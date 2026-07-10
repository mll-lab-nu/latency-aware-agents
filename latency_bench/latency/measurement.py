from __future__ import annotations

import time
import torch
from collections.abc import Callable
from typing import Any

from latency_bench.core.types import LatencyRecord


class RuntimeLatencyMeasurer:
    """Measure runtime policy latency and emit latency records."""

    def __init__(self, device: str | None = None, sync_cuda: bool = True):
        self.device = device
        self.sync_cuda = sync_cuda
        self._next_obs_id = 0

    def run_and_measure(self, fn: Callable, *args, **kwargs) -> tuple[Any, LatencyRecord]:
        obs_id = kwargs.pop("obs_id", self._next_obs_id)
        self._next_obs_id = max(self._next_obs_id, obs_id + 1)
        self.synchronize_if_needed()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.synchronize_if_needed()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        return result, LatencyRecord(
            obs_id=obs_id,
            latency_ms=elapsed_ms,
            start_wall_time=start,
            end_wall_time=end,
            inference_ms=elapsed_ms,
            device=self.device,
            latency_type="measured",
        )

    def synchronize_if_needed(self) -> None:
        if not self.sync_cuda or self.device is None or not self.device.startswith("cuda"):
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
