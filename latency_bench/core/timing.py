from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Iterator


class StageProfiler:
    """Enabled-aware, CUDA-sync-aware stage timing collector.

    Values in ``timings`` are stage durations in milliseconds.

    A disabled profiler is a no-op: ``time()`` returns a ``nullcontext`` so it
    records nothing and never synchronizes CUDA. Threading one through a hot
    path therefore costs nothing when profiling is off, which matters because
    accurate GPU-stage timing requires a ``torch.cuda.synchronize()`` at each
    stage boundary that would otherwise serialize the GPU.

    ``sync_cuda`` controls that boundary sync independently of ``enabled``. Keep
    it on (default) for offline GPU-stage profiling where accurate per-stage
    attribution matters. Turn it OFF on wall-clock-measured paths (env-fps
    calibration, realtime latency measurement), where a sync inside the measured
    call would perturb the very latency being measured.
    """

    def __init__(self, *, enabled: bool = False, sync_cuda: bool = True) -> None:
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self.timings: dict[str, float] = {}

    def time(self, stage: str):
        if not self.enabled:
            return nullcontext()
        return self._timer(stage)

    @contextmanager
    def _timer(self, stage: str) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[stage] = (time.perf_counter() - start) * 1000.0

    def _sync(self) -> None:
        if not self.sync_cuda:
            return
        # Lazy import keeps this module torch-free at import time so it stays on
        # the lightweight policy-base import chain.
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


# Shared disabled singleton. Never records (time() is a nullcontext), so sharing
# it as the contextvar default is safe.
NULL_PROFILER = StageProfiler(enabled=False)

_current_profiler: ContextVar[StageProfiler] = ContextVar(
    "current_profiler", default=NULL_PROFILER
)


def current_profiler() -> StageProfiler:
    """Return the profiler bound to the current context (NULL_PROFILER by default)."""
    return _current_profiler.get()


@contextmanager
def profiler_scope(profiler: StageProfiler) -> Iterator[StageProfiler]:
    """Bind ``profiler`` as the ambient profiler for the duration of the block.

    Must wrap the policy call in the same thread that runs it (contextvars are
    thread-local).
    """
    token = _current_profiler.set(profiler)
    try:
        yield profiler
    finally:
        _current_profiler.reset(token)
