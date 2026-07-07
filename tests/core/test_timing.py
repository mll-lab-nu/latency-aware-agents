from __future__ import annotations

from contextlib import nullcontext

from latency_bench.core.timing import (
    NULL_PROFILER,
    StageProfiler,
    current_profiler,
    profiler_scope,
)


def test_disabled_profiler_is_noop_and_needs_no_torch():
    profiler = StageProfiler(enabled=False)
    # time() returns a nullcontext, so entering it never touches CUDA / torch.
    ctx = profiler.time("stage")
    assert isinstance(ctx, nullcontext)
    with ctx:
        pass
    assert profiler.timings == {}


def test_enabled_profiler_records_stage_durations():
    profiler = StageProfiler(enabled=True)
    profiler._sync = lambda: None  # avoid CUDA in CI
    with profiler.time("stage_a"):
        pass
    with profiler.time("stage_b"):
        pass
    assert set(profiler.timings) == {"stage_a", "stage_b"}
    assert profiler.timings["stage_a"] >= 0.0


def test_sync_disabled_records_without_touching_torch():
    # sync_cuda=False must record stages but never import/call torch, so it is
    # safe on wall-clock-measured paths (calibration, realtime) with no perturbation.
    profiler = StageProfiler(enabled=True, sync_cuda=False)
    with profiler.time("stage"):
        pass
    assert "stage" in profiler.timings


def test_profiler_scope_sets_and_restores_ambient_profiler():
    assert current_profiler() is NULL_PROFILER
    profiler = StageProfiler(enabled=True)
    with profiler_scope(profiler):
        assert current_profiler() is profiler
    assert current_profiler() is NULL_PROFILER


def test_default_ambient_profiler_is_disabled_singleton():
    assert current_profiler() is NULL_PROFILER
    assert NULL_PROFILER.enabled is False
