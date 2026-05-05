"""Compatibility shim for the old runtime latency profiler module."""

from __future__ import annotations

from latency_bench.latency.measurement import LatencyProfiler, RuntimeLatencyMeasurer

__all__ = ["RuntimeLatencyMeasurer", "LatencyProfiler"]
