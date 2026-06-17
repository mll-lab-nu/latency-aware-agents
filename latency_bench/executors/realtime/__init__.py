"""Realtime latency executor.

Three submodules — ``workers`` (IPC protocol + the two subprocess programs),
``pool`` (parent-side process pool + result coordinator), and ``executor`` (the
executor itself). The public surface is :class:`RealtimeLatencyExecutor`; reach
into ``realtime.workers`` / ``realtime.pool`` for the internals.
"""
from __future__ import annotations

# Re-exported so tests can monkeypatch ``latency_bench.executors.realtime.mp.get_context``;
# this is the same multiprocessing module the submodules use.
import multiprocessing as mp  # noqa: F401

from latency_bench.executors.realtime.executor import RealtimeLatencyExecutor

__all__ = ["RealtimeLatencyExecutor"]
