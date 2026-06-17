"""Latency-sensitive evaluation.

A small acyclic layering for running and reporting evaluations:

- :mod:`~latency_bench.eval.config`    config / latency interpretation (leaf)
- :mod:`~latency_bench.eval.reporting` summary / W&B / progress
- :mod:`~latency_bench.eval.suites`    eval-suite spec parsing
- :mod:`~latency_bench.eval.driver`    ``run_from_config`` (single-run driver)
- :mod:`~latency_bench.eval.sweeps`    latency-sweep / suite / retained orchestration

The CLI entry point stays at :mod:`latency_bench.run`.
"""
from __future__ import annotations

from latency_bench.eval.driver import run_from_config
from latency_bench.eval.suites import has_eval_suites, parse_eval_suites
from latency_bench.eval.sweeps import run_eval_suites_from_config, run_eval_sweep_from_config

__all__ = [
    "run_from_config",
    "run_eval_sweep_from_config",
    "run_eval_suites_from_config",
    "has_eval_suites",
    "parse_eval_suites",
]
