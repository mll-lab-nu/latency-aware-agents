"""CLI entry point for latency-sensitive evaluation.

Thin orchestration layer: parses args, applies runtime overrides, and dispatches
to the eval suites / sweep drivers. The actual work lives in
:mod:`latency_bench.eval.driver` (single run), :mod:`latency_bench.eval.sweeps`
(sweeps/suites), :mod:`latency_bench.eval.reporting`, and
:mod:`latency_bench.eval.config`.

``build_executor`` / ``run_from_config`` / ``run_eval_sweep_from_config`` /
``run_eval_suites_from_config`` are re-exported here for backward compatibility.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.eval.config import _is_sample_factory_policy
from latency_bench.eval.reporting import (
    _log_corrected_eval_summaries_to_wandb,
    _summary_payloads_for_printing,
)
from latency_bench.eval.suites import has_eval_suites
from latency_bench.eval import sweeps as eval_sweeps
# Re-exported for external callers (`from latency_bench.run import run_eval_*`).
# main() itself calls them via the `eval_sweeps` module so the retained and the
# explicit paths share one monkeypatch target.
from latency_bench.eval.sweeps import run_eval_suites_from_config, run_eval_sweep_from_config
from latency_bench.eval.driver import run_from_config
from latency_bench.executors.factory import build_executor

__all__ = [
    "main",
    "apply_runtime_overrides",
    "run_from_config",
    "run_eval_sweep_from_config",
    "run_eval_suites_from_config",
    "build_executor",
]


def apply_runtime_overrides(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint_kind: str | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        config["policy"]["checkpoint_path"] = str(checkpoint_path)
    if run_name is not None:
        config["experiment"]["name"] = str(run_name)
    if output_dir is not None:
        config["logging"]["output_dir"] = str(output_dir)
    if checkpoint_kind is not None:
        from training.common.config_adapter import (
            apply_checkpoint_kind_override,
            normalize_checkpoint_kind,
        )

        normalized_kind = normalize_checkpoint_kind(checkpoint_kind)
        apply_checkpoint_kind_override(config.setdefault("backend", {}), normalized_kind)
        config.setdefault("policy", {})["checkpoint_kind"] = normalized_kind
    return config


def _has_explicit_sample_factory_checkpoint_selection(config: dict[str, Any]) -> bool:
    if not _is_sample_factory_policy(config):
        return False
    policy_cfg = config["policy"]
    if "checkpoint_kind" in policy_cfg:
        return True
    return "checkpoint_path" in policy_cfg and bool(policy_cfg["checkpoint_path"])


def _episode_ids_from_cli_args(args: argparse.Namespace) -> list[int] | None:
    if args.episode_start is None and args.episode_count is None:
        return None
    if args.episode_start is None or args.episode_count is None:
        raise SystemExit("--episode-start and --episode-count must be provided together")
    start = int(args.episode_start)
    count = int(args.episode_count)
    return list(range(start, start + count))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a latency-sensitive control benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-count", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        checkpoint_kind=args.checkpoint_kind,
    )
    episode_ids = _episode_ids_from_cli_args(args)
    if has_eval_suites(config):
        if _has_explicit_sample_factory_checkpoint_selection(config):
            summaries = eval_sweeps.run_eval_suites_from_config(config, progress=True, episode_ids=episode_ids)
            _log_corrected_eval_summaries_to_wandb(config, summaries)
        else:
            summaries = eval_sweeps._run_retained_eval_checkpoints_from_config(config, progress=True, episode_ids=episode_ids)
            if not summaries and not _is_sample_factory_policy(config):
                summaries = eval_sweeps.run_eval_suites_from_config(config, progress=True, episode_ids=episode_ids)
                _log_corrected_eval_summaries_to_wandb(config, summaries)
        for summary in summaries:
            print(
                f"summary suite={summary['suite_name']} "
                f"latency={summary['latency']} "
                f"episodes={len(summary['returns'])} "
                f"mean_return={float(summary['mean_return']):.3f} "
                f"std_return={float(summary['std_return']):.3f} "
                f"mean_length={float(summary['mean_length']):.1f}"
            )
        return
    metrics = eval_sweeps.run_eval_sweep_from_config(config, progress=True, episode_ids=episode_ids)
    for episode_metrics in metrics:
        eval_latency = episode_metrics.metadata.get("eval_latency_steps")
        latency_fragment = f"eval_latency={eval_latency} " if eval_latency is not None else ""
        print(
            f"episode={episode_metrics.episode_id} "
            f"{latency_fragment}"
            f"return={episode_metrics.episode_return:.3f} "
            f"steps={episode_metrics.survival_steps} "
            f"mean_latency_ms={episode_metrics.mean_latency_ms}"
        )
    summaries = _summary_payloads_for_printing(config, metrics)
    _log_corrected_eval_summaries_to_wandb(config, summaries)
    for summary in summaries:
        print(
            f"summary latency={summary['latency']} "
            f"episodes={len(summary['returns'])} "
            f"mean_return={float(summary['mean_return']):.3f} "
            f"std_return={float(summary['std_return']):.3f} "
            f"mean_length={float(summary['mean_length']):.1f}"
        )


if __name__ == "__main__":
    main()
