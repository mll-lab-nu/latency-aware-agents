"""CLI entry point for latency-sensitive evaluation.

Thin orchestration layer: parses args, applies runtime overrides, and dispatches
to the eval suites / sweep drivers. The actual work lives in
:mod:`latency_bench.eval.driver` (single run), :mod:`latency_bench.eval.sweeps`
(sweeps/suites), :mod:`latency_bench.eval.reporting`, and
:mod:`latency_bench.eval.config`.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import latency_bench.eval.sweeps as eval_sweeps
from latency_bench.core.config import load_config
from latency_bench.eval.config import apply_runtime_overrides as _apply_runtime_overrides
from latency_bench.eval.config import _is_sample_factory_policy
from latency_bench.eval.reporting import (
    _log_corrected_eval_summaries_to_wandb,
    _summary_payloads_for_printing,
)
from latency_bench.eval.suites import has_eval_suites


def _has_explicit_sample_factory_checkpoint_selection(config: dict[str, Any]) -> bool:
    if not _is_sample_factory_policy(config):
        return False
    policy_cfg = config["policy"]
    if "checkpoint_kind" in policy_cfg:
        return True
    return "checkpoint" in policy_cfg and policy_cfg["checkpoint"] is not None


def _episode_ids_from_cli_args(args: argparse.Namespace) -> list[int] | None:
    if args.episode_start is None and args.episode_count is None:
        return None
    if args.episode_start is None or args.episode_count is None:
        raise SystemExit("--episode-start and --episode-count must be provided together")
    start = int(args.episode_start)
    count = int(args.episode_count)
    return list(range(start, start + count))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m latency_bench eval", description="Run a latency-sensitive control benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--task-manifest-path", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--latency-prompt-map-path", type=Path)
    parser.add_argument("--latency-raw-frames", type=int)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-count", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        task_manifest_path=args.task_manifest_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        checkpoint_kind=args.checkpoint_kind,
        latency_prompt_map_path=args.latency_prompt_map_path,
        latency_raw_frames=args.latency_raw_frames,
        max_env_steps=args.max_env_steps,
        parallel_envs=args.parallel_envs,
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
        print(f"OUTPUT_DIR={Path(config['logging']['output_dir']).expanduser().absolute()}")
        print("EVALUATION_STATUS=" + ("complete" if summaries else "no_results"))
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
    print(f"OUTPUT_DIR={Path(config['logging']['output_dir']).expanduser().absolute()}")
    print("EVALUATION_STATUS=" + ("complete" if metrics else "no_results"))
