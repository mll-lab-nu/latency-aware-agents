"""Reporting side of an eval run: summary payloads, W&B logging, progress prints.

Depends on :mod:`latency_bench.eval.config` for config interpretation and on the
logging package for summary/W&B helpers; it never calls back into the run
orchestration, so it stays a leaf of ``run.py``.
"""
from __future__ import annotations

import sys
from typing import Any

import wandb

from latency_bench.core.types import EpisodeMetrics
from latency_bench.eval.config import (
    _eval_episodes,
    _eval_latency_values,
    _infer_latency_steps,
)
from latency_bench.latency.config import latency_type_from_config
from latency_bench.logging.summary import build_queue_eval_summary, write_queue_eval_summary
from latency_bench.logging.wandb_eval import eval_scalar_payload, normalize_wandb_tags


def _print_progress_start(config: dict[str, Any], *, num_sweeps: int, num_episodes: int | None = None) -> None:
    if num_episodes is None:
        num_episodes = _eval_episodes(config)
    print(
        f"[bench] run={config['experiment'].get('name', 'latency_bench')} "
        f"sweeps={num_sweeps} "
        f"episodes_per_sweep={num_episodes} "
        f"total_episode_runs={num_sweeps * num_episodes}",
        file=sys.stderr,
        flush=True,
    )


def _progress_latency_label(latency_label: int | str | None) -> str:
    if latency_label is None:
        return "current"
    return str(latency_label)


def _write_non_sweep_summary(
    config: dict[str, Any],
    metrics: list[EpisodeMetrics],
) -> dict[str, Any] | None:
    latency_label = _summary_latency_label(config)
    if latency_label is None:
        return None
    return write_queue_eval_summary(
        config["logging"]["output_dir"],
        metrics,
        experiment_name=str(config["experiment"].get("name", "latency_bench")),
        seed=int(config["experiment"].get("seed", 0)),
        latency=latency_label,
    )


def _summary_latency_label(config: dict[str, Any]) -> int | str | None:
    executor_mode = str(config.get("executor", {}).get("mode", "")).strip().lower()
    if executor_mode == "realtime":
        return "measured"
    latency_steps = _infer_latency_steps(config)
    if latency_steps is not None:
        return latency_steps

    return latency_type_from_config(config["latency"])


def _summary_payloads_for_printing(
    config: dict[str, Any],
    metrics: list[EpisodeMetrics],
) -> list[dict[str, Any]]:
    experiment_name = str(config["experiment"].get("name", "latency_bench"))
    seed = int(config["experiment"].get("seed", 0))
    eval_latency_values = _eval_latency_values(config)
    if eval_latency_values is None:
        latency_label = _summary_latency_label(config)
        if latency_label is None:
            return []
        return [
            build_queue_eval_summary(
                metrics,
                experiment_name=experiment_name,
                seed=seed,
                latency=latency_label,
            )
        ]

    payloads = []
    for latency_steps in eval_latency_values:
        latency_metrics = [
            metric
            for metric in metrics
            if _metric_eval_latency_steps(metric) == int(latency_steps)
        ]
        payloads.append(
            build_queue_eval_summary(
                latency_metrics,
                experiment_name=experiment_name,
                seed=seed,
                latency=int(latency_steps),
            )
        )
    return payloads


def _metric_eval_latency_steps(metric: EpisodeMetrics) -> int | None:
    value = metric.metadata.get("eval_latency_steps")
    if value is None:
        return None
    return int(value)


def _log_corrected_eval_summaries_to_wandb(config: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    logging_cfg = config.get("logging", {})
    run_id = logging_cfg.get("wandb_name")
    project = logging_cfg.get("wandb_project")
    if not run_id or not project or not summaries:
        return

    init_kwargs = {
        "project": project,
        "entity": logging_cfg.get("wandb_user", logging_cfg.get("wandb_entity")),
        "id": str(run_id),
        "name": str(run_id),
        "resume": "allow",
        "group": logging_cfg.get("wandb_group"),
        "job_type": logging_cfg.get("wandb_job_type"),
        "tags": normalize_wandb_tags(logging_cfg.get("wandb_tags")),
    }
    wandb.init(**init_kwargs)
    wandb.define_metric("global_step")
    wandb.define_metric("eval_corrected/global_steps")
    wandb.define_metric("eval_corrected/*", step_metric="global_step")
    payload = eval_scalar_payload(
        summaries,
        env_steps=_corrected_eval_global_step(config, summaries),
        prefix="eval_corrected",
    )
    wandb.log(payload)
    if wandb.run is not None:
        for key, value in payload.items():
            if key.startswith("eval_corrected/") and key != "eval_corrected/global_steps":
                wandb.run.summary[key] = value
        wandb.run.finish()


def _corrected_eval_global_step(config: dict[str, Any], summaries: list[dict[str, Any]]) -> int:
    for key in ("train_env_steps", "checkpoint_global_step"):
        for summary in summaries:
            value = summary.get(key)
            if value is not None:
                return int(value)
    return int(config.get("training", {}).get("train_for_env_steps", 0))
