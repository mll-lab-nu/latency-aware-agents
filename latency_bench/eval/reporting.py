"""Reporting side of an eval run: summary payloads, W&B logging, progress prints.

Depends on :mod:`latency_bench.eval.config` for config interpretation and on the
logging package for summary/W&B helpers; it never calls back into the run
orchestration, so it stays a leaf of ``run.py``.
"""
from __future__ import annotations

import sys
from typing import Any

from latency_bench.core.types import EpisodeMetrics
from latency_bench.eval.config import (
    _episodes,
    _evaluation_seed,
    _latency_steps_from_config,
)
from latency_bench.latency.config import latency_type_from_config
from latency_bench.logging.summary import build_queue_eval_summary, write_queue_eval_summary
from latency_bench.logging.wandb_eval import eval_scalar_payload, normalize_wandb_tags


def _print_progress_start(config: dict[str, Any], *, num_sweeps: int, num_episodes: int | None = None) -> None:
    if num_episodes is None:
        num_episodes = _episodes(config)
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
        seed=_evaluation_seed(config),
        latency=latency_label,
    )


def _summary_latency_label(config: dict[str, Any]) -> int | str | None:
    executor_mode = config["executor"]["mode"].strip().lower()
    if executor_mode == "realtime":
        return "measured"
    if executor_mode == "paused":
        return "zero"
    latency_steps = _latency_steps_from_config(config)
    if latency_steps is not None:
        return latency_steps

    return latency_type_from_config(config["latency"])


def _summary_payloads_for_printing(
    config: dict[str, Any],
    metrics: list[EpisodeMetrics],
) -> list[dict[str, Any]]:
    experiment_name = str(config["experiment"].get("name", "latency_bench"))
    seed = _evaluation_seed(config)
    latency_label = _summary_latency_label(config)
    if latency_label is None:
        return []
    return [build_queue_eval_summary(
        metrics, experiment_name=experiment_name, seed=seed, latency=latency_label,
    )]

def _log_corrected_eval_summaries_to_wandb(
    config: dict[str, Any], summaries: list[dict[str, Any]], *, prefix: str = "eval_corrected",
) -> None:
    logging_cfg = config.get("logging", {})
    run_id = logging_cfg.get("wandb_name")
    project = logging_cfg.get("wandb_project")
    if not run_id or not project or not summaries:
        return

    import wandb

    init_kwargs = {
        "project": project,
        "entity": logging_cfg.get("wandb_user", logging_cfg.get("wandb_entity")),
        "id": str(run_id),
        "name": str(logging_cfg.get("wandb_display_name", run_id)),
        "resume": "allow",
        "group": logging_cfg.get("wandb_group"),
        "job_type": logging_cfg.get("wandb_job_type"),
        "tags": normalize_wandb_tags(logging_cfg.get("wandb_tags")),
    }
    wandb.init(**init_kwargs)
    wandb.define_metric("global_step")
    wandb.define_metric(f"{prefix}/global_steps")
    wandb.define_metric(f"{prefix}/*", step_metric="global_step")
    payload = _corrected_eval_scalar_payload(config, summaries, prefix=prefix)
    wandb.log(payload)
    if wandb.run is not None:
        for key, value in payload.items():
            if key.startswith(f"{prefix}/") and key != f"{prefix}/global_steps":
                wandb.run.summary[key] = value
        wandb.run.finish()


def _corrected_eval_scalar_payload(
    config: dict[str, Any], summaries: list[dict[str, Any]], *, prefix: str = "eval_corrected",
) -> dict[str, float]:
    return eval_scalar_payload(
        summaries,
        env_steps=_corrected_eval_global_step(config, summaries),
        prefix=prefix,
    )


def _corrected_eval_global_step(config: dict[str, Any], summaries: list[dict[str, Any]]) -> int:
    for key in ("train_env_steps", "checkpoint_global_step"):
        for summary in summaries:
            value = summary.get(key)
            if value is not None:
                return int(value)
    return int(config.get("training", {}).get("train_for_env_steps", 0))
