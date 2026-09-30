"""Multi-run evaluation orchestration: latency sweeps, eval suites, and the
retained-checkpoint replay. Each individual run is delegated to
:func:`latency_bench.eval.driver.run_from_config`; the shared run+summarize+
progress tail is factored into :func:`_run_one_sweep`.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import EpisodeMetrics
from latency_bench.eval.config import (
    _apply_eval_latency,
    _evaluation_seed,
    _is_sample_factory_policy,
    _latency_steps_from_config,
    _latency_steps_to_ms,
    _selected_episode_count,
    resolve_evaluation_config,
)
from latency_bench.eval.reporting import (
    _log_corrected_eval_summaries_to_wandb,
    _print_progress_start,
    _progress_latency_label,
    _summary_latency_label,
)
from latency_bench.eval.suites import has_eval_suites, parse_eval_suites
from latency_bench.eval.driver import run_from_config
from latency_bench.logging.summary import write_queue_eval_summary
from latency_bench.policy.factory import build_policy


def _prepare_fixed_sweep_config(
    config: dict[str, Any],
    latency_steps: int,
    output_dir: Path,
) -> tuple[dict[str, Any], float]:
    sweep_config = deepcopy(config)
    configured_latency_ms = _latency_steps_to_ms(sweep_config, latency_steps)
    _apply_eval_latency(sweep_config, latency_steps, configured_latency_ms)
    sweep_config["logging"]["output_dir"] = str(output_dir)
    return sweep_config, configured_latency_ms


def _run_one_sweep(
    sweep_config: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    summary_dir: Path,
    latency_label: int | str,
    summary_metadata: dict[str, Any],
    progress: bool,
    sweep_index: int,
    num_sweeps: int,
    num_episodes: int,
    episode_ids: Sequence[int] | None,
    policy: Any | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    """Run one prepared sweep config, write its summary, print progress.

    Shared tail of the latency-sweep and eval-suite loops; the caller prepares
    ``sweep_config`` (latency applied, output_dir set) and the summary target.
    """
    start_time = time.perf_counter()
    completed = 0

    def on_episode_complete(_episode_metrics: EpisodeMetrics) -> None:
        nonlocal completed
        completed += 1
        print(
            f"[bench] sweep {sweep_index}/{num_sweeps} episode {completed}/{num_episodes} done",
            file=sys.stderr,
            flush=True,
        )

    metrics = run_from_config(
        sweep_config,
        extra_metadata=summary_metadata,
        write_summary=False,
        on_episode_complete=on_episode_complete if progress else None,
        episode_ids=episode_ids,
        **({"policy": policy} if policy is not None else {}),
    )
    summary = write_queue_eval_summary(
        summary_dir,
        metrics,
        experiment_name=experiment_name,
        seed=seed,
        latency=latency_label,
        metadata=summary_metadata,
    )
    if progress:
        elapsed = time.perf_counter() - start_time
        print(
            f"[bench] sweep {sweep_index}/{num_sweeps} complete elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return metrics, summary


def run_eval_sweep_from_config(
    config: dict[str, Any],
    *,
    progress: bool = False,
    episode_ids: Sequence[int] | None = None,
    policy: Any | None = None,
) -> list[EpisodeMetrics]:
    if has_eval_suites(config):
        run_eval_suites_from_config(config, progress=progress, episode_ids=episode_ids, policy=policy)
        return []

    resolve_evaluation_config(config)
    eval_latency_values = config["evaluation"]["eval_latency_values"]
    if eval_latency_values is None:
        if not progress:
            return run_from_config(
                config,
                episode_ids=episode_ids,
                **({"policy": policy} if policy is not None else {}),
            )
        num_episodes = _selected_episode_count(config, episode_ids)
        _print_progress_start(config, num_sweeps=1, num_episodes=num_episodes)
        start_time = time.perf_counter()
        latency_label = _progress_latency_label(_summary_latency_label(config))
        print(f"[bench] sweep 1/1: eval_latency={latency_label}", file=sys.stderr, flush=True)
        completed = 0

        def on_episode_complete(_episode_metrics: EpisodeMetrics) -> None:
            nonlocal completed
            completed += 1
            print(
                f"[bench] sweep 1/1 episode {completed}/{num_episodes} done",
                file=sys.stderr,
                flush=True,
            )

        metrics = run_from_config(
            config,
            on_episode_complete=on_episode_complete,
            episode_ids=episode_ids,
            **({"policy": policy} if policy is not None else {}),
        )
        elapsed = time.perf_counter() - start_time
        print(f"[bench] sweep 1/1 complete elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
        return metrics

    base_output_dir = Path(config["logging"]["output_dir"])
    train_latency_steps = _latency_steps_from_config(config)
    experiment_name = str(config["experiment"].get("name", "latency_bench"))
    seed = _evaluation_seed(config)
    all_metrics: list[EpisodeMetrics] = []
    num_sweeps = len(eval_latency_values)
    num_episodes = _selected_episode_count(config, episode_ids)
    if progress:
        _print_progress_start(config, num_sweeps=num_sweeps, num_episodes=num_episodes)
    for sweep_index, eval_latency_steps in enumerate(eval_latency_values, start=1):
        if progress:
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps}: eval_latency={eval_latency_steps}",
                file=sys.stderr,
                flush=True,
            )
        sweep_config, configured_latency_ms = _prepare_fixed_sweep_config(
            config, eval_latency_steps, base_output_dir / f"eval_latency_{eval_latency_steps}"
        )
        summary_metadata = {
            "train_latency_steps": train_latency_steps,
            "eval_latency_steps": eval_latency_steps,
            "configured_latency_ms": configured_latency_ms,
            "latency_type": "fixed_sweep",
        }
        latency_metrics, _ = _run_one_sweep(
            sweep_config,
            experiment_name=experiment_name,
            seed=seed,
            summary_dir=base_output_dir,
            latency_label=eval_latency_steps,
            summary_metadata=summary_metadata,
            progress=progress,
            sweep_index=sweep_index,
            num_sweeps=num_sweeps,
            num_episodes=num_episodes,
            episode_ids=episode_ids,
            policy=policy,
        )
        all_metrics.extend(latency_metrics)
    return all_metrics


def run_eval_suites_from_config(
    config: dict[str, Any],
    *,
    progress: bool = False,
    extra_metadata: dict[str, Any] | None = None,
    episode_ids: Sequence[int] | None = None,
    policy: Any | None = None,
) -> list[dict[str, Any]]:
    suites = parse_eval_suites(config)
    resolve_evaluation_config(config)
    policy = _reuse_checkpoint_policy(config, policy)
    base_output_dir = Path(config["logging"]["output_dir"])
    train_latency_steps = _latency_steps_from_config(config)
    experiment_name = str(config["experiment"].get("name", "latency_bench"))
    seed = _evaluation_seed(config)
    num_episodes = _selected_episode_count(config, episode_ids)
    num_sweeps = sum(len(suite.fixed_latency_values or [suite.latency_label]) for suite in suites)
    if progress:
        _print_progress_start(config, num_sweeps=num_sweeps, num_episodes=num_episodes)

    summaries: list[dict[str, Any]] = []
    sweep_index = 0
    for suite in suites:
        suite_output_dir = base_output_dir / suite.name
        if suite.fixed_latency_values is not None:
            for eval_latency_steps in suite.fixed_latency_values:
                sweep_index += 1
                if progress:
                    print(
                        f"[bench] sweep {sweep_index}/{num_sweeps}: "
                        f"suite={suite.name} eval_latency={eval_latency_steps}",
                        file=sys.stderr,
                        flush=True,
                    )
                sweep_config, configured_latency_ms = _prepare_fixed_sweep_config(
                    config, eval_latency_steps,
                    suite_output_dir / f"eval_latency_{eval_latency_steps}",
                )
                summary_metadata = {
                    "train_latency_steps": train_latency_steps,
                    "eval_latency_steps": int(eval_latency_steps),
                    "configured_latency_ms": configured_latency_ms,
                    **suite.metadata,
                }
                if extra_metadata:
                    summary_metadata.update(extra_metadata)
                _, summary = _run_one_sweep(
                    sweep_config,
                    experiment_name=experiment_name,
                    seed=seed,
                    summary_dir=suite_output_dir,
                    latency_label=int(eval_latency_steps),
                    summary_metadata=summary_metadata,
                    progress=progress,
                    sweep_index=sweep_index,
                    num_sweeps=num_sweeps,
                    num_episodes=num_episodes,
                    episode_ids=episode_ids,
                    policy=policy,
                )
                summaries.append(summary)
            continue

        sweep_index += 1
        if progress:
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps}: "
                f"suite={suite.name} eval_latency={suite.latency_label}",
                file=sys.stderr,
                flush=True,
            )
        suite_config = deepcopy(config)
        suite_config["latency"] = dict(suite.latency_config or {})
        suite_config["logging"]["output_dir"] = str(suite_output_dir)
        summary_metadata = dict(suite.metadata)
        if extra_metadata:
            summary_metadata.update(extra_metadata)
        _, summary = _run_one_sweep(
            suite_config,
            experiment_name=experiment_name,
            seed=seed,
            summary_dir=suite_output_dir,
            latency_label=suite.latency_label,
            summary_metadata=summary_metadata,
            progress=progress,
            sweep_index=sweep_index,
            num_sweeps=num_sweeps,
            num_episodes=num_episodes,
            episode_ids=episode_ids,
            policy=policy,
        )
        summaries.append(summary)
    return summaries


def _reuse_checkpoint_policy(
    config: dict[str, Any],
    policy: Any | None,
) -> Any | None:
    if policy is not None:
        return policy

    executor_cfg = config["executor"]
    if executor_cfg["mode"] == "realtime" or (
        executor_cfg["mode"] == "simulated"
        and executor_cfg["simulated_inference_pool"]
    ):
        return None

    policy_type = config["policy"]["type"]
    if "checkpoint_path" not in config["policy"] or not (
        _is_sample_factory_policy(config) or policy_type == "starvla"
    ):
        return policy

    if policy_type == "starvla":
        from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config

        prepare_starvla_checkpoint_input_config(config)

    return build_policy(config, ActionResolver(config["env"]))


def _retained_eval_checkpoint_markers(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not _is_sample_factory_policy(config):
        return []
    marker_dir = (
        Path(str(config["backend"]["train_dir"])).expanduser()
        / str(config["experiment"]["name"])
        / "checkpoint_p0"
        / ".latency_bench_eval_checkpoints"
    )
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in marker_dir.glob("*.json")]
    return sorted(markers, key=lambda item: int(item["train_env_steps"]))


def _run_retained_eval_checkpoints_from_config(
    config: dict[str, Any],
    *,
    progress: bool,
    episode_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    base_output_dir = Path(str(config["logging"]["output_dir"]))
    all_summaries: list[dict[str, Any]] = []
    for marker in _retained_eval_checkpoint_markers(config):
        train_env_steps = int(marker["train_env_steps"])
        checkpoint_config = deepcopy(config)
        checkpoint_config["policy"]["checkpoint_path"] = str(marker["checkpoint_path"])
        checkpoint_config["logging"]["output_dir"] = str(
            base_output_dir / "eval_corrected_replay" / f"step_{train_env_steps}"
        )
        summaries = run_eval_suites_from_config(
            checkpoint_config,
            progress=progress,
            extra_metadata=marker,
            episode_ids=episode_ids,
        )
        _log_corrected_eval_summaries_to_wandb(checkpoint_config, summaries)
        all_summaries.extend(summaries)
    return all_summaries
