from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.core.types import EpisodeMetrics
from latency_bench.eval_suites import has_eval_suites, parse_eval_suites
from latency_bench.executors.factory import build_executor
from latency_bench.logging.summary import build_queue_eval_summary, write_queue_eval_summary
from latency_bench.logging.wandb_eval import eval_scalar_payload, normalize_wandb_tags

_SAMPLE_FACTORY_POLICY_TYPES = {
    "flappy_sf",
    "demon_attack_sf",
    "deadly_corridor_sf",
}


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


def _is_sample_factory_policy(config: dict[str, Any]) -> bool:
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    return policy_type in _SAMPLE_FACTORY_POLICY_TYPES


def _episode_seed(config: dict[str, Any], episode_id: int) -> int:
    seed = int(config["experiment"].get("seed", 0))
    if _is_sample_factory_policy(config):
        return seed + 104729 + int(episode_id)
    return seed + int(episode_id)


def _eval_max_steps(config: dict[str, Any]) -> int:
    evaluation_cfg = config.get("evaluation")
    if isinstance(evaluation_cfg, dict) and evaluation_cfg.get("eval_max_steps") is not None:
        return int(evaluation_cfg["eval_max_steps"])
    raise ValueError("evaluation.eval_max_steps is required for latency_bench episode limits")


def run_from_config(
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    *,
    write_summary: bool = True,
    on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
) -> list[EpisodeMetrics]:
    experiment_cfg = config["experiment"]
    policy_cfg = config["policy"]
    logging_cfg = config["logging"]
    seed = int(experiment_cfg.get("seed", 0))
    num_episodes = int(experiment_cfg.get("num_episodes", 1))
    eval_max_steps = _eval_max_steps(config)
    if _is_sample_factory_policy(config):
        from training.common.utils import seed_everything

        seed_everything(seed)

    executor = build_executor(config)
    metrics = []
    try:
        run_episodes = getattr(executor, "run_episodes", None)
        if callable(run_episodes):
            episode_ids = list(range(num_episodes))
            run_episodes_kwargs: dict[str, Any] = {
                "episode_ids": episode_ids,
                "seeds": [_episode_seed(config, episode_id) for episode_id in episode_ids],
                "eval_max_steps": eval_max_steps,
            }
            if on_episode_complete is not None:
                run_episodes_kwargs["on_episode_complete"] = on_episode_complete
            metrics = list(run_episodes(**run_episodes_kwargs))
        else:
            for episode_id in range(num_episodes):
                episode_metrics = executor.run_episode(
                    episode_id=episode_id,
                    seed=_episode_seed(config, episode_id),
                    eval_max_steps=eval_max_steps,
                )
                metrics.append(episode_metrics)
                if on_episode_complete is not None:
                    on_episode_complete(episode_metrics)
        metrics.sort(key=lambda item: int(item.episode_id))
        for episode_metrics in metrics:
            for key, value in _evaluation_raw_fact_metadata(config, int(episode_metrics.episode_id)).items():
                if episode_metrics.metadata.get(key) is None:
                    episode_metrics.metadata[key] = value
            episode_metrics.metadata["config_name"] = experiment_cfg.get("name")
            episode_metrics.metadata["run_name"] = experiment_cfg.get("name")
            if "checkpoint_path" in policy_cfg:
                episode_metrics.metadata["checkpoint_path"] = str(policy_cfg["checkpoint_path"])
            if "checkpoint_kind" in policy_cfg:
                episode_metrics.metadata["checkpoint_kind"] = str(policy_cfg["checkpoint_kind"])
            episode_metrics.metadata["output_dir"] = str(logging_cfg["output_dir"])
            if extra_metadata:
                episode_metrics.metadata.update(extra_metadata)
        if executor.logger is not None:
            executor.logger.flush()
        if write_summary:
            _write_non_sweep_summary(config, metrics)
    finally:
        close = getattr(executor, "close", None)
        if callable(close):
            close()
        else:
            executor.env.close()
    return metrics


def _evaluation_raw_fact_metadata(config: dict[str, Any], episode_id: int) -> dict[str, Any]:
    env_cfg = config.get("env", {})
    policy_cfg = config.get("policy", {})
    latency_cfg = config.get("latency", {})
    evaluation_cfg = config.get("evaluation", {})
    executor_cfg = config.get("executor", {})
    env_fps = float(env_cfg["env_fps"]) if "env_fps" in env_cfg else None
    obs_fps = float(env_cfg["obs_fps"]) if "obs_fps" in env_cfg else None
    frame_ms = None if env_fps is None or env_fps <= 0 else 1000.0 / env_fps
    executor_mode = str(executor_cfg.get("mode", "")).strip().lower()
    latency_type = str(latency_cfg.get("type", "")).strip().lower()
    if executor_mode == "paused":
        latency_type = "zero"
    elif executor_mode == "realtime":
        latency_type = "measured"
    return {
        "mode": executor_cfg.get("mode"),
        "episode_seed": _episode_seed(config, episode_id),
        "policy_id": _metadata_id(policy_cfg, "policy_id", "id", "type"),
        "env_id": _metadata_id(env_cfg, "env_id", "id", "name"),
        "profile_id": latency_cfg.get("profile_id"),
        "profile_type": latency_cfg.get("profile_type"),
        "target_device": latency_cfg.get("target_device"),
        "replay_host": latency_cfg.get("replay_host", evaluation_cfg.get("replay_host")),
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "frame_ms": frame_ms,
        "latency_type": latency_type,
    }


def _metadata_id(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return str(value)
    return None


def run_eval_sweep_from_config(
    config: dict[str, Any],
    *,
    progress: bool = False,
) -> list[EpisodeMetrics]:
    if has_eval_suites(config):
        run_eval_suites_from_config(config, progress=progress)
        return []

    eval_latency_values = _eval_latency_values(config)
    if eval_latency_values is None:
        if not progress:
            return run_from_config(config)
        _print_progress_start(config, num_sweeps=1)
        start_time = time.perf_counter()
        latency_label = _progress_latency_label(_summary_latency_label(config))
        print(f"[bench] sweep 1/1: eval_latency={latency_label}", file=sys.stderr, flush=True)
        completed = 0
        num_episodes = int(config["experiment"].get("num_episodes", 1))

        def on_episode_complete(_episode_metrics: EpisodeMetrics) -> None:
            nonlocal completed
            completed += 1
            print(
                f"[bench] sweep 1/1 episode {completed}/{num_episodes} done",
                file=sys.stderr,
                flush=True,
            )

        metrics = run_from_config(config, on_episode_complete=on_episode_complete)
        elapsed = time.perf_counter() - start_time
        print(f"[bench] sweep 1/1 complete elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
        return metrics

    base_output_dir = Path(config["logging"]["output_dir"])
    train_latency_steps = _train_latency_steps(config)
    all_metrics: list[EpisodeMetrics] = []
    num_sweeps = len(eval_latency_values)
    num_episodes = int(config["experiment"].get("num_episodes", 1))
    if progress:
        _print_progress_start(config, num_sweeps=num_sweeps)
    for sweep_index, eval_latency_steps in enumerate(eval_latency_values, start=1):
        if progress:
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps}: eval_latency={eval_latency_steps}",
                file=sys.stderr,
                flush=True,
            )
        start_time = time.perf_counter()
        sweep_config = deepcopy(config)
        configured_latency_ms = _latency_steps_to_ms(sweep_config, eval_latency_steps)
        _apply_eval_latency(sweep_config, eval_latency_steps, configured_latency_ms)
        sweep_output_dir = base_output_dir / f"eval_latency_{eval_latency_steps}"
        sweep_config["logging"]["output_dir"] = str(sweep_output_dir)
        summary_metadata = {
            "train_latency_steps": train_latency_steps,
            "eval_latency_steps": eval_latency_steps,
            "configured_latency_ms": configured_latency_ms,
        }
        completed = 0

        def on_episode_complete(_episode_metrics: EpisodeMetrics) -> None:
            nonlocal completed
            completed += 1
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps} episode {completed}/{num_episodes} done",
                file=sys.stderr,
                flush=True,
            )

        latency_metrics = run_from_config(
            sweep_config,
            extra_metadata=summary_metadata,
            write_summary=False,
            on_episode_complete=on_episode_complete if progress else None,
        )
        write_queue_eval_summary(
            base_output_dir,
            latency_metrics,
            experiment_name=str(config["experiment"].get("name", "latency_bench")),
            seed=int(config["experiment"].get("seed", 0)),
            latency=eval_latency_steps,
            metadata=summary_metadata,
        )
        if progress:
            elapsed = time.perf_counter() - start_time
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps} complete elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        all_metrics.extend(latency_metrics)
    return all_metrics


def run_eval_suites_from_config(
    config: dict[str, Any],
    *,
    progress: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    suites = parse_eval_suites(config)
    base_output_dir = Path(config["logging"]["output_dir"])
    train_latency_steps = _train_latency_steps(config)
    num_episodes = int(config["experiment"].get("num_episodes", 1))
    num_sweeps = sum(len(suite.fixed_latency_values or [suite.latency_label]) for suite in suites)
    if progress:
        _print_progress_start(config, num_sweeps=num_sweeps)

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
                start_time = time.perf_counter()
                sweep_config = deepcopy(config)
                configured_latency_ms = _latency_steps_to_ms(sweep_config, eval_latency_steps)
                _apply_eval_latency(sweep_config, eval_latency_steps, configured_latency_ms)
                sweep_config["logging"]["output_dir"] = str(suite_output_dir / f"eval_latency_{eval_latency_steps}")
                summary_metadata = {
                    "train_latency_steps": train_latency_steps,
                    "eval_latency_steps": int(eval_latency_steps),
                    "configured_latency_ms": configured_latency_ms,
                    **suite.metadata,
                }
                if extra_metadata:
                    summary_metadata.update(extra_metadata)
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
                )
                summary = write_queue_eval_summary(
                    suite_output_dir,
                    metrics,
                    experiment_name=str(config["experiment"].get("name", "latency_bench")),
                    seed=int(config["experiment"].get("seed", 0)),
                    latency=int(eval_latency_steps),
                    metadata=summary_metadata,
                )
                summaries.append(summary)
                if progress:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[bench] sweep {sweep_index}/{num_sweeps} complete elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
            continue

        sweep_index += 1
        if progress:
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps}: "
                f"suite={suite.name} eval_latency={suite.latency_label}",
                file=sys.stderr,
                flush=True,
            )
        start_time = time.perf_counter()
        suite_config = deepcopy(config)
        suite_config["latency"] = dict(suite.latency_config or {})
        suite_config["logging"]["output_dir"] = str(suite_output_dir)
        summary_metadata = dict(suite.metadata)
        if extra_metadata:
            summary_metadata.update(extra_metadata)
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
            suite_config,
            extra_metadata=summary_metadata,
            write_summary=False,
            on_episode_complete=on_episode_complete if progress else None,
        )
        summary = write_queue_eval_summary(
            suite_output_dir,
            metrics,
            experiment_name=str(config["experiment"].get("name", "latency_bench")),
            seed=int(config["experiment"].get("seed", 0)),
            latency=suite.latency_label,
            metadata=summary_metadata,
        )
        summaries.append(summary)
        if progress:
            elapsed = time.perf_counter() - start_time
            print(
                f"[bench] sweep {sweep_index}/{num_sweeps} complete elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return summaries


def _print_progress_start(config: dict[str, Any], *, num_sweeps: int) -> None:
    num_episodes = int(config["experiment"].get("num_episodes", 1))
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


def _eval_latency_values(config: dict[str, Any]) -> list[int] | None:
    evaluation_cfg = config.get("evaluation")
    if not isinstance(evaluation_cfg, dict) or "eval_latency_values" not in evaluation_cfg:
        return None
    values = evaluation_cfg["eval_latency_values"]
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [item.strip() for item in str(values).split(",") if item.strip()]
    elif not isinstance(values, Sequence):
        values = [values]
    return [int(value) for value in values]


def _train_latency_steps(config: dict[str, Any]) -> int | None:
    return _latency_steps_from_config(config)


def _latency_steps_to_ms(config: dict[str, Any], latency_steps: int) -> float:
    env_fps = float(config["env"]["env_fps"])
    if env_fps <= 0:
        raise ValueError("env.env_fps must be positive")
    return int(latency_steps) * 1000.0 / env_fps


def _apply_eval_latency(config: dict[str, Any], latency_steps: int, latency_ms: float) -> None:
    latency_cfg = config.setdefault("latency", {})
    if int(latency_steps) == 0:
        latency_cfg["type"] = "zero"
        latency_cfg.pop("fixed_latency_ms", None)
        return
    latency_cfg["type"] = "fixed"
    latency_cfg["fixed_latency_ms"] = float(latency_ms)


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


def _infer_latency_steps(config: dict[str, Any]) -> int | None:
    return _latency_steps_from_config(config)


def _latency_steps_from_config(config: dict[str, Any]) -> int | None:
    executor_mode = str(config.get("executor", {}).get("mode", "")).strip().lower()
    if executor_mode == "realtime":
        return None
    latency_cfg = config.get("latency", {})
    latency_type = str(latency_cfg.get("type", "")).strip().lower()
    if latency_type == "zero":
        return 0

    if "fixed_latency_ms" not in latency_cfg:
        return None
    latency_steps = (
        float(latency_cfg["fixed_latency_ms"])
        * float(config["env"]["env_fps"])
        / 1000.0
    )
    rounded_steps = round(latency_steps)
    if abs(latency_steps - rounded_steps) > 1e-9:
        return None
    return int(rounded_steps)


def _summary_latency_label(config: dict[str, Any]) -> int | str | None:
    executor_mode = str(config.get("executor", {}).get("mode", "")).strip().lower()
    if executor_mode == "realtime":
        return "measured"
    latency_steps = _infer_latency_steps(config)
    if latency_steps is not None:
        return latency_steps

    latency_type = str(config.get("latency", {}).get("type", "")).strip().lower()
    if latency_type:
        return latency_type
    return None


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
    try:
        import wandb
    except ImportError:
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a latency-sensitive control benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        checkpoint_kind=args.checkpoint_kind,
    )
    if has_eval_suites(config):
        summaries = run_eval_suites_from_config(config, progress=True)
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
    metrics = run_eval_sweep_from_config(config, progress=True)
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
