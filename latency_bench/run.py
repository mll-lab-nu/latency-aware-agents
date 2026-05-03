from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.core.types import EpisodeMetrics
from latency_bench.executors.factory import build_executor
from latency_bench.logging.summary import build_queue_eval_summary, write_queue_eval_summary


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
        from training.sample_factory_common.config_adapter import (
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
) -> list[EpisodeMetrics]:
    experiment_cfg = config["experiment"]
    policy_cfg = config["policy"]
    logging_cfg = config["logging"]
    seed = int(experiment_cfg.get("seed", 0))
    num_episodes = int(experiment_cfg.get("num_episodes", 1))
    eval_max_steps = _eval_max_steps(config)
    if _is_sample_factory_policy(config):
        from training.sample_factory_common.common import seed_everything

        seed_everything(seed)

    executor = build_executor(config)
    metrics = []
    try:
        for episode_id in range(num_episodes):
            episode_metrics = executor.run_episode(
                episode_id=episode_id,
                seed=_episode_seed(config, episode_id),
                eval_max_steps=eval_max_steps,
            )
            episode_metrics.metadata["config_name"] = experiment_cfg.get("name")
            episode_metrics.metadata["run_name"] = experiment_cfg.get("name")
            if "checkpoint_path" in policy_cfg:
                episode_metrics.metadata["checkpoint_path"] = str(policy_cfg["checkpoint_path"])
            if "checkpoint_kind" in policy_cfg:
                episode_metrics.metadata["checkpoint_kind"] = str(policy_cfg["checkpoint_kind"])
            episode_metrics.metadata["output_dir"] = str(logging_cfg["output_dir"])
            if extra_metadata:
                episode_metrics.metadata.update(extra_metadata)
            metrics.append(episode_metrics)
        if executor.logger is not None:
            executor.logger.flush()
        if write_summary:
            _write_non_sweep_summary(config, metrics)
    finally:
        executor.env.close()
    return metrics


def run_eval_sweep_from_config(config: dict[str, Any]) -> list[EpisodeMetrics]:
    eval_latency_values = _eval_latency_values(config)
    if eval_latency_values is None:
        return run_from_config(config)

    base_output_dir = Path(config["logging"]["output_dir"])
    train_latency_steps = _train_latency_steps(config)
    all_metrics: list[EpisodeMetrics] = []
    for eval_latency_steps in eval_latency_values:
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
        latency_metrics = run_from_config(
            sweep_config,
            extra_metadata=summary_metadata,
            write_summary=False,
        )
        write_queue_eval_summary(
            base_output_dir,
            latency_metrics,
            experiment_name=str(config["experiment"].get("name", "latency_bench")),
            seed=int(config["experiment"].get("seed", 0)),
            latency=eval_latency_steps,
            metadata=summary_metadata,
        )
        all_metrics.extend(latency_metrics)
    return all_metrics


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
    train_values = config.get("latency", {}).get("train_values")
    if not train_values:
        return None
    return int(train_values[0])


def _latency_steps_to_ms(config: dict[str, Any], latency_steps: int) -> float:
    fps_env = float(config["env"]["fps_env"])
    if fps_env <= 0:
        raise ValueError("env.fps_env must be positive")
    return int(latency_steps) * 1000.0 / fps_env


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
    latency_steps = _infer_latency_steps(config)
    if latency_steps is None:
        return None
    return write_queue_eval_summary(
        config["logging"]["output_dir"],
        metrics,
        experiment_name=str(config["experiment"].get("name", "latency_bench")),
        seed=int(config["experiment"].get("seed", 0)),
        latency=latency_steps,
    )


def _infer_latency_steps(config: dict[str, Any]) -> int | None:
    latency_cfg = config.get("latency", {})
    latency_type = str(latency_cfg.get("type", "")).strip().lower()
    if latency_type == "zero":
        return 0

    train_latency_steps = _single_latency_value(latency_cfg.get("train_values"))
    if train_latency_steps is not None:
        return train_latency_steps

    if "fixed_latency_ms" not in latency_cfg:
        return None
    latency_steps = (
        float(latency_cfg["fixed_latency_ms"])
        * float(config["env"]["fps_env"])
        / 1000.0
    )
    rounded_steps = round(latency_steps)
    if abs(latency_steps - rounded_steps) > 1e-9:
        return None
    return int(rounded_steps)


def _single_latency_value(values: Any) -> int | None:
    if values is None or isinstance(values, (str, bytes)):
        return None
    if not isinstance(values, Sequence):
        return int(values)
    if len(values) != 1:
        return None
    return int(values[0])


def _summary_payloads_for_printing(
    config: dict[str, Any],
    metrics: list[EpisodeMetrics],
) -> list[dict[str, Any]]:
    experiment_name = str(config["experiment"].get("name", "latency_bench"))
    seed = int(config["experiment"].get("seed", 0))
    eval_latency_values = _eval_latency_values(config)
    if eval_latency_values is None:
        latency_steps = _infer_latency_steps(config)
        if latency_steps is None:
            return []
        return [
            build_queue_eval_summary(
                metrics,
                experiment_name=experiment_name,
                seed=seed,
                latency=latency_steps,
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
    metrics = run_eval_sweep_from_config(config)
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
    for summary in _summary_payloads_for_printing(config, metrics):
        print(
            f"summary latency={int(summary['latency'])} "
            f"episodes={len(summary['returns'])} "
            f"mean_return={float(summary['mean_return']):.3f} "
            f"std_return={float(summary['std_return']):.3f} "
            f"mean_length={float(summary['mean_length']):.1f}"
        )


if __name__ == "__main__":
    main()
