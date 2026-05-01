from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.core.types import EpisodeMetrics
from latency_bench.executors.factory import build_executor


def apply_runtime_overrides(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        config["policy"]["checkpoint_path"] = str(checkpoint_path)
    if run_name is not None:
        config["experiment"]["name"] = str(run_name)
    if output_dir is not None:
        config["logging"]["output_dir"] = str(output_dir)
    return config


def run_from_config(config: dict[str, Any], extra_metadata: dict[str, Any] | None = None) -> list[EpisodeMetrics]:
    executor = build_executor(config)
    experiment_cfg = config["experiment"]
    policy_cfg = config["policy"]
    logging_cfg = config["logging"]
    seed = int(experiment_cfg.get("seed", 0))
    num_episodes = int(experiment_cfg.get("num_episodes", 1))
    max_env_steps = int(experiment_cfg.get("max_env_steps", 10000))
    metrics = []
    try:
        for episode_id in range(num_episodes):
            episode_metrics = executor.run_episode(
                episode_id=episode_id,
                seed=seed + episode_id,
                max_env_steps=max_env_steps,
            )
            episode_metrics.metadata["config_name"] = experiment_cfg.get("name")
            episode_metrics.metadata["run_name"] = experiment_cfg.get("name")
            if "checkpoint_path" in policy_cfg:
                episode_metrics.metadata["checkpoint_path"] = str(policy_cfg["checkpoint_path"])
            episode_metrics.metadata["output_dir"] = str(logging_cfg["output_dir"])
            if extra_metadata:
                episode_metrics.metadata.update(extra_metadata)
            metrics.append(episode_metrics)
        if executor.logger is not None:
            executor.logger.flush()
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
        all_metrics.extend(
            run_from_config(
                sweep_config,
                extra_metadata={
                    "train_latency_steps": train_latency_steps,
                    "eval_latency_steps": eval_latency_steps,
                    "configured_latency_ms": configured_latency_ms,
                },
            )
        )
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a latency-sensitive control benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
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


if __name__ == "__main__":
    main()
