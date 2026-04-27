from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.core.types import EpisodeMetrics
from latency_bench.executors.factory import build_executor


def run_from_config(config: dict[str, Any], extra_metadata: dict[str, Any] | None = None) -> list[EpisodeMetrics]:
    executor = build_executor(config)
    experiment_cfg = config["experiment"]
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
            if extra_metadata:
                episode_metrics.metadata.update(extra_metadata)
            metrics.append(episode_metrics)
        if executor.logger is not None:
            executor.logger.flush()
    finally:
        executor.env.close()
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a latency-sensitive control benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metrics = run_from_config(config)
    for episode_metrics in metrics:
        print(
            f"episode={episode_metrics.episode_id} "
            f"return={episode_metrics.episode_return:.3f} "
            f"steps={episode_metrics.survival_steps} "
            f"mean_latency_ms={episode_metrics.mean_latency_ms}"
        )


if __name__ == "__main__":
    main()
