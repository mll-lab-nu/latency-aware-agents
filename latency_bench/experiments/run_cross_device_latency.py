from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.experiments.run_simulated_latency import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay profiled latency trace in simulated control.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cross_device = config.get("cross_device", {})
    metrics = run_from_config(config, extra_metadata={"cross_device": cross_device})
    for episode_metrics in metrics:
        print(
            f"episode={episode_metrics.episode_id} "
            f"return={episode_metrics.episode_return:.3f} "
            f"steps={episode_metrics.survival_steps} "
            f"profile_device={cross_device.get('profile_device')}"
        )


if __name__ == "__main__":
    main()
