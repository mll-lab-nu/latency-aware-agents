from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.envs.factory import build_env_adapter
from latency_bench.eval.sweeps import run_eval_sweep_from_config
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from latency_bench.policy.factory import build_policy
from latency_bench.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct policy latency with realtime handoff latency.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--direct-samples", type=int, default=80)
    args = parser.parse_args()

    config = load_config(args.config)
    direct_summary = _direct_policy_baseline(config, samples=args.direct_samples)

    realtime_config = load_config(args.config)
    realtime_config["logging"]["output_dir"] = str(args.output_dir)
    realtime_config["evaluation"]["eval_episodes"] = int(args.episodes)
    metrics = run_eval_sweep_from_config(realtime_config, progress=True)

    pipeline_summary_path = args.output_dir / "realtime_pipeline_summary.json"
    pipeline_summary = read_json(pipeline_summary_path)
    result = {
        "direct_policy_baseline": direct_summary,
        "realtime": {
            "episodes": len(metrics),
            "mean_return": statistics.mean(metric.episode_return for metric in metrics),
            "mean_length": statistics.mean(metric.survival_steps for metric in metrics),
            "pipeline_summary": pipeline_summary,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "realtime_handoff_smoke_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _direct_policy_baseline(config: dict[str, Any], *, samples: int) -> dict[str, Any]:
    config["policy"]["device"] = config["latency"]["device"]
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    policy = build_policy(config, action_resolver)
    measurer = RuntimeLatencyMeasurer(
        device=config["latency"]["device"],
        sync_cuda=bool(config["latency"]["sync_cuda"]),
    )
    values: list[float] = []
    try:
        env.reset(seed=0)
        for obs_id in range(samples):
            observation = env.observe()
            _policy_output, record = measurer.run_and_measure(policy.predict, observation, obs_id=obs_id)
            values.append(float(record.latency_ms))
            env.step(action_resolver.noop_action)
    finally:
        env.close()
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


if __name__ == "__main__":
    main()
