from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.analysis.action_trace_replay_metrics import replay_step_records
from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.envs.factory import build_env_adapter
from latency_bench.logging.action_trace_replay import load_step_records_by_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved realtime steps without policy or video.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, action="append")
    args = parser.parse_args()

    config = load_config(args.config)
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    try:
        selected_episode_ids = args.episode_id
        steps_by_episode = load_step_records_by_episode(
            args.output_dir,
            episode_ids=selected_episode_ids,
            action_space=getattr(env, "action_space", None),
        )
        metrics = _load_metrics(args.output_dir)
        summaries = []
        for episode_id, records in steps_by_episode.items():
            env.reset(seed=metrics[episode_id]["metadata"]["episode_seed"])
            summary = replay_step_records(env, records)
            summary["episode_id"] = episode_id
            summary["recorded_survival_steps"] = metrics[episode_id]["survival_steps"]
            summary["recorded_episode_return"] = metrics[episode_id]["episode_return"]
            summaries.append(summary)
    finally:
        env.close()
    print(json.dumps({"episodes": summaries}, indent=2, sort_keys=True))


def _load_metrics(output_dir: Path) -> dict[int, dict]:
    rows = [
        json.loads(line)
        for line in (output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return {int(row["episode_id"]): row for row in rows}


if __name__ == "__main__":
    main()
