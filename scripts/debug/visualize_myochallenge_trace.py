from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.analysis.myochallenge_trace_viz import (
    build_myochallenge_timeline,
    replay_myochallenge_episode,
    write_myochallenge_timeline_png,
)
from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.envs.factory import build_env_adapter
from latency_bench.logging.action_trace_replay import load_step_records_by_episode
from latency_bench.utils.io import load_jsonl_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a MyoChallenge Table Tennis action trace and write MP4/PNG diagnostics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--viz-dir",
        type=Path,
        help="Directory for generated trace_viz MP4/PNG (default: OUTPUT_DIR/trace_viz).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    try:
        steps_by_episode = load_step_records_by_episode(
            args.output_dir,
            episode_ids=[args.episode_id],
            action_space=env.action_space,
        )
        steps = steps_by_episode[args.episode_id]
        actions = load_jsonl_records(args.output_dir / "actions.jsonl")
        metric = _metrics_by_episode(args.output_dir)[args.episode_id]
        seed = metric["metadata"]["episode_seed"]
        viz_dir = args.viz_dir or args.output_dir / "trace_viz"
        video_path = viz_dir / f"myochallenge_episode_{args.episode_id:06d}.mp4"
        timeline_path = viz_dir / f"myochallenge_episode_{args.episode_id:06d}.png"
        replay_rows, replay_summary = replay_myochallenge_episode(
            env,
            steps,
            seed=seed,
            episode_id=args.episode_id,
            video_path=video_path,
            fps=float(config["env"]["env_fps"]),
        )
        timeline = build_myochallenge_timeline(
            replay_rows,
            actions,
            frame_ms=1000.0 / float(config["env"]["env_fps"]),
        )
        write_myochallenge_timeline_png(timeline, timeline_path, episode_id=args.episode_id)
    finally:
        env.close()

    print(
        json.dumps(
            {
                "episode_id": args.episode_id,
                "video_path": str(video_path),
                "timeline_path": str(timeline_path),
                "timeline_rows": len(timeline),
                "replay": replay_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _metrics_by_episode(output_dir: Path) -> dict[int, dict]:
    return {
        int(row["episode_id"]): row
        for row in load_jsonl_records(output_dir / "episode_metrics.jsonl")
    }


if __name__ == "__main__":
    main()
