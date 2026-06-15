from __future__ import annotations

import json
from pathlib import Path

from latency_bench.analysis.realtime_trace_diagnostics import summarize_realtime_trace


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_realtime_trace_reports_drops_reused_actions_and_latency_frames(tmp_path: Path):
    _write_jsonl(
        tmp_path / "episode_metrics.jsonl",
        [
            {
                "episode_id": 0,
                "episode_return": 1.2,
                "game_score": 0,
                "survival_steps": 4,
                "metadata": {"episode_seed": 0, "actual_env_fps": 41.9},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "actions.jsonl",
        [
            {"action_id": 10, "latency_raw_frames": 2, "metadata": {"raw_action_id": 0}},
            {"action_id": 11, "latency_raw_frames": 3, "metadata": {"raw_action_id": 1}},
        ],
    )
    _write_jsonl(
        tmp_path / "steps.jsonl",
        [
            {
                "episode_id": 0,
                "env_step": 0,
                "active_action": {"name": "noop"},
                "reward": 0.1,
                "done": False,
                "info": {
                    "active_action_id": 10,
                    "active_action_obs_env_step": 0,
                    "observation_dropped": False,
                    "observation_submitted": True,
                },
            },
            {
                "episode_id": 0,
                "env_step": 1,
                "active_action": {"name": "noop"},
                "reward": 0.1,
                "done": False,
                "info": {
                    "active_action_id": 10,
                    "active_action_obs_env_step": 0,
                    "observation_dropped": True,
                    "observation_submitted": False,
                },
            },
            {
                "episode_id": 0,
                "env_step": 2,
                "active_action": {"name": "flap"},
                "reward": 0.1,
                "done": False,
                "info": {
                    "active_action_id": 11,
                    "active_action_obs_env_step": 1,
                    "observation_dropped": False,
                    "observation_submitted": True,
                },
            },
            {
                "episode_id": 0,
                "env_step": 3,
                "active_action": {"name": "flap"},
                "reward": -1.0,
                "done": True,
                "info": {
                    "active_action_id": 11,
                    "active_action_obs_env_step": 1,
                    "observation_dropped": False,
                    "observation_submitted": True,
                },
            },
        ],
    )

    summary = summarize_realtime_trace(tmp_path, first_steps=3)

    assert summary["episodes"][0]["survival_steps"] == 4
    assert summary["episodes"][0]["latency_raw_frames"] == {"2": 1, "3": 1}
    assert summary["episodes"][0]["observation_dropped_count"] == 1
    assert summary["episodes"][0]["observation_not_submitted_count"] == 1
    assert summary["episodes"][0]["reused_active_action_obs_count"] == 2
    assert summary["episodes"][0]["first_active_actions"] == [
        {"env_step": 0, "action": "noop", "active_action_obs_env_step": 0},
        {"env_step": 1, "action": "noop", "active_action_obs_env_step": 0},
        {"env_step": 2, "action": "flap", "active_action_obs_env_step": 1},
    ]
