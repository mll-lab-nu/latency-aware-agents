from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_realtime_trace(output_dir: str | Path, *, first_steps: int = 60) -> dict[str, Any]:
    output_path = Path(output_dir)
    metrics = _read_jsonl(output_path / "episode_metrics.jsonl")
    actions = _read_jsonl(output_path / "actions.jsonl")
    steps = _read_jsonl(output_path / "steps.jsonl")
    metrics_by_episode = {int(row["episode_id"]): row for row in metrics}
    actions_by_id = {int(row["action_id"]): row for row in actions}
    steps_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        steps_by_episode[int(row["episode_id"])].append(row)
    episodes = [
        _summarize_episode(
            episode_id=episode_id,
            steps=sorted(episode_steps, key=lambda row: int(row["env_step"])),
            metric=metrics_by_episode[episode_id],
            actions_by_id=actions_by_id,
            first_steps=first_steps,
        )
        for episode_id, episode_steps in sorted(steps_by_episode.items())
    ]
    return {
        "output_dir": str(output_path),
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def _summarize_episode(
    *,
    episode_id: int,
    steps: list[dict[str, Any]],
    metric: dict[str, Any],
    actions_by_id: dict[int, dict[str, Any]],
    first_steps: int,
) -> dict[str, Any]:
    active_action_ids = {
        int(step["info"]["active_action_id"])
        for step in steps
        if step["info"]["active_action_id"] is not None
    }
    action_events = [
        actions_by_id[action_id]
        for action_id in sorted(active_action_ids)
        if action_id in actions_by_id
    ]
    action_names = Counter(str(step["active_action"]["name"]) for step in steps)
    latency_raw_frames = Counter(str(event["latency_raw_frames"]) for event in action_events)
    active_obs_steps = [
        step["info"]["active_action_obs_env_step"]
        for step in steps
        if step["info"]["active_action_obs_env_step"] is not None
    ]
    reused_active_action_obs_count = sum(
        1
        for prev, current in zip(active_obs_steps, active_obs_steps[1:])
        if current == prev
    )
    action_age_frames = [
        int(step["env_step"]) - int(step["info"]["active_action_obs_env_step"])
        for step in steps
        if step["info"]["active_action_obs_env_step"] is not None
    ]
    return {
        "episode_id": episode_id,
        "episode_seed": metric["metadata"]["episode_seed"],
        "survival_steps": metric["survival_steps"],
        "episode_return": metric["episode_return"],
        "game_score": metric["game_score"],
        "actual_env_fps": metric["metadata"]["actual_env_fps"],
        "active_actions": dict(sorted(action_names.items())),
        "latency_raw_frames": dict(sorted(latency_raw_frames.items())),
        "observation_dropped_count": sum(1 for step in steps if step["info"]["observation_dropped"]),
        "observation_not_submitted_count": sum(1 for step in steps if not step["info"]["observation_submitted"]),
        "reused_active_action_obs_count": reused_active_action_obs_count,
        "max_action_age_frames": max(action_age_frames) if action_age_frames else 0,
        "first_active_actions": [
            {
                "env_step": step["env_step"],
                "action": step["active_action"]["name"],
                "active_action_obs_env_step": step["info"]["active_action_obs_env_step"],
            }
            for step in steps[:first_steps]
        ],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
