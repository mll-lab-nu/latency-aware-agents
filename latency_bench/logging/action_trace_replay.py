from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, EpisodeMetrics, StepRecord
from latency_bench.envs.factory import build_env_adapter
from latency_bench.logging.video import EpisodeVideoRecorder, VideoSelection


def load_step_records_by_episode(
    output_dir: str | Path,
    *,
    episode_ids: Iterable[int] | None = None,
    action_space: Any | None = None,
) -> dict[int, list[StepRecord]]:
    path = Path(output_dir) / "steps.jsonl"
    selected_episode_ids = None if episode_ids is None else set(episode_ids)
    grouped: dict[int, list[StepRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            episode_id = payload["episode_id"]
            if selected_episode_ids is not None and episode_id not in selected_episode_ids:
                continue
            record = _step_record_from_json(payload, action_space=action_space)
            grouped.setdefault(record.episode_id, []).append(record)
    return {
        episode_id: sorted(records, key=lambda record: record.env_step)
        for episode_id, records in sorted(grouped.items())
    }


def record_videos_from_action_trace(
    config: dict[str, Any],
    *,
    selections: list[VideoSelection],
    metrics: list[EpisodeMetrics],
) -> None:
    output_dir = Path(config["logging"]["output_dir"])
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    try:
        selection_episode_ids = [selection.episode_id for selection in selections]
        steps_by_episode = load_step_records_by_episode(
            output_dir,
            episode_ids=selection_episode_ids,
            action_space=getattr(env, "action_space", None),
        )
        metrics_by_episode = {metric.episode_id: metric for metric in metrics}
        recorder = EpisodeVideoRecorder(
            output_dir / "videos",
            fps=config["env"]["env_fps"],
        )
        for selection in selections:
            episode_id = selection.episode_id
            metric = metrics_by_episode[episode_id]
            _record_episode_from_steps(
                env=env,
                recorder=recorder,
                episode_id=episode_id,
                seed=metric.metadata["episode_seed"],
                step_records=steps_by_episode[episode_id],
                metadata={
                    "bin_index": selection.bin_index,
                    "episode_return": selection.episode_return,
                },
            )
    finally:
        env.close()


def _record_episode_from_steps(
    *,
    env: Any,
    recorder: EpisodeVideoRecorder,
    episode_id: int,
    seed: int | None,
    step_records: list[StepRecord],
    metadata: dict[str, Any],
) -> None:
    env.reset(seed=seed)
    recorder.start_episode(episode_id=episode_id, metadata=metadata)
    recorder.record_frame(env.render_game_frame())
    try:
        for record in step_records:
            result = env.step(record.active_action)
            replay_done = result.done or result.truncated
            if not math.isclose(
                float(result.reward),
                float(record.reward),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"action trace replay reward mismatch at episode {episode_id} "
                    f"env_step {record.env_step}: expected {record.reward}, got {result.reward}"
                )
            if replay_done != record.done:
                raise ValueError(
                    f"action trace replay done mismatch at episode {episode_id} "
                    f"env_step {record.env_step}: expected {record.done}, got {replay_done}"
                )
            if replay_done:
                break
            recorder.record_frame(env.render_game_frame())
    except Exception:
        recorder.abort_episode()
        raise
    recorder.close_episode({})


def _step_record_from_json(payload: dict[str, Any], *, action_space: Any | None = None) -> StepRecord:
    return StepRecord(
        episode_id=payload["episode_id"],
        env_step=payload["env_step"],
        sim_time_ms=payload["sim_time_ms"],
        active_action=_action_from_json(payload["active_action"], action_space=action_space),
        reward=payload["reward"],
        done=payload["done"],
        info=payload["info"],
    )


def _action_from_json(payload: dict[str, Any], *, action_space: Any | None = None) -> Action:
    return Action(
        value=_action_value_from_json(payload["value"], action_space=action_space),
        name=payload["name"],
        is_noop=payload["is_noop"],
        is_oneshot=payload["is_oneshot"],
    )


def _action_value_from_json(value: Any, *, action_space: Any | None) -> Any:
    from gymnasium.spaces import MultiDiscrete, Tuple

    if isinstance(action_space, (Tuple, MultiDiscrete)):
        return tuple(value)
    return value
