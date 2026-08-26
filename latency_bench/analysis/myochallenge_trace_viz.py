"""Replay and visualize MyoChallenge table-tennis traces.

The task wrapper records the native MyoSuite information in ``info``.  This
module keeps the visualization independent of the wrapper by accepting the
JSON representation written by :mod:`latency_bench.logging.trajectory_logger`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latency_bench.analysis.action_trace_replay_metrics import replay_step_records
from latency_bench.core.types import StepRecord
from latency_bench.logging.video import AsyncVideoWriter
from latency_bench.utils.io import to_jsonable

TOUCHING_LABELS = ("paddle", "own", "opponent", "net", "ground", "environment")


def build_myochallenge_timeline(
    step_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    frame_ms: float,
) -> list[dict[str, Any]]:
    """Extract plot-ready rows from one episode of JSONL trace records.

    ``step_rows`` must be the recorded 100 Hz steps for one episode.  Action
    records are global to a run, so their ``action_id`` is joined through the
    ``active_action_id`` field in each step's ``info``.  The dynamics use the
    native MyoSuite ``obs_dict``/``rwd_dict`` fields written by the wrapper.
    """

    actions_by_id = {int(row["action_id"]): row for row in action_rows}
    timeline: list[dict[str, Any]] = []
    for step in sorted(step_rows, key=lambda row: int(row["env_step"])):
        info = step["info"]
        action_info = step["active_action"]
        active_action_id = info["active_action_id"]
        event = None if active_action_id is None else actions_by_id[int(active_action_id)]
        observations = info["obs_dict"]
        rewards = info["rwd_dict"]
        ball_pos = _vector(observations["ball_pos"])
        ball_vel = _vector(observations["ball_vel"])
        paddle_pos = _vector(observations["paddle_pos"])
        paddle_vel = _vector(observations["paddle_vel"])
        contact_values = _vector(observations["touching_info"])
        action_values = _vector(action_info["value"])
        muscle_l2, muscle_mean_abs = _action_magnitudes(action_values[:273])
        ball_paddle_distance = float(np.linalg.norm(np.asarray(ball_pos) - np.asarray(paddle_pos)))

        timeline.append(
            {
                "env_step": int(step["env_step"]),
                "time_ms": float(step["sim_time_ms"]),
                "time_s": float(step["sim_time_ms"]) / 1000.0,
                "reward": float(step["reward"]),
                "action_id": None if active_action_id is None else int(active_action_id),
                "action_name": action_info["name"],
                "issued_time_ms": _event_time(event, "obs_env_step", frame_ms),
                "ready_time_ms": _number(event, "ready_time_ms"),
                "applied_time_ms": _event_time(event, "apply_raw_frame", frame_ms),
                "ball_pos": ball_pos,
                "ball_vel": ball_vel,
                "paddle_pos": paddle_pos,
                "paddle_vel": paddle_vel,
                "ball_paddle_distance": ball_paddle_distance,
                "pelvis_x": action_values[273],
                "pelvis_y": action_values[274],
                "contact": bool(np.any(np.asarray(contact_values))),
                "contact_count": int(np.count_nonzero(np.asarray(contact_values))),
                **{
                    f"touching_{label}": bool(contact_values[index])
                    for index, label in enumerate(TOUCHING_LABELS)
                },
                "solved": bool(np.asarray(rewards["solved"]).reshape(-1)[0]),
                "muscle_l2": muscle_l2,
                "muscle_mean_abs": muscle_mean_abs,
            }
        )
    return timeline


def _action_magnitudes(values: Any) -> tuple[float, float]:
    """Return L2 and mean-absolute magnitudes for the muscle action slice."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    return float(np.sqrt(np.sum(vector * vector))), float(np.mean(np.abs(vector)))


def write_myochallenge_timeline_png(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    episode_id: int,
) -> None:
    """Write a six-panel PNG for one replayed MyoChallenge episode."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray([row["time_s"] for row in rows], dtype=float)
    figure, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    figure.suptitle(f"MyoChallenge Table Tennis P2 - episode {episode_id}")

    position_axis = axes[0, 0]
    _plot_components(position_axis, time_s, rows, "ball_pos", "ball", "tab:blue")
    _plot_components(position_axis, time_s, rows, "paddle_pos", "paddle", "tab:orange")
    position_axis.plot(
        time_s,
        _series(rows, "ball_paddle_distance"),
        "--",
        label="ball-paddle distance",
        color="tab:green",
    )
    position_axis.set_title("Ball / paddle position")
    position_axis.set_ylabel("position (m)")
    position_axis.legend(loc="upper right", ncol=2, fontsize="small")

    speed_axis = axes[0, 1]
    _plot_norm(speed_axis, time_s, rows, "ball_vel", "ball speed", "tab:blue")
    _plot_norm(speed_axis, time_s, rows, "paddle_vel", "paddle speed", "tab:orange")
    speed_axis.set_title("Ball / paddle dynamics")
    speed_axis.set_ylabel("speed (m/s)")
    speed_axis.legend(loc="upper right", fontsize="small")

    timing_axis = axes[1, 0]
    timing_axis.plot(time_s, _series(rows, "issued_time_ms", scale=0.001), "o-", ms=2, label="issued")
    timing_axis.plot(time_s, _series(rows, "ready_time_ms", scale=0.001), "o-", ms=2, label="ready")
    timing_axis.plot(time_s, _series(rows, "applied_time_ms", scale=0.001), "o-", ms=2, label="applied")
    timing_axis.set_title("Action timing")
    timing_axis.set_ylabel("event time (s)")
    timing_axis.legend(loc="upper left", fontsize="small")

    event_axis = axes[1, 1]
    event_axis.plot(time_s, _series(rows, "reward"), label="reward", color="tab:green")
    event_axis.set_title("Reward / contact / solved")
    event_axis.set_ylabel("reward")
    event_axis.legend(loc="upper left", fontsize="small")
    event_overlay = event_axis.twinx()
    event_overlay.step(time_s, _series(rows, "contact"), where="post", color="tab:red", label="contact")
    event_overlay.step(time_s, _series(rows, "solved"), where="post", color="black", label="solved")
    for label in TOUCHING_LABELS:
        event_overlay.step(
            time_s,
            _series(rows, f"touching_{label}"),
            where="post",
            alpha=0.45,
            label=label,
        )
    event_overlay.set_ylim(-0.05, 1.05)
    event_overlay.set_ylabel("event")
    event_overlay.legend(loc="upper right", fontsize="small")

    muscle_axis = axes[2, 0]
    muscle_axis.plot(time_s, _series(rows, "muscle_l2"), label="L2", color="tab:purple")
    muscle_axis.plot(time_s, _series(rows, "muscle_mean_abs"), label="mean abs", color="tab:brown")
    muscle_axis.set_title("Muscle action magnitude")
    muscle_axis.set_ylabel("magnitude")
    muscle_axis.set_xlabel("simulation time (s)")
    muscle_axis.legend(loc="upper left", fontsize="small")

    pelvis_axis = axes[2, 1]
    pelvis_axis.plot(time_s, _series(rows, "pelvis_x"), label="pelvis x", color="tab:cyan")
    pelvis_axis.plot(time_s, _series(rows, "pelvis_y"), label="pelvis y", color="tab:pink")
    pelvis_axis.set_title("Pelvis actuator commands")
    pelvis_axis.set_ylabel("normalized control")
    pelvis_axis.set_xlabel("simulation time (s)")
    pelvis_axis.legend(loc="upper left", fontsize="small")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def replay_myochallenge_episode(
    env: Any,
    step_records: list[StepRecord],
    *,
    seed: int | None,
    episode_id: int,
    video_path: str | Path,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay one episode, verify parity, capture MP4 frames, and extract rows."""

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    env.reset(seed=seed)
    writer = AsyncVideoWriter(video_path, fps=fps)
    replay_rows: list[dict[str, Any]] = []

    class _CaptureEnv:
        def step(self, action: Any) -> Any:
            result = env.step(action)
            record = step_records[len(replay_rows)]
            payload = to_jsonable(record)
            payload["info"] = {**to_jsonable(record.info), **to_jsonable(result.info)}
            replay_rows.append(payload)
            writer.write(np.asarray(env.render_game_frame()))
            return result

    replay_env = _CaptureEnv()
    try:
        writer.write(np.asarray(env.render_game_frame()))
        replay_summary = replay_step_records(replay_env, step_records)
    finally:
        writer.close()

    if replay_summary["reward_mismatch_count"]:
        raise ValueError(
            f"action trace replay reward mismatch at episode {episode_id}: "
            f"{replay_summary['reward_mismatch_count']} steps"
        )
    if replay_summary["done_mismatch_count"]:
        raise ValueError(
            f"action trace replay done mismatch at episode {episode_id}: "
            f"{replay_summary['done_mismatch_count']} steps"
        )

    return replay_rows, replay_summary


def _vector(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _event_time(event: Mapping[str, Any] | None, key: str, frame_ms: float) -> float | None:
    value = _number(event, key)
    return None if value is None else value * frame_ms


def _number(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if mapping is None or key not in mapping or mapping[key] is None:
        return None
    return float(mapping[key])


def _series(rows: Sequence[Mapping[str, Any]], key: str, *, scale: float = 1.0) -> np.ndarray:
    return np.asarray(
        [math.nan if row[key] is None else float(row[key]) * scale for row in rows],
        dtype=float,
    )


def _plot_components(
    axis: Any,
    time_s: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    key: str,
    prefix: str,
    color: str,
) -> None:
    components = [
        np.asarray(
            [math.nan if row[key] is None else row[key][index] for row in rows],
            dtype=float,
        )
        for index in range(3)
    ]
    for index, values in enumerate(components):
        axis.plot(time_s, values, label=f"{prefix} {('xyz')[index]}", color=color, alpha=0.45 + index * 0.2)


def _plot_norm(
    axis: Any,
    time_s: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    key: str,
    label: str,
    color: str,
) -> None:
    values = [
        math.nan if row[key] is None else float(np.linalg.norm(np.asarray(row[key], dtype=float)))
        for row in rows
    ]
    axis.plot(time_s, values, label=label, color=color)
