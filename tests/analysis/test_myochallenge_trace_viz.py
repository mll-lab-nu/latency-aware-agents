from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.analysis.myochallenge_trace_viz import build_myochallenge_timeline, write_myochallenge_timeline_png


def _step(env_step: int, *, action_id: int | None, solved: bool) -> dict:
    action = [0.1, -0.2, 0.3] + [0.0] * 270 + [0.4, -0.5]
    return {
        "env_step": env_step,
        "sim_time_ms": float(env_step * 10),
        "reward": float(env_step),
        "active_action": {"name": "muscle", "value": action},
        "info": {
            "active_action_id": action_id,
            "obs_dict": {
                "ball_pos": [env_step, 1.0, 2.0],
                "ball_vel": [1.0, 2.0, 2.0],
                "paddle_pos": [3.0, env_step, 5.0],
                "paddle_vel": [0.0, 0.0, 4.0],
                "pelvis_pos": [6.0, 7.0, 8.0],
                "act": [0.1, 0.2, 0.3],
                "touching_info": [0, env_step == 1, 0, 0, 0, 0],
            },
            "rwd_dict": {"solved": [[solved]]},
        },
    }


def test_build_myochallenge_timeline_extracts_official_info_and_timings():
    rows = build_myochallenge_timeline(
        [_step(0, action_id=4, solved=False), _step(1, action_id=4, solved=True)],
        [
            {
                "action_id": 4,
                "obs_env_step": 0,
                "ready_time_ms": 20.0,
                "apply_raw_frame": 3,
            }
        ],
        frame_ms=10.0,
    )

    assert rows[0]["issued_time_ms"] == 0.0
    assert rows[0]["ready_time_ms"] == 20.0
    assert rows[0]["applied_time_ms"] == 30.0
    assert rows[1]["contact"] is True
    assert rows[1]["touching_own"] is True
    assert rows[1]["ball_paddle_distance"] == pytest.approx(3.6055512755)
    assert rows[1]["solved"] is True
    assert rows[1]["muscle_l2"] == pytest.approx(0.3741657387)
    assert rows[1]["pelvis_x"] == 0.4
    assert rows[1]["pelvis_y"] == -0.5


def test_timeline_png(tmp_path: Path):
    rows = build_myochallenge_timeline(
        [_step(0, action_id=None, solved=False)],
        [],
        frame_ms=10.0,
    )
    output_path = tmp_path / "timeline.png"
    write_myochallenge_timeline_png(rows, output_path, episode_id=0)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
