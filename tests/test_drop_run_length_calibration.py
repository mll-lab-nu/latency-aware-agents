from __future__ import annotations

import json
from pathlib import Path

from latency_bench.analysis.drop_run_length_calibration import compare_run_dirs


# Observation pattern per episode: submitted, drop, drop, submitted, submitted, drop
# -> consecutive-drop run lengths [2, 1]; drop rate 3/6 = 0.5; 3 submitted obs.
_OBS_PATTERN = [True, False, False, True, True, False]
_SERVICE_FRAMES = [1, 1, 2]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _write_run_dir(run_dir: Path, *, episode_return: float, real_style: bool, frame_ms: float = 40.0) -> None:
    steps = [
        {
            "episode_id": 0,
            "env_step": env_step,
            "info": {"observation_submitted": submitted, "observation_dropped": not submitted},
        }
        for env_step, submitted in enumerate(_OBS_PATTERN)
    ]
    _write_jsonl(run_dir / "steps.jsonl", steps)

    actions = []
    for action_id, frames in enumerate(_SERVICE_FRAMES):
        if real_style:
            metadata = {
                "realtime_profile": {
                    "parent_to_worker_ipc_ms": 0.0,
                    "worker_recv_overhead_ms": 0.0,
                    "worker_to_parent_ipc_ms": 0.0,
                }
            }
            latency_ms = float(frames) * frame_ms
        else:
            metadata = {"worker_service_raw_frames": frames, "frame_ms": frame_ms}
            latency_ms = float(frames) * frame_ms
        actions.append(
            {
                "action_id": action_id,
                "obs_env_step": action_id,
                "ready_raw_frame": action_id + frames,
                "applied": True,
                "dropped": False,
                "latency": {"latency_ms": latency_ms},
                "metadata": metadata,
            }
        )
    _write_jsonl(run_dir / "actions.jsonl", actions)

    _write_jsonl(
        run_dir / "episode_metrics.jsonl",
        [{"episode_id": 0, "num_actions": len(_SERVICE_FRAMES), "episode_return": episode_return, "metadata": {"frame_ms": frame_ms}}],
    )


def test_calibration_is_zero_when_real_equals_sim(tmp_path: Path):
    real_dir = tmp_path / "real"
    sim_dir = tmp_path / "sim"
    _write_run_dir(real_dir, episode_return=10.0, real_style=True)
    _write_run_dir(sim_dir, episode_return=10.0, real_style=False)

    report = compare_run_dirs(real_dir, sim_dir)

    assert report["wasserstein_drop_run_length"] == 0.0
    assert report["drop_rate_gap"] == 0.0
    assert report["reward_gap"] == 0.0
    assert report["acf_l2_gap"] == 0.0


def test_calibration_reports_drop_run_length_and_reward_gap(tmp_path: Path):
    real_dir = tmp_path / "real"
    sim_dir = tmp_path / "sim"
    _write_run_dir(real_dir, episode_return=10.0, real_style=True)
    _write_run_dir(sim_dir, episode_return=7.0, real_style=False)

    report = compare_run_dirs(real_dir, sim_dir)

    assert report["drop_run_length"]["real"]["mean"] == 1.5
    assert report["drop_run_length"]["real"]["max"] == 2.0
    assert report["drop_rate"]["real"] == 0.5
    assert report["reward_gap"] == 3.0
