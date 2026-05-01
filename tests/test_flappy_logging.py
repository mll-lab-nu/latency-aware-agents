from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from training.flappy_ppo.cli import build_parser
from training.flappy_ppo.logging import (
    JsonlEventLogger,
    format_update,
)


def test_jsonl_event_logger_writes_parseable_scalar_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlEventLogger(str(path), run_id="run-1")

    logger.emit(
        {
            "event": "train_start",
            "global_step": np.int64(0),
            "update": None,
            "torch_scalar": torch.tensor(2.5),
            "nonfinite": float("nan"),
        }
    )
    logger.emit(
        {
            "event": "update",
            "global_step": 8,
            "update": 1,
            "nested": {"value": np.float32(1.25)},
        }
    )
    logger.close()

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["train_start", "update"]
    assert events[0]["run_id"] == "run-1"
    assert events[0]["torch_scalar"] == 2.5
    assert events[0]["nonfinite"] is None
    assert events[1]["nested"]["value"] == np.float32(1.25).item()


def test_jsonl_event_logger_empty_path_is_noop():
    logger = JsonlEventLogger("", run_id="run-1")
    logger.emit({"event": "train_start", "global_step": 0, "update": None})
    logger.close()


def test_update_console_line_stays_compact():
    line = format_update(
        {
            "event": "update",
            "run_id": "20260428_215419_cpu023_12345",
            "global_step": 999424,
            "update": 305,
            "num_updates": 305,
            "sps": 3678,
            "episode_count": 8735,
            "ep_rew_mean": 1207.164,
            "ep_len_mean": 9760.2,
            "loss": 0.245039,
            "approx_kl": 0.002783,
            "clipfrac": 0.078,
            "learning_rate": 0.000245131,
        }
    )

    assert line.startswith("[update] ")
    assert len(line) < 160


def test_cli_accepts_run_id_and_metrics_jsonl_flags():
    args = build_parser().parse_args(
        ["--run-id", "run-1", "--metrics-jsonl", "logs/run-1/metrics.jsonl"]
    )

    assert args.run_id == "run-1"
    assert args.metrics_jsonl == "logs/run-1/metrics.jsonl"


def test_target_training_scripts_do_not_append_to_logs():
    scripts = [
        Path("scripts/bash_scripts/train_flappy_fs0.sh"),
        Path("scripts/train_torch.sh"),
        Path("scripts/train_torch_fs1.sh"),
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "tee -a" not in text
        assert 'tee "${LOG_PATH}"' in text
        assert "--metrics-jsonl" in text
        assert "WANDB_CONSOLE" in text
