from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from helpers import REPO_ROOT


RUN_PATH = (
    REPO_ROOT
    / "paper"
    / "latency-sensitive-vla"
    / "figures"
    / "4_1_cross_device_sim"
    / "fig6_cross_device"
    / "run.py"
)


def _load_fig6_module():
    spec = importlib.util.spec_from_file_location("fig6_cross_device_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_diff_ci95_pairs_by_episode_id():
    fig6 = _load_fig6_module()

    # Episode 2 exists only on the realtime side and must be excluded from pairing.
    diff_mean, half_width = fig6.paired_diff_ci95(
        {0: 10.0, 1: 20.0, 2: 99.0},
        {0: 30.0, 1: 50.0},
    )

    assert diff_mean == 25.0
    assert half_width == 1.96 * np.std([20.0, 30.0], ddof=1) / np.sqrt(2)


def test_panel_b_ci_inputs_use_realtime_and_simulated_episode_rows():
    fig6 = _load_fig6_module()
    rows = [
        {
            "row_type": "realtime_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_id": "0",
            "episode_return_env": "10.0",
        },
        {
            "row_type": "realtime_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_id": "1",
            "episode_return_env": "20.0",
        },
        {
            "row_type": "simulated_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_id": "0",
            "episode_return_env": "30.0",
        },
        {
            "row_type": "simulated_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_id": "1",
            "episode_return_env": "50.0",
        },
    ]

    realtime, simulated = fig6.panel_b_episode_returns(rows)

    assert realtime == {"RTX 3090": {0: 10.0, 1: 20.0}}
    assert simulated == {"RTX 3090": {0: 30.0, 1: 50.0}}
