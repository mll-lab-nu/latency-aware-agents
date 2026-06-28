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


def test_ci95_errors_are_relative_to_panel_b_center_point():
    fig6 = _load_fig6_module()

    lower, upper = fig6.ci95_errors([10.0, 20.0], center=15.0)

    expected = 1.96 * np.std([10.0, 20.0], ddof=1) / np.sqrt(2)
    assert lower == expected
    assert upper == expected


def test_panel_b_ci_inputs_use_realtime_and_simulated_episode_rows():
    fig6 = _load_fig6_module()
    rows = [
        {
            "row_type": "realtime_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_return_env": "10.0",
        },
        {
            "row_type": "realtime_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_return_env": "20.0",
        },
        {
            "row_type": "simulated_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_return_env": "30.0",
        },
        {
            "row_type": "simulated_episode_return",
            "policy": "openvla-flappy",
            "env": "Flappy",
            "hardware": "RTX 3090",
            "episode_return_env": "50.0",
        },
    ]

    realtime, simulated = fig6.panel_b_episode_returns(rows)

    assert realtime == {"RTX 3090": [10.0, 20.0]}
    assert simulated == {"RTX 3090": [30.0, 50.0]}
