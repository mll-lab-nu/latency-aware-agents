from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SEEDS = tuple(range(10, 20))
FIG_ROOT = REPO_ROOT / "paper" / "latency-sensitive-vla" / "figures" / "3_4_factor_sweeps"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_context_window_fetch_specs_use_official_seed_grid():
    fetch = _load_module("factor_sweep_context_window_fetch", FIG_ROOT / "fig_context_window" / "fetch_wandb.py")

    specs = fetch.build_query_specs()

    assert len(specs) == 70
    assert {spec.expected_seeds for spec in specs} == {OFFICIAL_SEEDS}
    assert {spec.x_label for spec in specs if spec.env_id == "flappy"} == {"4", "5", "6", "7", "8"}
    assert {spec.x_label for spec in specs if spec.env_id == "demon-attack"} == {"4", "5", "6", "7", "8"}
    assert {spec.x_label for spec in specs if spec.env_id == "deadly-corridor"} == {"4", "6", "8", "10"}
    assert "flappy-fs5-fixed_l1" in {spec.group_name for spec in specs}
    assert "deadly-corridor-fs10-fixed_l2" in {spec.group_name for spec in specs}
    assert "flappy-frame_stack" not in {spec.group_name for spec in specs}


def test_observation_stride_fetch_specs_use_official_seed_grid_and_stride_groups():
    fetch = _load_module(
        "factor_sweep_observation_stride_fetch",
        FIG_ROOT / "fig_observation_stride" / "fetch_wandb.py",
    )

    specs = fetch.build_query_specs()

    assert len(specs) == 75
    assert {spec.expected_seeds for spec in specs} == {OFFICIAL_SEEDS}
    assert {spec.x_label for spec in specs if spec.env_id == "flappy"} == {"1", "2", "3", "4", "5"}
    assert {spec.x_label for spec in specs if spec.env_id == "demon-attack"} == {"2", "3", "4", "5", "6"}
    assert {spec.x_label for spec in specs if spec.env_id == "deadly-corridor"} == {"2", "3", "4", "5", "6"}
    group_names = {spec.group_name for spec in specs}
    assert "flappy-stride5-fixed_l2" in group_names
    assert "demon-attack-stride2-fixed_l2" in group_names
    assert "deadly-corridor-stride6-uniform_u2_6" in group_names
    assert all("-obs" not in group_name for group_name in group_names)
