from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from helpers import REPO_ROOT


OFFICIAL_SEEDS = tuple(range(10, 15))
FIG_ROOT = REPO_ROOT / "paper" / "latency-sensitive-vla" / "figures" / "3_4_factor_sweeps"
JITTER_GRID_ROOT = (
    REPO_ROOT
    / "paper"
    / "latency-sensitive-vla"
    / "figures"
    / "3_3_jitter_destructive"
    / "fig_train_eval_latency_grid"
)


def _load_module(name: str, path: Path):
    # The paper/ figure tree is gitignored (local-only), so these fetch scripts
    # are absent on clean checkouts / CI. Skip rather than fail when missing.
    if not path.exists():
        pytest.skip(f"paper figure script not present: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_context_window_fetch_specs_use_official_seed_grid():
    fetch = _load_module("factor_sweep_context_window_fetch", FIG_ROOT / "fig_context_window" / "capped" / "fetch_wandb.py")

    specs = fetch.build_query_specs()

    assert len(specs) == 38
    assert {spec.expected_seeds for spec in specs} == {OFFICIAL_SEEDS}
    assert {spec.x_label for spec in specs if spec.env_id == "flappy"} == {"4", "5", "6", "7", "8"}
    assert {spec.x_label for spec in specs if spec.env_id == "demon-attack"} == {"4", "5", "6", "7", "8"}
    assert {spec.x_label for spec in specs if spec.env_id == "deadly-corridor"} == {"1", "2", "4", "6", "8", "10", "12", "14", "16"}
    assert "flappy-fs5-fixed_l2" in {spec.group_name for spec in specs}
    assert "deadly-corridor-fs10-fixed_l4" in {spec.group_name for spec in specs}
    assert "flappy-frame_stack" not in {spec.group_name for spec in specs}


def test_observation_stride_fetch_specs_use_official_seed_grid_and_stride_groups():
    fetch = _load_module(
        "factor_sweep_observation_stride_fetch",
        FIG_ROOT / "fig_observation_stride" / "capped" / "fetch_wandb.py",
    )

    specs = fetch.build_query_specs()

    assert len(specs) == 32
    assert {spec.expected_seeds for spec in specs} == {OFFICIAL_SEEDS}
    assert {spec.x_label for spec in specs if spec.env_id == "flappy"} == {"1", "2", "3", "4", "5"}
    assert {spec.x_label for spec in specs if spec.env_id == "demon-attack"} == {"2", "3", "4", "5", "6"}
    assert {spec.x_label for spec in specs if spec.env_id == "deadly-corridor"} == {"1", "2", "3", "4", "5", "6"}
    group_names = {spec.group_name for spec in specs}
    assert "flappy-stride5-fixed_l2" in group_names
    assert "demon-attack-stride2-fixed_l4" in group_names
    assert "deadly-corridor-stride6-uniform_u2_6" in group_names
    assert all("-obs" not in group_name for group_name in group_names)


class _FakeRun:
    def __init__(self, run_id: str, seed: int, created_at: int, summary: dict[str, float]):
        self.id = run_id
        self.tags = [f"seed{seed}"]
        self.created_at = created_at
        self.summary = summary


def test_jitter_grid_fetch_uses_new_flappy_seeds_and_existing_vizdoom_seeds():
    fetch = _load_module("jitter_grid_fetch", JITTER_GRID_ROOT / "fetch_wandb.py")

    flappy = [spec for spec in fetch.ENV_SPECS if spec.env_id == "flappy"][0]
    demon = [spec for spec in fetch.ENV_SPECS if spec.env_id == "demon-attack"][0]
    metric_key = flappy.eval_settings[0].metric_key

    assert flappy.expected_seeds == (911, 922, 933)
    assert demon.expected_seeds == (0, 42, 114514)

    runs = fetch.latest_run_per_seed(
        [
            _FakeRun("old-seed0", 0, 9, {metric_key: 10.0}),
            _FakeRun("old-seed1", 1, 10, {metric_key: 20.0}),
            _FakeRun("new-seed911-old", 911, 11, {metric_key: 30.0}),
            _FakeRun("new-seed911", 911, 12, {metric_key: 40.0}),
            _FakeRun("new-seed922", 922, 13, {metric_key: 50.0}),
            _FakeRun("new-seed933", 933, 14, {metric_key: 60.0}),
        ],
        [metric_key],
        flappy.expected_seeds,
    )

    assert [run.id for run in runs] == ["new-seed911", "new-seed922", "new-seed933"]
