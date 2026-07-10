from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from latency_bench.cross_device import configs


def test_generated_sim_config_separates_eval_and_latency_seeds(tmp_path, monkeypatch):
    base_config = {
        "experiment": {"seed": 0},
        "env": {"env_fps": 16, "obs_fps": 16},
        "evaluation": {},
        "executor": {},
        "latency": {},
        "logging": {},
    }
    monkeypatch.setattr(configs, "_row_base_config", lambda row: deepcopy(base_config))
    monkeypatch.setattr(
        configs,
        "load_profile_manifest",
        lambda path: {"inference_worker_count": 2, "latency_method": "temporal"},
    )
    monkeypatch.setattr(configs, "_patch_env_from_profile", lambda config, path: None)
    task = SimpleNamespace(
        stage="sim_cross",
        seed=7,
        latency_seed=13,
        n_episodes=100,
        smoke_max_steps=None,
        output_dir=tmp_path / "run",
        row={},
        target_profile_path=tmp_path / "profile.json",
        target_device={"profile_id": "L_A100", "id": "a100"},
        replay_device={"id": "rtx5090"},
    )

    generated = configs._generated_config(task, {"sim_parallel_envs": 1})

    assert generated["experiment"]["seed"] == 7
    assert generated["latency"]["seed"] == 13
