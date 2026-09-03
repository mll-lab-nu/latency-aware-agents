from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


REPO = Path(__file__).resolve().parents[2]


def test_balance_fasttd3_config_has_43d_unit_action_bounds():
    config = yaml.safe_load(
        (
            REPO
            / "configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_train.yaml"
        ).read_text()
    )
    action_space = config["env"]["action_space"]

    assert len(action_space["labels"]) == 43
    assert action_space["low"] == [-1.0] * 43
    assert action_space["high"] == [1.0] * 43


def _load_runner(monkeypatch):
    loader_module = ModuleType("gr00t.data.dataset.lerobot_episode_loader")
    loader_module.LeRobotEpisodeLoader = object
    policy_module = ModuleType("gr00t.policy")
    policy_module.Gr00tPolicy = object
    monkeypatch.setitem(
        sys.modules, "gr00t.data.dataset.lerobot_episode_loader", loader_module
    )
    monkeypatch.setitem(sys.modules, "gr00t.policy", policy_module)
    spec = importlib.util.spec_from_file_location(
        "train_balance_simple_periodic_eval",
        REPO / "scripts/official_gr00t/train_balance_simple_periodic_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balance_runner_uses_explicit_training_paths(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch)
    calls = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    base_model = tmp_path / "base-model"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    runner.train_to_next_checkpoint(
        base_model=base_model,
        dataset=dataset,
        output=output,
        run_id="balance-test",
        step=500,
    )

    command = calls[0][0][0]
    assert command[command.index("--base-model-path") + 1] == str(base_model)
    assert command[command.index("--dataset-path") + 1] == str(dataset)
    assert command[command.index("--output-dir") + 1] == str(output)
    assert command[command.index("--experiment-name") + 1] == "balance-test"
    assert calls[0][1]["env"]["WANDB_RUN_ID"] == "balance-test"
