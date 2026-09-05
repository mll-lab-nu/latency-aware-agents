from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
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


def test_balance_modality_repeats_one_action_across_h40(monkeypatch):
    registered = {}

    class ModalityConfig:
        def __init__(self, *, delta_indices, **kwargs):
            self.delta_indices = delta_indices
            self.kwargs = kwargs

    class ActionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class EmbodimentTag:
        NEW_EMBODIMENT = "NEW_EMBODIMENT"

    configs = ModuleType("gr00t.configs.data.embodiment_configs")
    configs.register_modality_config = lambda config, embodiment_tag: registered.update(
        config=config, embodiment_tag=embodiment_tag
    )
    types = ModuleType("gr00t.data.types")
    types.ActionConfig = ActionConfig
    types.ActionFormat = SimpleNamespace(DEFAULT="default")
    types.ActionRepresentation = SimpleNamespace(ABSOLUTE="absolute")
    types.ActionType = SimpleNamespace(NON_EEF="non_eef")
    types.ModalityConfig = ModalityConfig
    tags = ModuleType("gr00t.data.embodiment_tags")
    tags.EmbodimentTag = EmbodimentTag
    for name, module in {
        "gr00t": ModuleType("gr00t"),
        "gr00t.configs": ModuleType("gr00t.configs"),
        "gr00t.configs.data": ModuleType("gr00t.configs.data"),
        "gr00t.configs.data.embodiment_configs": configs,
        "gr00t.data": ModuleType("gr00t.data"),
        "gr00t.data.types": types,
        "gr00t.data.embodiment_tags": tags,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "balance_modality",
        REPO / "latency_bench/policy/official_gr00t_modality.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert registered["config"]["state"].delta_indices == [0]
    assert registered["config"]["action"].delta_indices == [0] * 40


def _load_runner(monkeypatch):
    loader_module = ModuleType("gr00t.data.dataset.lerobot_episode_loader")
    loader_module.LeRobotEpisodeLoader = object
    stats_module = ModuleType("gr00t.data.stats")
    stats_module.generate_stats = lambda dataset: None
    policy_module = ModuleType("gr00t.policy")
    policy_module.Gr00tPolicy = object
    monkeypatch.setitem(
        sys.modules, "gr00t.data.dataset.lerobot_episode_loader", loader_module
    )
    monkeypatch.setitem(sys.modules, "gr00t.data.stats", stats_module)
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
    checkpoint = runner.train_to_next_checkpoint(
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
    assert checkpoint == output / "balance-test" / "checkpoint-500"


def test_balance_runner_accepts_a_finite_resume_schedule(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    calls = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    for step, resume in ((1, False), (2, True)):
        runner.train_to_next_checkpoint(
            base_model=tmp_path / "base-model",
            dataset=tmp_path / "dataset",
            output=tmp_path / "output",
            run_id="balance-finite-test",
            step=step,
            max_steps=2,
            step_interval=1,
            resume=resume,
            num_gpus=1,
            global_batch_size=1,
        )

    first_command = calls[0][0][0]
    second_command = calls[1][0][0]
    assert first_command[first_command.index("--max-steps") + 1] == "2"
    assert first_command[first_command.index("--save-steps") + 1] == "1"
    assert "--nproc_per_node=1" in first_command
    assert first_command[first_command.index("--global-batch-size") + 1] == "1"
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--resume-from-checkpoint" not in first_command
    assert "--resume-from-checkpoint" in second_command


def test_balance_validation_supports_short_repeat_current_episode(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch)
    (tmp_path / "val" / "meta").mkdir(parents=True)
    monkeypatch.setattr(
        runner, "generate_stats",
        lambda dataset: (dataset / "meta" / "stats.json").write_text("{}"),
    )
    episode = pd.DataFrame(
        {
            "state.state": [np.zeros(198, dtype=np.float32)] * 19,
            "video.ego_view": [np.zeros((2, 2, 3), dtype=np.uint8)] * 19,
            "language.annotation.human.task_description": ["balance"] * 19,
            "action.action": [np.arange(43, dtype=np.float32) + i for i in range(19)],
        }
    )

    class FakePolicy:
        def __init__(self, **kwargs):
            pass

        def get_modality_config(self):
            return {}

        def get_action(self, batch):
            size = len(batch["state"]["state"])
            return {"action": np.zeros((size, 40, 43), dtype=np.float32)}, None

    class FakeLoader:
        def __init__(self, dataset, modality_config):
            json.loads((dataset / "meta" / "stats.json").read_text())

        def __getitem__(self, index):
            return episode

    monkeypatch.setattr(runner, "Gr00tPolicy", FakePolicy)
    monkeypatch.setattr(runner, "LeRobotEpisodeLoader", FakeLoader)

    metrics = runner.validation_metrics(tmp_path / "checkpoint", tmp_path / "val")

    assert metrics["body_mae"] == 23.0
    assert metrics["all_mae"] == 30.0
    assert metrics["repeat_std_max"] == 0.0


def test_balance_periodic_eval_propagates_full_latency_config(monkeypatch, tmp_path):
    runner = _load_runner(monkeypatch)
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 0},
                "executor": {"mode": "simulated", "simulated_worker_capacity": 1},
                "latency": {"method": "zero", "sync_cuda": False},
                "scheduler": {
                    "hold_policy": "hold",
                    "ordering_policy": "latest_ready",
                },
                "policy": {"model_path": "base"},
                "logging": {"output_dir": str(tmp_path / "base")},
            },
            sort_keys=False,
        )
    )
    latency_config = tmp_path / "latency.yaml"
    latency_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"seed": 37},
                "executor": {
                    "mode": "simulated",
                    "simulated_worker_capacity": 3,
                },
                "latency": {
                    "method": "temporal",
                    "profile_path": "profile-from-config.json",
                    "profile_worker_slot": 2,
                    "seed": 19,
                },
                "scheduler": {
                    "hold_policy": "one_frame_then_noop",
                    "ordering_policy": "issue_order_fifo",
                },
            },
            sort_keys=False,
        )
    )
    observed = []

    def fake_run(command, **kwargs):
        config = yaml.safe_load(Path(command[-1]).read_text())
        observed.append(config)
        output = Path(config["logging"]["output_dir"])
        output.mkdir(parents=True)
        (output / "episode_metrics.jsonl").write_text(
            json.dumps({"episode_return": 2.0, "survival_steps": 4}) + "\n"
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for method in ("zero", "temporal"):
        runner.closed_loop_metrics(
            tmp_path / "checkpoint",
            method,
            output=tmp_path / "output",
            eval_config=eval_config,
            latency_profile_path=tmp_path / "profile-from-arg.json",
            latency_config_path=latency_config,
            latency_method=method,
        )

    assert len(observed) == 2
    for config, method in zip(observed, ("zero", "temporal")):
        assert config["experiment"]["seed"] == 0
        assert config["executor"]["simulated_worker_capacity"] == 3
        assert config["latency"]["method"] == method
        assert config["latency"]["seed"] == 19
        assert config["latency"]["profile_worker_slot"] == 2
        assert config["latency"]["profile_path"] == "profile-from-config.json"
        assert config["scheduler"]["hold_policy"] == "one_frame_then_noop"
        assert config["scheduler"]["ordering_policy"] == "issue_order_fifo"
