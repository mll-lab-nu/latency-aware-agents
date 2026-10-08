from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from starVLA.training import train_starvla
from starVLA.training.train_starvla import VLATrainer


class _FakeAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 1
        self.gradient_accumulation_steps = 1
        self.save_state_calls: list[str] = []
        self.messages: list[str] = []

    def save_state(self, output_dir: str, safe_serialization: bool) -> None:
        self.save_state_calls.append(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def wait_for_everyone(self) -> None:
        return None

    def print(self, message: str) -> None:
        self.messages.append(message)

    def get_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return model.state_dict()


@pytest.mark.parametrize(
    ("wandb_name", "expected"),
    [("mikasa-bootstrap-qwenoft", "mikasa-bootstrap-qwenoft"), (None, "bootstrap")],
)
def test_wandb_reports_name_batch_and_optimizer_step(tmp_path, monkeypatch, wandb_name, expected):
    config = {
        "output_dir": str(tmp_path),
        "run_id": "bootstrap",
        "wandb_project": "starvla_tasks",
        "wandb_entity": None,
        "wandb_group": "vla-train",
        "wandb_tags": ["mikasa", "h1", "formal", "profile-latency"],
        "training_latency_condition": "profile-latency",
        "datasets": {"vla_data": {"per_device_batch_size": 16}},
        "trainer": {"logging_frequency": 5},
    }
    if wandb_name is not None:
        config["wandb_name"] = wandb_name
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create(config)
    trainer.accelerator = _FakeAccelerator()
    trainer.accelerator.gradient_accumulation_steps = 8
    trainer.total_batch_size = 128
    init_calls = []
    config_updates = []
    metric_definitions = []
    history = []
    monkeypatch.setattr(train_starvla.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(train_starvla.wandb, "config", SimpleNamespace(
        update=lambda values, **kwargs: config_updates.append((values, kwargs)),
    ))
    monkeypatch.setattr(train_starvla.wandb, "define_metric", lambda name, **kwargs:
                        metric_definitions.append((name, kwargs)))
    monkeypatch.setattr(train_starvla.wandb, "log", lambda values, **kwargs:
                        history.append((values, kwargs)))

    trainer._init_wandb()

    assert init_calls[0]["name"] == expected
    assert init_calls[0]["group"] == "vla-train"
    assert init_calls[0]["tags"] == ["mikasa", "h1", "formal", "profile-latency"]
    assert config_updates == [({
        "micro_batch": 16, "gradient_accumulation_steps": 8, "global_batch": 128,
        "training_latency_condition": "profile-latency",
    }, {"allow_val_change": True})]
    assert metric_definitions == [("global_step", {}), ("*", {"step_metric": "global_step"})]

    trainer.optimizer = SimpleNamespace(param_groups=[{"name": "base"}])
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [3e-5])
    trainer.vla_train_dataloader = SimpleNamespace(dataset=range(1280))
    for step in (150, 155):
        trainer.completed_steps = step
        trainer._log_metrics({"train/loss": 0.25})
    assert [values["global_step"] for values, _ in history] == [150, 155]
    assert [kwargs for _, kwargs in history] == [{}, {}]
    assert all(values["train/loss"] == 0.25 for values, _ in history)


def test_model_only_checkpoint_does_not_save_full_training_state(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "trainer": {},
            "checkpoint": {
                "save_best_model": False,
                "save_final_model": True,
                "save_pt_file": True,
                "save_training_state": False,
                "local": {"keep_last_n": 0},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            },
        }
    )
    model = torch.nn.Linear(2, 2, bias=False)
    accelerator = _FakeAccelerator()
    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=None,
        vla_eval_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        accelerator=accelerator,
    )
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(trainer.checkpoint_dir).mkdir()
    trainer.completed_steps = 400

    trainer._save_checkpoint()

    model_path = Path(trainer.checkpoint_dir) / "steps_400_pytorch_model.pt"
    state_path = Path(trainer.checkpoint_dir) / "steps_400_state"
    summary_path = tmp_path / "summary.jsonl"

    assert accelerator.save_state_calls == []
    assert model_path.exists()
    assert not state_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8").strip()) == {"steps": 400}


def test_completed_dagger_boundary_refreshes_dataset_without_reopening_service(tmp_path: Path) -> None:
    class Dataset:
        def __init__(self) -> None:
            self.refreshes = 0

        def refresh_custom_mixture(self) -> None:
            self.refreshes += 1

    class Loader:
        def __init__(self) -> None:
            self.dataset = Dataset()
            self.iterator_count = 0

        def __iter__(self):
            self.iterator_count += 1
            return iter(())

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    metrics = {"round": 2, "eval/mean_length": 42.0}
    (control_dir / "step_500.continue.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    trainer = VLATrainer.__new__(VLATrainer)
    trainer._dagger_control_dir = control_dir
    trainer.completed_steps = 500
    trainer.accelerator = _FakeAccelerator()
    trainer.vla_train_dataloader = Loader()
    trainer.graceful_stop_requested = False
    logged = []
    trainer._log_wandb = logged.append

    assert trainer._wait_for_dagger_round() is True
    assert not (control_dir / "step_500.ready").exists()
    assert trainer.vla_train_dataloader.dataset.refreshes == 1
    assert trainer.vla_train_dataloader.iterator_count == 1
    assert logged == [metrics]


def test_dagger_boundary_serves_eval_model_then_restores_training(tmp_path, monkeypatch) -> None:
    from latency_bench.integrations import starvla_resident_dagger
    from latency_bench.policy import starvla

    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 1)
            self.memory_resets = 0

        def reset_memory(self):
            self.memory_resets += 1

    class Accelerator(_FakeAccelerator):
        process_index = 0

        def unwrap_model(self, model):
            return model

    class Dataset:
        def __init__(self):
            self.refreshes = 0

        def refresh_custom_mixture(self):
            self.refreshes += 1

    class Loader:
        def __init__(self):
            self.dataset = Dataset()

        def __iter__(self):
            return iter(())

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / "resident_policy.json").write_text(json.dumps({
        "socket_dir": str(control_dir), "policy": {}
    }))
    model = Model()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    policy = SimpleNamespace(predict_batch=lambda observations: observations)
    served = []
    inference_modes = []
    monkeypatch.setattr(
        starvla,
        "build_live_starvla_policy",
        lambda **kwargs: served.append((kwargs["framework"], kwargs["framework"].training)) or policy,
    )
    monkeypatch.setattr(
        starvla_resident_dagger,
        "serve_resident_inference",
        lambda **kwargs: (
            inference_modes.append(torch.is_inference_mode_enabled())
            or {"metrics": {"round": 2}}
        ),
    )
    trainer = VLATrainer.__new__(VLATrainer)
    trainer._dagger_control_dir = control_dir
    trainer.completed_steps = 500
    trainer.accelerator = Accelerator()
    trainer.model = model
    trainer.config = OmegaConf.create({"datasets": {"vla_data": {}}})
    trainer.vla_train_dataloader = Loader()
    trainer.graceful_stop_requested = False
    trainer._log_wandb = lambda metrics: None

    assert trainer._wait_for_dagger_round() is True
    assert served == [(model, False)]
    assert inference_modes == [True]
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )
    assert model.training is True
    assert model.memory_resets == 1
    assert trainer.vla_train_dataloader.dataset.refreshes == 1


@pytest.mark.parametrize("round_boundary", [False, True])
def test_round_resume_restores_optimizer_and_lr_schedule(tmp_path, monkeypatch, round_boundary):
    from accelerate import Accelerator

    monkeypatch.setattr(VLATrainer, '_init_wandb', lambda self: None)

    def make_trainer(output, checkpoint=None):
        output.mkdir()
        cfg = OmegaConf.create({
            'output_dir': str(output), 'seed': 0,
            'datasets': {'vla_data': {'per_device_batch_size': 1}},
            'trainer': {'max_train_steps': 4, 'is_resume': checkpoint is not None,
                        'pretrained_checkpoint': None if checkpoint is None else str(checkpoint),
                        'freeze_modules': '', 'freeze_vit': False,
                        'freeze_tied_embedding': False, 'freeze_llm_layers': []},
            'checkpoint': {'save_best_model': False, 'save_final_model': True,
                           'save_pt_file': True, 'save_training_state': True,
                           'local': {'keep_last_n': 0},
                           'sync': {'enabled': False, 'repo_id': None, 'keep_last_n': 0}},
        })
        if round_boundary:
            cfg.trainer.stop_after_steps = 2 if checkpoint is None else 4
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 5)
        trainer = VLATrainer(cfg, model, None, None, optimizer, scheduler, Accelerator(cpu=True))
        trainer.prepare_training()
        return trainer

    def update(trainer):
        trainer.optimizer.zero_grad()
        trainer.accelerator.backward(trainer.model(torch.ones(1, 2)).square().sum())
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.completed_steps += 1

    original = make_trainer(tmp_path / 'original')
    update(original)
    update(original)
    original._save_checkpoint()
    expected_torch = torch.rand(5)
    expected_numpy = np.random.random(5)
    expected_python = random.random()
    resumed = make_trainer(tmp_path / 'resumed',
                           Path(original.checkpoint_dir) / 'steps_2_state')
    assert resumed.completed_steps == 2
    assert resumed.lr_scheduler.last_epoch == original.lr_scheduler.last_epoch
    assert resumed.optimizer.param_groups[0]['lr'] == original.optimizer.param_groups[0]['lr']
    torch.testing.assert_close(torch.rand(5), expected_torch)
    np.testing.assert_array_equal(np.random.random(5), expected_numpy)
    assert random.random() == expected_python
    update(original)
    update(resumed)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert resumed.lr_scheduler.get_last_lr() == original.lr_scheduler.get_last_lr()


def test_fresh_action_head_initialization_uses_configured_seed(monkeypatch):
    monkeypatch.setattr(train_starvla, "build_framework", lambda cfg: torch.nn.Linear(11, 6))
    cfg = OmegaConf.create({"seed": 42})
    first = train_starvla.build_initial_framework(cfg)
    torch.randn(100)
    second = train_starvla.build_initial_framework(cfg)
    for expected, actual in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    different = train_starvla.build_initial_framework(OmegaConf.create({"seed": 43}))
    assert not torch.equal(first.weight, different.weight)
