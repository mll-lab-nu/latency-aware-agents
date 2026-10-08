from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from starVLA.training.train_starvla import VLATrainer


class _FakeAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0

    def unwrap_model(self, model: "_FakeModel") -> "_FakeModel":
        return model


class _FakeModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class _FakeRunner:
    def __init__(self) -> None:
        self.model_training_during_run: bool | None = None

    def run(
        self,
        model: _FakeModel,
        step: int,
        stage: str,
        *,
        shard_rank: int,
        shard_count: int,
        save: bool,
    ) -> SimpleNamespace:
        self.model_training_during_run = bool(model.training)
        return SimpleNamespace(
            per_latency={},
            aggregate={
                "stage": stage,
                "step": step,
                "mean_reward": 0.0,
            },
            path=None,
        )


def test_rl_games_eval_runs_in_eval_mode_and_restores_training_mode() -> None:
    trainer = object.__new__(VLATrainer)
    model = _FakeModel()
    runner = _FakeRunner()
    trainer.model = model
    trainer.accelerator = _FakeAccelerator()
    trainer._rl_games_eval_runner = runner
    trainer.completed_steps = 7

    result = trainer._run_rl_games_eval_with_model_mode(stage="mid_train")

    assert result.aggregate["stage"] == "mid_train"
    assert runner.model_training_during_run is False
    assert model.training is True


def test_training_entry_enables_pretrained_eval_mode_submodules(monkeypatch) -> None:
    """Regression: trainer entry must recursively enable training behavior."""
    trainer = object.__new__(VLATrainer)
    trainer.model = torch.nn.Sequential(torch.nn.Dropout(p=1)).eval()
    trainer.config = OmegaConf.create({'trainer': {'stop_after_steps': 1}})
    trainer.completed_steps = 1
    trainer.accelerator = SimpleNamespace(is_local_main_process=False)
    trainer.graceful_stop_requested = False
    for method in ('_log_training_config', '_create_data_iterators',
                   '_apply_latency_curriculum', '_finalize_training'):
        monkeypatch.setattr(trainer, method, lambda **kwargs: None)

    trainer.train()

    torch.testing.assert_close(trainer.model(torch.ones(2)), torch.zeros(2))
