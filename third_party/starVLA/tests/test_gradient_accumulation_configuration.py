from types import SimpleNamespace

import pytest

from starVLA.training import train_starvla
from starVLA.training.trainer_utils import trainer_tools


def test_build_accelerator_uses_trainer_gradient_accumulation_steps(monkeypatch):
    plugin = object()
    accelerator_calls = []

    class FakeAccelerator:
        state = "test-state"

        def __init__(self, **kwargs):
            accelerator_calls.append(kwargs)

        def print(self, value):
            assert value == self.state

    monkeypatch.setattr(trainer_tools, "DeepSpeedPlugin", lambda: plugin)
    monkeypatch.setattr(trainer_tools, "Accelerator", FakeAccelerator)

    accelerator = trainer_tools.build_accelerator(
        SimpleNamespace(
            trainer=SimpleNamespace(
                gradient_accumulation_steps=4,
            )
        ),
        use_deepspeed=True,
    )

    assert isinstance(accelerator, FakeAccelerator)
    assert accelerator_calls == [
        {
            "gradient_accumulation_steps": 4,
            "deepspeed_plugin": plugin,
        }
    ]


def test_build_accelerator_requires_gradient_accumulation_steps(monkeypatch):
    monkeypatch.setattr(trainer_tools, "Accelerator", object)

    with pytest.raises(AttributeError):
        trainer_tools.build_accelerator(
            SimpleNamespace(
                trainer=SimpleNamespace(),
            ),
            use_deepspeed=False,
        )


def test_build_accelerator_without_deepspeed_plugin(monkeypatch):
    accelerator_calls = []

    class FakeAccelerator:
        state = "test-state"

        def __init__(self, **kwargs):
            accelerator_calls.append(kwargs)

        def print(self, value):
            assert value == self.state

    monkeypatch.setattr(trainer_tools, "Accelerator", FakeAccelerator)

    trainer_tools.build_accelerator(
        SimpleNamespace(
            trainer=SimpleNamespace(
                gradient_accumulation_steps=4,
            )
        ),
        use_deepspeed=False,
    )

    assert accelerator_calls == [{"gradient_accumulation_steps": 4}]


def test_quota_steps_use_accelerator_effective_accumulation(monkeypatch):
    effective_batch_sizes = []
    plan = [
        {
            "phase_idx": 0,
            "name": "test",
            "steps": 5,
            "dataset_size": 160,
            "quotas": {0: 1.0},
            "rows_by_latency": {0: 160},
            "end_step": 5,
        }
    ]

    def fake_build_plan(cfg, dataset, effective_batch_size):
        effective_batch_sizes.append(effective_batch_size)
        return plan

    monkeypatch.setattr(train_starvla, "_build_quota_cumulative_plan", fake_build_plan)
    curriculum = SimpleNamespace(
        enabled=True,
        strategy="quota_cumulative",
        step_budget_mode="auto",
    )
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            gradient_accumulation_steps=4,
            max_train_steps=0,
        ),
        datasets=SimpleNamespace(
            vla_data=SimpleNamespace(
                per_device_batch_size=2,
                latency_curriculum=curriculum,
            )
        ),
    )
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=8,
        num_processes=2,
    )

    train_starvla._configure_quota_cumulative_training_steps(
        cfg=cfg,
        dataloader=SimpleNamespace(dataset=object()),
        accelerator=accelerator,
    )

    assert effective_batch_sizes == [32]
    assert cfg.trainer.max_train_steps == 5
    assert curriculum.computed_plan == plan
