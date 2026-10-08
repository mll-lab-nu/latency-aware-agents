from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _ForbiddenAccelerator:
    is_local_main_process = False
    sync_gradients = False

    def __init__(self):
        from accelerate.utils import DistributedType

        self.distributed_type = DistributedType.DEEPSPEED

    def accumulate(self, model):
        raise AssertionError("DeepSpeed path must not use accelerator.accumulate")

    def backward(self, loss):
        raise AssertionError("DeepSpeed path must call model.backward directly")


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _DeepSpeedLikeModel:
    def __init__(self, boundary_every):
        self.boundary_every = boundary_every
        self.forward_calls = 0
        self.backward_calls = 0
        self.step_calls = 0

    def forward(self, batch):
        import torch

        self.forward_calls += 1
        return {"action_loss": torch.tensor(float(self.forward_calls)), "loss_weight": float(self.forward_calls)}

    def backward(self, loss):
        self.backward_calls += 1

    def is_gradient_accumulation_boundary(self):
        return self.backward_calls % self.boundary_every == 0

    def step(self):
        self.step_calls += 1

    def get_global_grad_norm(self):
        return 7.5


def _trainer(boundary_every=4):
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = _DeepSpeedLikeModel(boundary_every)
    trainer.accelerator = _ForbiddenAccelerator()
    trainer.lr_scheduler = _Scheduler()
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))
    trainer._profile_timing_should_log = lambda *args, **kwargs: False
    trainer._train_loss_sum = 0.0
    trainer._train_loss_weight = 0.0
    return trainer


def test_deepspeed_engine_owns_accumulation_boundary():
    from starVLA.training import train_starvla

    original_autocast = train_starvla.torch.autocast
    train_starvla.torch.autocast = lambda *args, **kwargs: nullcontext()
    try:
        trainer = _trainer(boundary_every=4)
        metrics = [trainer._train_step({"batch": idx}) for idx in range(4)]
    finally:
        train_starvla.torch.autocast = original_autocast

    assert [item["_optimizer_step"] for item in metrics] == [False, False, False, True]
    assert "train/loss" not in metrics[0]
    assert metrics[3]["train/loss"] == 3.0
    assert metrics[3]["train/grad_norm_pre_clip"] == 7.5
    assert trainer.model.forward_calls == 4
    assert trainer.model.backward_calls == 4
    assert trainer.model.step_calls == 4
    assert trainer.lr_scheduler.steps == 1


class _AccumulatingAccelerator:
    is_local_main_process = False

    def __init__(self, boundary_every):
        from accelerate.utils import DistributedType

        self.distributed_type = DistributedType.NO
        self.boundary_every = boundary_every
        self.backward_calls = 0
        self.clip_calls = 0

    @property
    def sync_gradients(self):
        return self.backward_calls > 0 and self.backward_calls % self.boundary_every == 0

    def accumulate(self, model):
        return nullcontext()

    def backward(self, loss):
        self.backward_calls += 1

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_calls += 1
        return 3.25


class _Optimizer:
    def __init__(self):
        self.steps = 0
        self.zero_grad_calls = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        self.zero_grad_calls += 1


class _PlainModel:
    def __init__(self):
        self.forward_calls = 0

    def forward(self, batch):
        import torch

        self.forward_calls += 1
        return {"action_loss": torch.tensor(float(self.forward_calls)), "loss_weight": float(self.forward_calls)}

    def parameters(self):
        return []


def _plain_trainer(boundary_every=4):
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = _PlainModel()
    trainer.accelerator = _AccumulatingAccelerator(boundary_every)
    trainer.optimizer = _Optimizer()
    trainer.lr_scheduler = _Scheduler()
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=1.0))
    trainer._profile_timing_should_log = lambda *args, **kwargs: False
    trainer._train_loss_sum = 0.0
    trainer._train_loss_weight = 0.0
    return trainer


def test_non_deepspeed_logs_effective_batch_loss_at_accumulation_boundary():
    from starVLA.training import train_starvla

    original_autocast = train_starvla.torch.autocast
    train_starvla.torch.autocast = lambda *args, **kwargs: nullcontext()
    try:
        trainer = _plain_trainer(boundary_every=4)
        metrics = [trainer._train_step({"batch": idx}) for idx in range(4)]
    finally:
        train_starvla.torch.autocast = original_autocast

    assert [item["_optimizer_step"] for item in metrics] == [False, False, False, True]
    assert "train/loss" not in metrics[0]
    assert metrics[3]["train/loss"] == 3.0
    assert metrics[3]["train/grad_norm_pre_clip"] == 3.25
    assert trainer.model.forward_calls == 4
    assert trainer.optimizer.steps == 4
    assert trainer.lr_scheduler.steps == 1


if __name__ == "__main__":
    test_deepspeed_engine_owns_accumulation_boundary()
