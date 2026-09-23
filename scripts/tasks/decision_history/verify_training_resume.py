"""Exercise the real two-rank ZeRO-2 checkpoint path with a small stochastic model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.training.train_starvla import VLATrainer, setup_optimizer_and_scheduler
from starVLA.training.trainer_utils.trainer_tools import build_accelerator


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = torch.nn.Linear(4, 8)
        self.dropout = torch.nn.Dropout(.15)
        self.action_model = torch.nn.Linear(8, 2)

    def forward(self, batch):
        output = self.action_model(self.dropout(self.qwen_vl_interface(batch)))
        return {"action_loss": output.float().square().mean(), "loss_weight": len(batch)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({
        "output_dir": str(args.output_dir), "seed": 42,
        "datasets": {"vla_data": {"per_device_batch_size": 64}},
        "trainer": {
            "max_train_steps": 8, "is_resume": False, "pretrained_checkpoint": None,
            "gradient_accumulation_steps": 1,
            "freeze_modules": "", "freeze_vit": False, "freeze_tied_embedding": False,
            "freeze_llm_layers": [], "profile_timing": {"enabled": False, "log_interval": 10},
            "learning_rate": {"base": 3e-5, "qwen_vl_interface": 3e-6, "action_model": 1e-4},
            "optimizer": {"betas": [.9, .95], "eps": 1e-8, "weight_decay": 1e-8, "fused": True},
            "lr_scheduler_type": "cosine_with_min_lr", "num_warmup_steps": 2,
            "scheduler_specific_kwargs": {"min_lr_rate": 1 / 30},
        },
        "checkpoint": {"save_best_model": False, "save_final_model": True, "save_pt_file": True,
                       "save_training_state": True, "local": {"keep_last_n": 2},
                       "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0}},
    })
    torch.manual_seed(42)
    model = StochasticModel()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
    accelerator = build_accelerator(cfg, use_deepspeed=True)
    assert accelerator.num_processes == 2
    loader = torch.utils.data.DataLoader(torch.ones(128, 4), batch_size=64)
    VLATrainer._init_wandb = lambda self: None
    trainer = VLATrainer(cfg, model, loader, None, optimizer, scheduler, accelerator)
    trainer.prepare_training()
    trainer.model.train()
    batch = torch.ones(64, 4, device=accelerator.device)

    def update():
        metrics = trainer._train_step(batch)
        assert metrics["_optimizer_step"]
        trainer.completed_steps += 1

    for _ in range(4):
        update()
    trainer._save_checkpoint()
    saved_lr = trainer.lr_scheduler.get_last_lr().copy()
    cpu_random = torch.rand(7)
    gpu_random = torch.rand(7, device=accelerator.device)
    numpy_random = np.random.random(7)
    python_random = random.random()
    update()
    expected = {key: value.detach().cpu().clone()
                for key, value in accelerator.get_state_dict(trainer.model).items()}
    expected_lr = trainer.lr_scheduler.get_last_lr().copy()
    trainer._load_checkpoint(Path(trainer.checkpoint_dir) / "steps_4_state")
    trainer.completed_steps = 4
    assert trainer.lr_scheduler.last_epoch == 4
    assert trainer.lr_scheduler.get_last_lr() == saved_lr
    torch.testing.assert_close(torch.rand(7), cpu_random, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(7, device=accelerator.device), gpu_random, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.random(7), numpy_random)
    assert random.random() == python_random
    update()
    assert trainer.lr_scheduler.get_last_lr() == expected_lr
    for key, value in accelerator.get_state_dict(trainer.model).items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result = {"passed": True, "world_size": 2, "saved_step": 4, "compared_step": 5,
                  "saved_lr": saved_lr, "next_lr": expected_lr,
                  "rng_streams": ["torch_cpu", "torch_cuda", "numpy", "python"]}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("HIST8_RESUME_VERIFIED", json.dumps(result), flush=True)
    trainer.cleanup_runtime()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
