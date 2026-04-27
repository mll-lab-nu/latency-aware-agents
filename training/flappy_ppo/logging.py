from __future__ import annotations

import os
import time


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


class NoopWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def close(self):
        return None


class MetricLogger:
    def __init__(self, writer, wandb_module=None):
        self.writer = writer
        self.wandb = wandb_module

    def add_scalar(self, tag: str, value, global_step: int) -> None:
        if hasattr(value, "item"):
            value = value.item()
        self.writer.add_scalar(tag, value, global_step)
        if self.wandb is not None:
            self.wandb.log({tag: value}, step=int(global_step))

    def close(self) -> None:
        self.writer.close()


def tensorboard_available() -> bool:
    return SummaryWriter is not None


def build_writer(args):
    if not getattr(args, "tensorboard", False):
        return NoopWriter()
    if SummaryWriter is None:
        print("[tensorboard] torch.utils.tensorboard unavailable. Disabling tensorboard.")
        return NoopWriter()
    run_name = (
        getattr(args, "run_name", "")
        or getattr(args, "wandb_run_name", "")
        or f"flappy_gpu_render__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(os.path.join(args.runs_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )
    return writer


def setup_wandb(args):
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] wandb not installed or failed to import ({e}). Continuing without wandb.")
        return None

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run_name = args.wandb_run_name or args.run_name or f"flappy_gpu_render_{int(time.time())}"
    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
            save_code=True,
        )
    except Exception as e:
        print(f"[wandb] wandb init failed ({e}). Continuing without wandb.")
        return None
    return wandb
