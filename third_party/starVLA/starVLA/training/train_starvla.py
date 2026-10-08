# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import logging
import math
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
import wandb
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.dataloader.worker_context import build_cpu_only_dataloader_kwargs
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.rl_games import CheckpointSyncManager, apply_action_spec, apply_model_alias, build_rl_games_eval_runner, sync_kv_memory_obs_window, validate_rl_games_config
from starVLA.training.rl_games.auth import login_training_services
from starVLA.training.rl_games.eval_core import EvalResult
from starVLA.training.rl_games import action_cc_f1
from starVLA.training.train_step_events import (
    calculate_epoch_progress,
    should_run_optional_step_interval_event,
    should_run_step_interval_event,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_accelerator, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = logging.getLogger(__name__)

_DYNAMO_RECOMPILE_LIMIT = 64
_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT = 1024
_GRACEFUL_STOP_REQUESTED = False
_GRACEFUL_STOP_SIGNAL = None


def _configure_torch_dynamo_recompile_limits() -> None:
    config = torch._dynamo.config
    config.recompile_limit = max(config.recompile_limit, _DYNAMO_RECOMPILE_LIMIT)
    config.accumulated_recompile_limit = max(
        config.accumulated_recompile_limit,
        _DYNAMO_ACCUMULATED_RECOMPILE_LIMIT,
    )


def _as_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _request_graceful_stop(signum, frame) -> None:
    global _GRACEFUL_STOP_REQUESTED, _GRACEFUL_STOP_SIGNAL
    _GRACEFUL_STOP_REQUESTED = True
    _GRACEFUL_STOP_SIGNAL = signum


def _install_graceful_stop_handlers() -> dict[int, Any]:
    global _GRACEFUL_STOP_REQUESTED, _GRACEFUL_STOP_SIGNAL
    _GRACEFUL_STOP_REQUESTED = False
    _GRACEFUL_STOP_SIGNAL = None
    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _request_graceful_stop)
    signal.signal(signal.SIGTERM, _request_graceful_stop)
    return previous_handlers


def _restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    backend = str(dist.get_backend()).lower()
    if backend == "nccl" and torch.cuda.is_available():
        device_idx = int(torch.cuda.current_device())
        dist.barrier(device_ids=[device_idx])
        return
    dist.barrier()


def _destroy_distributed_process_group(*, use_barrier: bool) -> None:
    if not dist.is_initialized():
        return
    if use_barrier:
        try:
            _distributed_barrier()
        except Exception as exc:
            logger.warning("Distributed barrier failed during shutdown: %s", exc)
    try:
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")
    except Exception as exc:
        logger.warning("Failed to destroy distributed process group: %s", exc)


def _sample_latency(sample: dict) -> int | None:
    if not isinstance(sample, dict):
        return None
    value = sample.get("latency", sample.get("latency_id", None))
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    return int(value)


def _sample_task(sample: dict) -> str | None:
    if not isinstance(sample, dict):
        return None
    value = sample.get("rl_games_task", sample.get("task", None))
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _group_examples_by_latency(examples: list[dict]) -> dict[int | None, list[dict]]:
    grouped: dict[int | None, list[dict]] = {}
    for example in examples:
        grouped.setdefault(_sample_latency(example), []).append(example)
    return grouped


def _group_examples_by_task_latency(examples: list[dict]) -> dict[tuple[str | None, int | None], list[dict]]:
    grouped: dict[tuple[str | None, int | None], list[dict]] = {}
    for example in examples:
        grouped.setdefault((_sample_task(example), _sample_latency(example)), []).append(example)
    return grouped


def _optional_positive_int(value) -> int | None:
    if value in (None, ""):
        return None
    value = int(value)
    return value if value > 0 else None


def _optional_int_list(value) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        if not stripped:
            return []
        return [int(item.strip()) for item in stripped.split(",") if item.strip()]
    return [int(item) for item in value]


def _quota_interval_row_count(base_count: int, pass_start: float, quota: float) -> int:
    """Rows selected by the pass interval [pass_start, pass_start + quota)."""
    if int(base_count) <= 0 or float(quota) <= 0.0:
        return 0
    cursor = float(pass_start)
    remaining = float(quota)
    selected = 0
    eps = 1e-9
    while remaining > eps:
        pass_idx = math.floor(cursor)
        frac_start = cursor - pass_idx
        if 1.0 - frac_start <= eps:
            cursor = float(pass_idx + 1)
            continue
        segment = min(remaining, 1.0 - frac_start)
        frac_end = frac_start + segment
        start_idx = int(math.ceil(float(base_count) * frac_start - eps))
        end_idx = int(math.ceil(float(base_count) * frac_end - eps))
        selected += max(0, min(int(base_count), end_idx) - min(int(base_count), start_idx))
        cursor += segment
        remaining -= segment
    return int(selected)


def _latency_curriculum_cfg_from_config(cfg):
    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    if vla_data is None:
        return None
    curriculum_cfg = getattr(vla_data, "latency_curriculum", None)
    if curriculum_cfg is None or not _as_bool(getattr(curriculum_cfg, "enabled", False), default=False):
        return None
    return curriculum_cfg


def _dataset_latency_step_counts(dataset) -> dict[int, int]:
    if dataset is not None and callable(getattr(dataset, "get_latency_step_counts", None)):
        return {int(k): int(v) for k, v in dataset.get_latency_step_counts().items()}

    counts: dict[int, int] = {}
    single_datasets = list(getattr(dataset, "datasets", None) or [dataset])
    for ds in single_datasets:
        if ds is None or not callable(getattr(ds, "get_latencies_for_all_steps", None)):
            continue
        for latency in ds.get_latencies_for_all_steps():
            latency = int(latency)
            counts[latency] = counts.get(latency, 0) + 1
    return counts


def _build_quota_cumulative_plan(cfg, dataset, effective_batch_size: int) -> list[dict[str, Any]]:
    curriculum_cfg = _latency_curriculum_cfg_from_config(cfg)
    if curriculum_cfg is None:
        return []
    strategy = str(getattr(curriculum_cfg, "strategy", "") or "").strip().lower()
    if strategy != "quota_cumulative":
        return []

    latencies = _optional_int_list(getattr(curriculum_cfg, "latencies", None))
    if not latencies:
        latencies = _dataset_latency_values(dataset)
    latencies = sorted(dict.fromkeys(int(value) for value in latencies or []))
    if not latencies:
        raise ValueError("quota_cumulative requires configured or discoverable curriculum latencies.")

    counts = _dataset_latency_step_counts(dataset)
    missing = [latency for latency in latencies if int(counts.get(latency, 0)) <= 0]
    if missing:
        raise ValueError(f"quota_cumulative found no training rows for latencies={missing}; counts={counts}.")

    new_latency_passes = float(getattr(curriculum_cfg, "new_latency_passes", 1.0))
    replay_passes = float(getattr(curriculum_cfg, "replay_passes", 0.25))
    target_total_passes = float(getattr(curriculum_cfg, "target_total_passes", 2.0))
    final_equalization = _as_bool(getattr(curriculum_cfg, "final_equalization", True), default=True)
    if new_latency_passes <= 0.0:
        raise ValueError("quota_cumulative.new_latency_passes must be positive.")
    if replay_passes < 0.0:
        raise ValueError("quota_cumulative.replay_passes must be non-negative.")
    if target_total_passes <= 0.0:
        raise ValueError("quota_cumulative.target_total_passes must be positive.")
    if effective_batch_size <= 0:
        raise ValueError(f"quota_cumulative effective batch size must be positive, got {effective_batch_size}.")

    cumulative_passes = {latency: 0.0 for latency in latencies}
    phases: list[dict[str, Any]] = []
    cursor = 0

    def append_phase(name: str, quotas: dict[int, float]) -> None:
        nonlocal cursor
        cleaned = {int(latency): float(quota) for latency, quota in quotas.items() if float(quota) > 0.0}
        if not cleaned:
            return
        quota_starts = {
            latency: float(cumulative_passes.get(latency, 0.0))
            for latency in cleaned
        }
        rows_by_latency = {
            latency: _quota_interval_row_count(int(counts[latency]), quota_starts[latency], quota)
            for latency, quota in cleaned.items()
        }
        total_rows = int(sum(rows_by_latency.values()))
        if total_rows <= 0:
            return
        steps = int(math.ceil(total_rows / effective_batch_size))
        start_step = cursor
        cursor += steps
        phases.append(
            {
                "phase_idx": len(phases),
                "name": name,
                "quotas": cleaned,
                "quota_starts": quota_starts,
                "rows_by_latency": rows_by_latency,
                "dataset_size": total_rows,
                "steps": steps,
                "start_step": start_step,
                "end_step": cursor,
            }
        )
        for latency, quota in cleaned.items():
            cumulative_passes[latency] = cumulative_passes.get(latency, 0.0) + float(quota)

    for idx, latency in enumerate(latencies):
        quotas = {prev_latency: replay_passes for prev_latency in latencies[:idx]}
        quotas[latency] = new_latency_passes
        append_phase(f"introduce_latency_{latency}", quotas)

    if final_equalization:
        top_up = {
            latency: max(0.0, target_total_passes - cumulative_passes.get(latency, 0.0))
            for latency in latencies
        }
        append_phase("equalize_total_coverage", top_up)

    if not phases:
        raise ValueError("quota_cumulative produced an empty phase plan.")
    return phases


def _configure_quota_cumulative_training_steps(cfg, dataloader, accelerator) -> None:
    curriculum_cfg = _latency_curriculum_cfg_from_config(cfg)
    if curriculum_cfg is None:
        return
    strategy = str(getattr(curriculum_cfg, "strategy", "") or "").strip().lower()
    if strategy != "quota_cumulative":
        return

    grad_accum_steps = accelerator.gradient_accumulation_steps
    per_device_bs = int(getattr(cfg.datasets.vla_data, "per_device_batch_size"))
    effective_batch_size = per_device_bs * int(accelerator.num_processes) * grad_accum_steps
    dataset = getattr(dataloader, "dataset", None)
    plan = _build_quota_cumulative_plan(cfg, dataset, effective_batch_size)
    total_steps = int(plan[-1]["end_step"])

    step_budget_mode = str(getattr(curriculum_cfg, "step_budget_mode", "auto") or "auto").strip().lower()
    if step_budget_mode != "auto":
        raise ValueError("quota_cumulative currently supports step_budget_mode=auto only.")

    cfg.trainer.max_train_steps = total_steps
    curriculum_cfg.computed_plan = plan
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "quota_cumulative derived max_train_steps=%s from %s phases and effective_batch_size=%s",
            total_steps,
            len(plan),
            effective_batch_size,
        )
        for phase in plan:
            logger.info(
                "quota_cumulative phase %s %s: steps=%s rows=%s quotas=%s rows_by_latency=%s end_step=%s",
                phase["phase_idx"],
                phase["name"],
                phase["steps"],
                phase["dataset_size"],
                phase["quotas"],
                phase["rows_by_latency"],
                phase["end_step"],
            )


def _parse_phase_distributions(raw: Any) -> list[dict[int, float]] | None:
    if raw is None:
        return None
    phases = list(raw)
    if not phases:
        return None
    return [{int(k): float(v) for k, v in dict(phase).items()} for phase in phases]


def _dataset_latency_values(dataset) -> list[int]:
    values: set[int] = set()
    single_datasets = list(getattr(dataset, "datasets", None) or [dataset])
    for ds in single_datasets:
        all_steps = getattr(ds, "all_steps", None) or []
        seen_trajectories: set[int] = set()
        for trajectory_id, _ in all_steps:
            trajectory_id = int(trajectory_id)
            if trajectory_id in seen_trajectories:
                continue
            seen_trajectories.add(trajectory_id)
            for latency_key in ("latency", "latency_id"):
                try:
                    latency_data = ds.get_trajectory_columns(trajectory_id, [latency_key])
                except Exception:
                    continue
                if latency_key not in latency_data.columns or len(latency_data) == 0:
                    continue
                values.update(int(value) for value in latency_data[latency_key].dropna().unique().tolist())
                break
    return sorted(values)


def _dataset_task_latency_values(dataset) -> dict[str, list[int]]:
    values: dict[str, set[int]] = {}
    single_datasets = list(getattr(dataset, "datasets", None) or [dataset])
    for ds in single_datasets:
        task = getattr(ds, "rl_games_task", None)
        if task in (None, ""):
            continue
        task = str(task)
        for latency in _dataset_latency_values(ds):
            values.setdefault(task, set()).add(int(latency))
    return {task: sorted(latencies) for task, latencies in values.items()}


def _episode_latency(ds, trajectory_id: int) -> int | None:
    for latency_key in ("latency", "latency_id"):
        try:
            latency_data = ds.get_trajectory_columns(int(trajectory_id), [latency_key])
        except Exception:
            continue
        if latency_key not in latency_data.columns or len(latency_data) == 0:
            continue
        values = latency_data[latency_key].dropna().unique().tolist()
        if values:
            return int(values[0])
    return None


def _identity_collate(batch):
    return batch


class _ActionCCF1FrameDataset(Dataset):
    def __init__(self, single_datasets: list, frame_records: list[tuple[int, int, str, int, str, int | None]]):
        self.single_datasets = single_datasets
        self.frame_records = frame_records

    def __len__(self) -> int:
        return len(self.frame_records)

    def __getitem__(self, index: int):
        dataset_index, flat_idx, episode_key, base_idx, ep_task, _episode_latency_value = self.frame_records[index]
        sample = self.single_datasets[dataset_index][flat_idx]
        sample_task = _sample_task(sample) or ep_task
        return sample_task, episode_key, base_idx, _sample_latency(sample), sample


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> tuple[DataLoader, DataLoader | None]:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vla_eval_dataloader = None
    eval_data_mix = getattr(cfg.datasets.vla_data, "eval_data_mix", None)
    if eval_data_mix:
        logger.info(f"Creating VLA Eval Dataset with Mixture `{eval_data_mix}`")
        vla_eval_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            data_mix=str(eval_data_mix),
            mode="eval",
            save_statistics_filename="dataset_statistics_eval.json",
        )

    accelerator.dataloader_config.dispatch_batches = False
    _distributed_barrier()
    return vla_train_dataloader, vla_eval_dataloader


def build_initial_framework(cfg):
    """Initialize the backbone and fresh action modules with the configured seed."""
    set_seed(cfg.seed)
    return build_framework(cfg)


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    fused = bool(getattr(cfg.trainer.optimizer, "fused", False))
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def _pin_cuda_device_from_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank in (None, ""):
        return
    try:
        device_idx = int(local_rank)
    except ValueError:
        logger.warning("Ignoring invalid LOCAL_RANK=%r for CUDA device pinning", local_rank)
        return

    visible_devices = torch.cuda.device_count()
    if device_idx < 0 or device_idx >= visible_devices:
        logger.warning(
            "LOCAL_RANK=%s cannot be mapped to a CUDA device; visible CUDA device count is %s",
            local_rank,
            visible_devices,
        )
        return
    torch.cuda.set_device(device_idx)


def _preload_model_checkpoint_before_accelerator(cfg, model):
    """Load model-only checkpoints before ZeRO-3 can replace params with shards."""
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return model

    pretrained_checkpoint = getattr(trainer_cfg, "pretrained_checkpoint", None)
    is_resume = bool(getattr(trainer_cfg, "is_resume", False))
    if not pretrained_checkpoint or is_resume:
        return model

    reload_modules = getattr(trainer_cfg, "reload_modules", None)
    model = TrainerUtils.load_pretrained_backbones(
        model,
        pretrained_checkpoint,
        reload_modules=reload_modules,
    )
    trainer_cfg.pretrained_checkpoint = None
    logger.info(
        "Preloaded model checkpoint before Accelerator/DeepSpeed initialization: %s",
        pretrained_checkpoint,
    )
    return model


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vla_eval_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._checkpoint_sync_manager = CheckpointSyncManager(cfg=self.config)
        self._rl_games_eval_runner = None
        self._pending_full_state_resume = None
        self._save_best_model_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_best_model", None),
            default=True,
        )
        self._save_final_model_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_final_model", None),
            default=True,
        )
        self._save_pt_file_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_pt_file", None),
            default=False,
        )
        self._save_training_state_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_training_state", None),
            default=True,
        )
        self._save_safetensors_file_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_safetensors_file", None),
            default=False,
        )
        self._best_score = float("-inf")
        self._best_step = 0
        self._best_state_path = None
        self._best_metadata_path = None
        self._latency_curriculum_phase = None
        self._action_cc_f1_frame_loader = None
        self._action_cc_f1_frame_loader_key = None
        self._train_loss_sum = 0.0
        self._train_loss_weight = 0.0
        self.graceful_stop_requested = False
        self._graceful_stop_logged = False
        self._interrupt_checkpoint_saved = False
        self._dagger_control_dir = (
            Path(self.config.trainer.dagger_control_dir)
            if "dagger_control_dir" in self.config.trainer
            else None
        )
        stop_file = os.environ.get("STARVLA_STOP_FILE")
        self._graceful_stop_file = Path(stop_file) if stop_file else None
        if hasattr(self.config, "rl_games") and hasattr(self.config.rl_games, "env_eval"):
            enabled = bool(getattr(self.config.rl_games.env_eval, "enabled", False))
            if enabled:
                self._rl_games_eval_runner = build_rl_games_eval_runner(
                    cfg=self.config,
                    output_dir=self.config.output_dir,
                )

    def _save_periodic_checkpoints_enabled(self) -> bool:
        return not self._save_best_model_enabled

    def prepare_training(self):
        _configure_torch_dynamo_recompile_limits()
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.model = self.freeze_vit_and_llm_layers(self.model, self.config)
        self.print_trainable_parameters(self.model)

        if self.vla_eval_dataloader is not None:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_eval_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_eval_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        if self._pending_full_state_resume:
            self._load_checkpoint(self._pending_full_state_resume)
        else:
            self._adjust_lr_scheduler_for_resume()

        self._init_wandb()
        self.accelerator.wait_for_everyone()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb_name = self.config.wandb_name if "wandb_name" in self.config else self.config.run_id
            wandb.init(
                name=wandb_name,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=self.config.wandb_group,
                tags=self.config.wandb_tags,
            )
            wandb.config.update({
                "micro_batch": self.config.datasets.vla_data.per_device_batch_size,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "global_batch": self.total_batch_size,
                "training_latency_condition": self.config.training_latency_condition,
            }, allow_val_change=True)
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._best_state_path = os.path.join(self.checkpoint_dir, "best_state")
        self._best_metadata_path = os.path.join(self.checkpoint_dir, "best_model_metadata.json")
        self._load_best_checkpoint_metadata()

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            if pretrained_checkpoint:
                resume_from_checkpoint = str(pretrained_checkpoint)
                self.completed_steps = int(
                    getattr(self.config.trainer, "resume_step", None) or self._step_from_checkpoint_path(resume_from_checkpoint)
                )
            else:
                resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(self.resume_from_checkpoint):
                    self._pending_full_state_resume = self.resume_from_checkpoint
                    logger.info(
                        f"Will resume full training state from checkpoint: "
                        f"{self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                else:
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                    logger.info(
                        f"Resuming model weights from checkpoint: "
                        f"{self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    @staticmethod
    def _step_from_checkpoint_path(checkpoint_path: str) -> int:
        match = re.search(r"steps_(\d+)", os.path.basename(str(checkpoint_path)))
        return int(match.group(1)) if match else 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _load_best_checkpoint_metadata(self):
        if not self._best_metadata_path or not os.path.isfile(self._best_metadata_path):
            return
        try:
            with open(self._best_metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self._best_score = float(metadata.get("best_score", self._best_score))
            self._best_step = int(metadata.get("best_step", self._best_step))
            logger.info(
                "Loaded best checkpoint metadata: step=%s score=%s",
                self._best_step,
                self._best_score,
            )
        except Exception as exc:
            logger.warning("Could not load best checkpoint metadata from %s: %s", self._best_metadata_path, exc)

    @staticmethod
    def _rl_games_best_score(eval_result) -> float:
        if "macro_mean_reward" in eval_result.aggregate:
            return float(eval_result.aggregate["macro_mean_reward"])
        bucket_scores = [
            float(metrics["mean_reward"])
            for metrics in eval_result.per_latency.values()
            if metrics is not None and "mean_reward" in metrics
        ]
        if bucket_scores:
            return float(np.mean(bucket_scores))
        return float(eval_result.aggregate.get("mean_reward", 0.0))

    @staticmethod
    def _eval_result_payload(eval_result) -> Dict[str, Any]:
        return {
            "per_latency": eval_result.per_latency,
            "aggregate": eval_result.aggregate,
        }

    def _distributed_rl_games_eval_enabled(self) -> bool:
        if self._rl_games_eval_runner is None:
            return False
        env_eval = getattr(self.config.rl_games, "env_eval", None)
        mode = str(getattr(env_eval, "distributed_mode", "none") or "none").strip().lower()
        return mode in {"rank_sharded", "distributed", "sharded"} and int(self.accelerator.num_processes) > 1

    def _run_rl_games_eval_with_model_mode(
        self,
        *,
        stage: str,
        shard_rank: int = 0,
        shard_count: int = 1,
        save: bool = True,
    ):
        was_training = self.model.training
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        try:
            run_eval = torch._dynamo.disable(self._rl_games_eval_runner.run)
            return run_eval(
                model=unwrapped,
                step=self.completed_steps,
                stage=stage,
                shard_rank=int(shard_rank),
                shard_count=int(shard_count),
                save=save,
            )
        finally:
            try:
                reset = getattr(unwrapped, "reset_memory", None)
                if callable(reset):
                    reset()
            finally:
                if was_training:
                    self.model.train()

    def _run_distributed_rl_games_eval(self, stage: str):
        if self._rl_games_eval_runner is None:
            return None

        local_result = self._run_rl_games_eval_with_model_mode(
            stage=stage,
            shard_rank=int(self.accelerator.process_index),
            shard_count=int(self.accelerator.num_processes),
            save=False,
        )
        local_payload = self._eval_result_payload(local_result)

        if dist.is_available() and dist.is_initialized():
            gathered_payloads = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_payloads, local_payload)
        else:
            gathered_payloads = [local_payload]

        partial_results = [
            EvalResult(per_latency=payload["per_latency"], aggregate=payload["aggregate"])
            for payload in gathered_payloads
            if payload is not None
        ]
        merged_result = self._rl_games_eval_runner.merge_results(
            partial_results,
            step=self.completed_steps,
            stage=stage,
        )
        if self.accelerator.is_main_process:
            self._rl_games_eval_runner.save(result=merged_result, step=self.completed_steps, stage=stage)
        self.accelerator.wait_for_everyone()
        return merged_result

    def _run_mid_train_rl_games_eval(self, step_metrics: dict[str, float]) -> dict[str, float]:
        if self._rl_games_eval_runner is None or not self._rl_games_eval_runner.is_enabled(stage="mid_train"):
            return step_metrics
        if self._refresh_graceful_stop_requested():
            if self.accelerator.is_main_process:
                logger.info("Skipping RL-games mid-train eval at step %s due to graceful stop", self.completed_steps)
            return step_metrics
        t_profile_eval = self._profile_start()
        if self._distributed_rl_games_eval_enabled():
            eval_result = self._run_distributed_rl_games_eval(stage="mid_train")
        else:
            eval_result = self._run_rl_games_eval_with_model_mode(stage="mid_train")
        step_metrics = self._append_rl_games_eval_metrics(
            step_metrics=step_metrics,
            eval_result=eval_result,
            stage="mid_train",
        )
        if self._profile_timing_should_log():
            step_metrics["timing/rl_games_mid_train_eval_total"] = self._profile_elapsed(t_profile_eval)
        eval_score = self._rl_games_best_score(eval_result)
        is_new_best = self._save_best_checkpoint(
            eval_result=eval_result,
            score=eval_score,
            stage="mid_train",
        )
        step_metrics["checkpoint/current_eval_score"] = float(eval_score)
        step_metrics["checkpoint/best_score"] = float(self._best_score)
        step_metrics["checkpoint/best_step"] = float(self._best_step)
        step_metrics["checkpoint/is_new_best"] = float(is_new_best)
        self._checkpoint_sync_manager.sync_eval_result(
            eval_path=eval_result.path,
            stage="mid_train",
            step=self.completed_steps,
        )
        return step_metrics

    def _best_config_paths(self) -> list[str]:
        output_dir = Path(self.config.output_dir)
        return [
            str(output_dir / "config.full.yaml"),
            str(output_dir / "config.yaml"),
        ]

    def _save_best_checkpoint(self, eval_result, score: float, stage: str) -> bool:
        if not self._save_best_model_enabled:
            return False
        if score <= self._best_score:
            return False
        if not self._best_state_path or not self._best_metadata_path:
            raise RuntimeError("Best checkpoint paths were not initialized")

        if self.accelerator.is_main_process and os.path.exists(self._best_state_path):
            shutil.rmtree(self._best_state_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self._best_state_path, safe_serialization=True)
        self.accelerator.wait_for_everyone()

        self._best_score = float(score)
        self._best_step = int(self.completed_steps)

        if self.accelerator.is_main_process:
            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            metadata = {
                "best_step": self._best_step,
                "best_score": self._best_score,
                "score_name": "mid_train_macro_mean_reward",
                "stage": stage,
                "eval_result_path": getattr(eval_result, "path", None),
                "aggregate": eval_result.aggregate,
                "per_latency_scores": {
                    key: float(metrics.get("mean_reward", 0.0))
                    for key, metrics in eval_result.per_latency.items()
                },
            }
            with open(self._best_metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            self.accelerator.print(
                f"✅ New best checkpoint saved at {self._best_state_path} "
                f"(step={self._best_step}, score={self._best_score:.6f})"
            )
            self._checkpoint_sync_manager.sync_best_checkpoint(
                state_path=self._best_state_path,
                metadata_path=self._best_metadata_path,
                config_paths=self._best_config_paths(),
            )
        self.accelerator.wait_for_everyone()
        return True

    def _save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        state_checkpoint_path = checkpoint_path + "_state"

        saved_state_path = None
        if self._save_training_state_enabled:
            if self.accelerator.is_main_process and os.path.exists(state_checkpoint_path):
                shutil.rmtree(state_checkpoint_path)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(state_checkpoint_path, safe_serialization=True)
            self.accelerator.wait_for_everyone()
            saved_state_path = state_checkpoint_path
            if self.accelerator.is_main_process:
                self.accelerator.print(f"✅ Full training state saved at {state_checkpoint_path}")

        if self.accelerator.is_main_process:
            model_checkpoint_path = None
            if self._save_pt_file_enabled or self._save_safetensors_file_enabled:
                state_dict = self.accelerator.get_state_dict(self.model)

            if self._save_pt_file_enabled:
                model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"
                torch.save(state_dict, model_checkpoint_path)

            if self._save_safetensors_file_enabled:
                from safetensors.torch import save_file

                safetensors_state_dict = {
                    name: tensor.detach().to(torch.bfloat16) if torch.is_floating_point(tensor) else tensor.detach()
                    for name, tensor in state_dict.items()
                }
                model_checkpoint_path = checkpoint_path + "_model.safetensors"
                save_file(safetensors_state_dict, model_checkpoint_path)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            if saved_state_path is not None:
                self.accelerator.print(f"✅ Checkpoint state saved at {saved_state_path}")
            if model_checkpoint_path is not None:
                self.accelerator.print(f"✅ Model checkpoint saved at {model_checkpoint_path}")
            self._checkpoint_sync_manager.register_local_checkpoint(
                step=self.completed_steps,
                state_path=saved_state_path,
                model_path=model_checkpoint_path,
            )

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_wandb(self, metrics):
        wandb.log({**metrics, "global_step": self.completed_steps})

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group["name"]
                metrics[f"learning_rate/{group_name}"] = last_lrs[i]
            dataset_size = len(self.vla_train_dataloader.dataset)
            metrics["epoch"] = round(
                calculate_epoch_progress(
                    completed_steps=self.completed_steps,
                    total_batch_size=self.total_batch_size,
                    dataset_size=dataset_size,
                ),
                2,
            )
            self._log_wandb(metrics)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    @staticmethod
    def _total_grad_norm(parameters) -> float:
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        device = grads[0].device
        norms = torch.stack([torch.linalg.vector_norm(g.to(device), 2) for g in grads])
        return float(torch.linalg.vector_norm(norms, 2).item())

    def _record_train_loss(self, action_loss, loss_weight) -> None:
        weight = float(loss_weight)
        self._train_loss_sum += float(action_loss.detach().float().item()) * weight
        self._train_loss_weight += weight

    def _finalize_train_loss(self, device) -> float:
        loss_stats = torch.tensor(
            [self._train_loss_sum, self._train_loss_weight],
            device=device,
            dtype=torch.float32,
        )
        if dist.is_initialized():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        self._train_loss_sum = 0.0
        self._train_loss_weight = 0.0
        return float((loss_stats[0] / loss_stats[1]).item())

    @staticmethod
    def _append_rl_games_eval_metrics(step_metrics: Dict[str, float], eval_result, stage: str) -> Dict[str, float]:
        aggregate = eval_result.aggregate
        prefix = f"rl_games_eval/{stage}"
        step_metrics[f"{prefix}/total_episodes"] = float(aggregate.get("total_episodes", 0))
        step_metrics[f"{prefix}/mean_reward"] = float(aggregate.get("mean_reward", 0.0))
        step_metrics[f"{prefix}/mean_length"] = float(aggregate.get("mean_length", 0.0))
        step_metrics[f"{prefix}/std_reward"] = float(aggregate.get("std_reward", 0.0))
        step_metrics[f"{prefix}/std_length"] = float(aggregate.get("std_length", 0.0))
        step_metrics[f"{prefix}/macro_mean_reward"] = float(
            aggregate.get("macro_mean_reward", aggregate.get("mean_reward", 0.0))
        )
        step_metrics[f"{prefix}/macro_mean_length"] = float(
            aggregate.get("macro_mean_length", aggregate.get("mean_length", 0.0))
        )
        step_metrics[f"{prefix}/task_count"] = float(aggregate.get("task_count", 0))

        for key, metrics in eval_result.per_latency.items():
            key_slug = key.replace("/", "__")
            step_metrics[f"{prefix}/{key_slug}/mean_reward"] = float(metrics.get("mean_reward", 0.0))
            step_metrics[f"{prefix}/{key_slug}/mean_length"] = float(metrics.get("mean_length", 0.0))
            step_metrics[f"{prefix}/{key_slug}/std_reward"] = float(metrics.get("std_reward", 0.0))
            step_metrics[f"{prefix}/{key_slug}/std_length"] = float(metrics.get("std_length", 0.0))
            step_metrics[f"{prefix}/{key_slug}/num_episodes"] = float(metrics.get("num_episodes", 0))
        return step_metrics

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _wait_for_dagger_round(self) -> bool:
        """Serve the resident model, then activate the controller's next mixture."""
        continue_path = self._dagger_control_dir / f"step_{self.completed_steps}.continue.json"
        if continue_path.is_file():
            command = json.loads(continue_path.read_text(encoding="utf-8"))
        else:
            # latency_bench is an optional integration for upstream StarVLA training.
            from latency_bench.integrations.starvla_resident_dagger import serve_resident_inference
            from latency_bench.policy.starvla import build_live_starvla_policy

            resident_config = json.loads(
                (self._dagger_control_dir / "resident_policy.json").read_text(encoding="utf-8")
            )
            model_config = OmegaConf.to_container(
                self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config,
                resolve=True,
            )
            was_training = self.model.training
            self.model.eval()
            unwrapped = self.accelerator.unwrap_model(self.model)
            policy = build_live_starvla_policy(
                framework=unwrapped,
                model_cfg=model_config,
                config=resident_config,
                action_resolver=None,
            )
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    command = serve_resident_inference(
                        policy=policy,
                        control_dir=self._dagger_control_dir,
                        socket_dir=Path(resident_config["socket_dir"]),
                        step=self.completed_steps,
                        rank=int(self.accelerator.process_index),
                        world_size=int(self.accelerator.num_processes),
                        stop_requested=self._refresh_graceful_stop_requested,
                    )
            finally:
                reset = getattr(unwrapped, "reset_memory", None)
                if callable(reset):
                    reset()
                if was_training:
                    self.model.train()

        if command is None:
            self._mark_graceful_stop_requested("dagger_round_wait")
            return False

        self.vla_train_dataloader.dataset.refresh_custom_mixture()
        self._create_data_iterators()
        _distributed_barrier()
        if self.accelerator.is_main_process:
            self._log_wandb(command["metrics"])
            logger.info("DAgger round %s resumed with refreshed data", self.completed_steps)
        return True

    def _mark_graceful_stop_requested(self, reason: str) -> None:
        self.graceful_stop_requested = True
        if self._graceful_stop_file is not None and not self._graceful_stop_file.exists():
            try:
                self._graceful_stop_file.parent.mkdir(parents=True, exist_ok=True)
                self._graceful_stop_file.write_text(f"reason={reason}\ntime={time.time():.6f}\n", encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to write graceful stop file %s: %s", self._graceful_stop_file, exc)
        if self.accelerator.is_main_process and not self._graceful_stop_logged:
            logger.info("Graceful stop requested (%s); stopping at the next safe training boundary.", reason)
            self._graceful_stop_logged = True

    def _refresh_graceful_stop_requested(self) -> bool:
        if _GRACEFUL_STOP_REQUESTED:
            signal_name = (
                signal.Signals(_GRACEFUL_STOP_SIGNAL).name
                if _GRACEFUL_STOP_SIGNAL is not None
                else "signal"
            )
            self._mark_graceful_stop_requested(signal_name)
        if self._graceful_stop_file is not None and self._graceful_stop_file.exists():
            self._mark_graceful_stop_requested(f"stop_file:{self._graceful_stop_file}")
        return self.graceful_stop_requested

    def _save_interrupt_checkpoint(self) -> None:
        if self._interrupt_checkpoint_saved or self.completed_steps <= 0:
            return
        self._interrupt_checkpoint_saved = True
        if self.accelerator.is_main_process:
            logger.info("Saving interrupt checkpoint at step %s", self.completed_steps)
        self._save_checkpoint()

    def _shutdown_dataloader_workers(self) -> None:
        seen: set[int] = set()

        def shutdown(obj) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            shutdown_fn = getattr(obj, "_shutdown_workers", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception as exc:
                    logger.warning("Failed to shutdown dataloader workers for %s: %s", type(obj).__name__, exc)
            for attr in ("_iterator", "dataloader", "_dataloader", "base_dataloader", "_loader"):
                shutdown(getattr(obj, attr, None))

        shutdown(getattr(self, "vla_iter", None))
        shutdown(self.vla_train_dataloader)
        shutdown(self.vla_eval_dataloader)
        shutdown(self._action_cc_f1_frame_loader)

    def cleanup_runtime(self) -> None:
        self._shutdown_dataloader_workers()
        unwrapped = None
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
        except Exception:
            unwrapped = self.model
        reset = getattr(unwrapped, "reset_memory", None)
        if callable(reset):
            try:
                reset()
            except Exception as exc:
                logger.warning("Failed to reset model memory during cleanup: %s", exc)

    def _profile_timing_enabled(self) -> bool:
        return self.config.trainer.profile_timing.enabled

    def _profile_timing_log_interval(self) -> int:
        return self.config.trainer.profile_timing.log_interval

    def _profile_timing_should_log(self, step: int | None = None) -> bool:
        if not self._profile_timing_enabled():
            return False
        step = self.completed_steps if step is None else step
        return step % self._profile_timing_log_interval() == 0

    def _profile_sync(self) -> None:
        if self._profile_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _profile_start(self) -> float:
        self._profile_sync()
        return time.perf_counter()

    def _profile_elapsed(self, start: float) -> float:
        self._profile_sync()
        return time.perf_counter() - start

    def _latency_curriculum_cfg(self):
        vla_data = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        if vla_data is None:
            return None
        cfg = getattr(vla_data, "latency_curriculum", None)
        if cfg is None or not _as_bool(getattr(cfg, "enabled", False), default=False):
            return None
        return cfg

    def _distribution_curriculum_phase_steps(self, cfg, num_phases: int) -> list[int]:
        explicit = _optional_int_list(getattr(cfg, "phase_steps", None))
        if explicit:
            if len(explicit) != num_phases:
                raise ValueError(
                    "datasets.vla_data.latency_curriculum.phase_steps must have one value per phase "
                    f"(got {len(explicit)} steps for {num_phases} phases)."
                )
            if any(step <= 0 for step in explicit):
                raise ValueError("latency_curriculum.phase_steps values must be positive.")
            return explicit
        total_steps = int(getattr(self.config.trainer, "max_train_steps"))
        base = total_steps // num_phases
        remainder = total_steps % num_phases
        return [base + (1 if idx < remainder else 0) for idx in range(num_phases)]

    def _latency_curriculum_phase_steps(self, cfg, latencies: list[int]) -> list[int]:
        explicit = _optional_int_list(getattr(cfg, "phase_steps", None))
        if explicit:
            if len(explicit) != len(latencies):
                raise ValueError(
                    "datasets.vla_data.latency_curriculum.phase_steps must have one value per latency "
                    f"(got {len(explicit)} steps for {len(latencies)} latencies)."
                )
            if any(step <= 0 for step in explicit):
                raise ValueError("latency_curriculum.phase_steps values must be positive.")
            return explicit

        total_steps = int(getattr(self.config.trainer, "max_train_steps"))
        if total_steps < len(latencies):
            raise ValueError(
                "trainer.max_train_steps must be at least the number of latency curriculum phases "
                f"when phase_steps is not set (got max_train_steps={total_steps}, latencies={latencies})."
            )
        base = total_steps // len(latencies)
        remainder = total_steps % len(latencies)
        return [base + (1 if idx < remainder else 0) for idx in range(len(latencies))]

    def _quota_curriculum_plan(self) -> list[dict[str, Any]]:
        cfg = self._latency_curriculum_cfg()
        if cfg is None:
            return []
        plan = getattr(cfg, "computed_plan", None)
        if plan in (None, ""):
            return []
        if isinstance(plan, list):
            return plan
        return list(plan)

    def _latency_curriculum_phase_for_step(
        self,
        step: int,
    ) -> tuple[int, list[int], dict[int, float] | None, dict[int, float] | None, str] | None:
        cfg = self._latency_curriculum_cfg()
        if cfg is None:
            return None
        strategy = str(getattr(cfg, "strategy", "exclusive") or "exclusive").strip().lower()
        if strategy == "quota_cumulative":
            plan = self._quota_curriculum_plan()
            if not plan:
                raise ValueError("quota_cumulative requires a computed_plan; did setup derive max_train_steps?")
            phase = plan[-1]
            for candidate in plan:
                if int(step) < int(candidate["end_step"]):
                    phase = candidate
                    break
            quotas = {int(k): float(v) for k, v in dict(phase["quotas"]).items()}
            quota_starts = {int(k): float(v) for k, v in dict(phase.get("quota_starts", {})).items()}
            active = sorted(quotas)
            return int(phase["phase_idx"]), active, quotas, quota_starts, str(phase.get("name", f"phase_{phase['phase_idx']}"))

        if strategy == "curriculum_cumulative_distribution":
            phase_distributions = _parse_phase_distributions(getattr(cfg, "phase_distributions", None))
            if not phase_distributions:
                raise ValueError("curriculum_cumulative_distribution requires phase_distributions to be configured.")
            num_phases = len(phase_distributions)
            phase_steps = self._distribution_curriculum_phase_steps(cfg, num_phases)
            cursor = 0
            phase_idx = num_phases - 1
            for idx, num_steps in enumerate(phase_steps):
                cursor += int(num_steps)
                if int(step) < cursor:
                    phase_idx = idx
                    break
            distribution = phase_distributions[phase_idx]
            active = sorted(distribution)
            return phase_idx, active, distribution, {}, f"phase_{phase_idx}"

        latencies = _optional_int_list(getattr(cfg, "latencies", None))
        if not latencies:
            latencies = _dataset_latency_values(getattr(self.vla_train_dataloader, "dataset", None))
        if not latencies:
            raise ValueError("latency_curriculum.enabled=true but no curriculum latencies were configured or found.")
        latencies = sorted(dict.fromkeys(int(value) for value in latencies))
        if strategy not in {"exclusive", "cumulative"}:
            raise ValueError(
                f"Unsupported latency_curriculum.strategy={strategy!r}; "
                "expected exclusive, cumulative, quota_cumulative, or curriculum_cumulative_distribution."
            )

        phase_steps = self._latency_curriculum_phase_steps(cfg, latencies)
        cursor = 0
        phase_idx = len(latencies) - 1
        for idx, num_steps in enumerate(phase_steps):
            cursor += int(num_steps)
            if int(step) < cursor:
                phase_idx = idx
                break

        if strategy == "exclusive":
            active = [latencies[phase_idx]]
        else:
            active = latencies[: phase_idx + 1]
        return phase_idx, active, None, None, f"phase_{phase_idx}"

    def _latency_curriculum_phase_end_at_step(self, step: int) -> bool:
        cfg = self._latency_curriculum_cfg()
        if cfg is None or int(step) <= 0:
            return False
        strategy = str(getattr(cfg, "strategy", "exclusive") or "exclusive").strip().lower()
        if strategy == "quota_cumulative":
            plan = self._quota_curriculum_plan()
            final_step = int(getattr(self.config.trainer, "max_train_steps"))
            return any(int(phase["end_step"]) == int(step) and int(step) < final_step for phase in plan)

        if strategy == "curriculum_cumulative_distribution":
            phase_distributions = _parse_phase_distributions(getattr(cfg, "phase_distributions", None))
            if not phase_distributions:
                return False
            phase_steps = self._distribution_curriculum_phase_steps(cfg, len(phase_distributions))
            cursor = 0
            final_step = int(getattr(self.config.trainer, "max_train_steps"))
            for num_steps in phase_steps[:-1]:
                cursor += int(num_steps)
                if cursor == int(step) and int(step) < final_step:
                    return True
            return False

        latencies = _optional_int_list(getattr(cfg, "latencies", None))
        if not latencies:
            latencies = _dataset_latency_values(getattr(self.vla_train_dataloader, "dataset", None))
        if not latencies:
            return False
        phase_steps = self._latency_curriculum_phase_steps(cfg, sorted(dict.fromkeys(int(value) for value in latencies)))
        cursor = 0
        final_step = int(getattr(self.config.trainer, "max_train_steps"))
        for num_steps in phase_steps[:-1]:
            cursor += int(num_steps)
            if cursor == int(step) and int(step) < final_step:
                return True
        return False

    def _apply_latency_curriculum(self, *, force: bool = False) -> dict[str, float]:
        phase = self._latency_curriculum_phase_for_step(self.completed_steps)
        if phase is None:
            return {}
        phase_idx, active_latencies, active_quotas, quota_starts, phase_name = phase
        active_key = (
            phase_idx,
            tuple(active_latencies),
            tuple(sorted((active_quotas or {}).items())),
            tuple(sorted((quota_starts or {}).items())),
        )
        if not force and active_key == self._latency_curriculum_phase:
            return {}

        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        if dataset is None:
            raise ValueError("latency curriculum requires a train dataset.")
        if active_quotas is not None:
            if not callable(getattr(dataset, "set_latency_quota_filter", None)):
                raise ValueError("latency curriculum with quotas requires a train dataset with set_latency_quota_filter(...).")
            dataset.set_latency_quota_filter(active_quotas, quota_starts=quota_starts)
        else:
            if not callable(getattr(dataset, "set_active_latency_filter", None)):
                raise ValueError("latency curriculum requires a train dataset with set_active_latency_filter(...).")
            dataset.set_active_latency_filter(active_latencies)
        self._latency_curriculum_phase = active_key
        if hasattr(self.vla_train_dataloader, "sampler") and callable(getattr(self.vla_train_dataloader.sampler, "set_epoch", None)):
            self.vla_train_dataloader.sampler.set_epoch(phase_idx)
        self._create_data_iterators()
        if self.accelerator.is_main_process:
            logger.info(
                "Latency curriculum phase %s (%s) at step %s: active_latencies=%s, active_quotas=%s, dataset_size=%s",
                phase_idx,
                phase_name,
                self.completed_steps,
                active_latencies,
                active_quotas,
                len(dataset),
            )
        metrics = {
            "latency_curriculum/phase": float(phase_idx),
            "latency_curriculum/active_count": float(len(active_latencies)),
            "latency_curriculum/max_active_latency": float(max(active_latencies)),
            "latency_curriculum/dataset_size": float(len(dataset)),
        }
        if active_quotas is not None:
            for latency, quota in active_quotas.items():
                metrics[f"latency_curriculum/quota/latency_{latency}"] = float(quota)
                if quota_starts and latency in quota_starts:
                    metrics[f"latency_curriculum/quota_start/latency_{latency}"] = float(quota_starts[latency])
            for phase_info in self._quota_curriculum_plan():
                if int(phase_info["phase_idx"]) == int(phase_idx):
                    for latency, rows in dict(phase_info.get("rows_by_latency", {})).items():
                        metrics[f"latency_curriculum/rows/latency_{latency}"] = float(rows)
                    metrics["latency_curriculum/phase_optimizer_steps"] = float(phase_info.get("steps", 0))
                    break
        return metrics

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self.model.train()
        self._log_training_config()
        self._create_data_iterators()
        self._apply_latency_curriculum(force=True)
        # DAgger pauses at round boundaries without shortening the LR schedule.
        stop_step = (
            self.config.trainer.stop_after_steps
            if "stop_after_steps" in self.config.trainer
            else self.config.trainer.max_train_steps
        )
        progress_bar = tqdm(
            total=stop_step,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        try:
            save_interval = int(self.config.trainer.save_interval)
            resume_continue_path = (
                self._dagger_control_dir / f"step_{self.completed_steps}.continue.json"
                if self._dagger_control_dir is not None
                else None
            )
            if (
                resume_continue_path is not None
                and self.completed_steps > 0
                and self.completed_steps < stop_step
                and self.completed_steps % save_interval == 0
                and not resume_continue_path.is_file()
                and not self._wait_for_dagger_round()
            ):
                return
            while self.completed_steps < stop_step:
                if self._refresh_graceful_stop_requested():
                    self._save_interrupt_checkpoint()
                    break

                profile_enabled = self._profile_timing_enabled()
                t_profile_data = self._profile_start() if profile_enabled else None
                t_start_data = time.perf_counter()
                batch_vla = self._get_next_batch()
                t_end_data = time.perf_counter()
                profile_data_seconds = self._profile_elapsed(t_profile_data) if profile_enabled else None

                t_profile_model = self._profile_start() if profile_enabled else None
                t_start_model = time.perf_counter()
                step_metrics = self._train_step(batch_vla)
                t_end_model = time.perf_counter()
                profile_model_seconds = self._profile_elapsed(t_profile_model) if profile_enabled else None

                optimizer_stepped = bool(step_metrics.pop("_optimizer_step"))
                stop_requested = False
                ran_mid_train_rl_eval = False
                if optimizer_stepped:
                    progress_bar.update(1)
                    self.completed_steps += 1
                    stop_requested = self._refresh_graceful_stop_requested()
                    phase_ended = self._latency_curriculum_phase_end_at_step(self.completed_steps)
                    curriculum_cfg = self._latency_curriculum_cfg()
                    if not stop_requested:
                        if (
                            phase_ended
                            and curriculum_cfg is not None
                            and _as_bool(getattr(curriculum_cfg, "eval_at_phase_end", False), default=False)
                        ):
                            if self.accelerator.is_main_process:
                                logger.info("Starting phase-end RL-games mid-train eval at step %s", self.completed_steps)
                            step_metrics = self._run_mid_train_rl_games_eval(step_metrics)
                            ran_mid_train_rl_eval = True
                            if self.accelerator.is_main_process:
                                logger.info("Finished phase-end RL-games mid-train eval at step %s", self.completed_steps)
                        if (
                            phase_ended
                            and curriculum_cfg is not None
                            and _as_bool(getattr(curriculum_cfg, "save_at_phase_end", False), default=False)
                            and self._save_periodic_checkpoints_enabled()
                        ):
                            self._save_checkpoint()
                        curriculum_metrics = self._apply_latency_curriculum()
                        if curriculum_metrics:
                            step_metrics.update(curriculum_metrics)
                    if self._profile_timing_should_log():
                        step_metrics["timing/dataloader_next"] = profile_data_seconds
                        step_metrics["timing/train_step_total"] = profile_model_seconds
                        if "batch/size" in step_metrics:
                            step_metrics["throughput/samples_per_sec"] = step_metrics["batch/size"] / profile_model_seconds
                        if "batch/effective_tokens" in step_metrics:
                            step_metrics["throughput/effective_tokens_per_sec"] = (
                                step_metrics["batch/effective_tokens"] / profile_model_seconds
                            )

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

                gradients_synced = optimizer_stepped
                if gradients_synced and not stop_requested:
                    if should_run_step_interval_event(
                        completed_steps=self.completed_steps,
                        interval=self.config.trainer.eval_interval,
                        gradients_synced=gradients_synced,
                    ):
                        if self.accelerator.is_main_process:
                            logger.info("Starting held-out action loss eval at step %s", self.completed_steps)
                        t_profile_eval = self._profile_start()
                        step_metrics = self.eval_action_loss(step_metrics)
                        if self._profile_timing_should_log():
                            step_metrics["timing/eval_action_loss_total"] = self._profile_elapsed(t_profile_eval)
                        if self.accelerator.is_main_process:
                            logger.info("Finished held-out action loss eval at step %s", self.completed_steps)
                    action_classification_interval = getattr(
                        self.config.trainer,
                        "eval_action_classification_interval",
                        None,
                    )
                    if action_classification_interval is None:
                        action_classification_interval = self.config.trainer.eval_interval
                    eval_action_classification_enabled = _as_bool(
                        getattr(self.config.trainer, "eval_action_classification", True),
                        default=True,
                    )
                    if eval_action_classification_enabled and should_run_step_interval_event(
                        completed_steps=self.completed_steps,
                        interval=action_classification_interval,
                        gradients_synced=gradients_synced,
                    ):
                        if self.accelerator.is_main_process:
                            logger.info("Starting held-out action classification eval at step %s", self.completed_steps)
                        t_profile_eval = self._profile_start()
                        step_metrics = self.eval_action_cc_f1(step_metrics)
                        if self._profile_timing_should_log():
                            step_metrics["timing/eval_action_classification_total"] = self._profile_elapsed(t_profile_eval)
                        if self.accelerator.is_main_process:
                            logger.info("Finished held-out action classification eval at step %s", self.completed_steps)
                    if self._rl_games_eval_runner is not None:
                        eval_every = self._rl_games_eval_runner.interval_steps()
                        if eval_every > 0 and should_run_step_interval_event(
                            completed_steps=self.completed_steps,
                            interval=eval_every,
                            gradients_synced=gradients_synced,
                        ) and not ran_mid_train_rl_eval:
                            step_metrics = self._run_mid_train_rl_games_eval(step_metrics)
                elif gradients_synced and stop_requested and self.accelerator.is_main_process:
                    logger.info("Skipping eval at step %s due to graceful stop", self.completed_steps)

                step_metrics["timing/data"] = t_end_data - t_start_data
                step_metrics["timing/model"] = t_end_model - t_start_model
                if stop_requested:
                    step_metrics["train/graceful_stop_requested"] = 1.0
                if should_run_step_interval_event(
                    completed_steps=self.completed_steps,
                    interval=self.config.trainer.logging_frequency,
                    gradients_synced=gradients_synced,
                ):
                    t_profile_log = self._profile_start() if self._profile_timing_should_log() else None
                    self._log_metrics(step_metrics)
                    if self._profile_timing_should_log() and self.accelerator.is_main_process:
                        self._log_wandb({"timing/log_metrics_total": self._profile_elapsed(t_profile_log)})

                if stop_requested:
                    self._save_interrupt_checkpoint()
                    break

                if self._save_periodic_checkpoints_enabled() and should_run_optional_step_interval_event(
                    completed_steps=self.completed_steps,
                    interval=self.config.trainer.save_interval,
                    gradients_synced=gradients_synced,
                ):
                    t_profile_checkpoint = self._profile_start() if self._profile_timing_should_log() else None
                    self._save_checkpoint()
                    if self._profile_timing_should_log() and self.accelerator.is_main_process:
                        self._log_wandb({"timing/checkpoint_total": self._profile_elapsed(t_profile_checkpoint)})
                    if (
                        self._dagger_control_dir is not None
                        and self.completed_steps < self.config.trainer.max_train_steps
                        and not self._wait_for_dagger_round()
                    ):
                        break

                if self.completed_steps >= stop_step:
                    break
        finally:
            progress_bar.close()
            if self.graceful_stop_requested:
                self.cleanup_runtime()

        if self.graceful_stop_requested:
            if self.accelerator.is_main_process:
                logger.info("Training stopped gracefully at step %s; skipping final/post-train eval.", self.completed_steps)
            return

        self._finalize_training()

    def eval_action_loss(self, step_metrics: dict = None) -> dict:
        """Compute held-out action loss over the validation LeRobot dataset."""
        step_metrics = step_metrics or {}
        if self._refresh_graceful_stop_requested():
            if self.accelerator.is_main_process:
                logger.info("Skipping held-out action loss eval at step %s due to graceful stop", self.completed_steps)
            return step_metrics
        if self.vla_eval_dataloader is None:
            return step_metrics

        was_training = self.model.training
        self.model.eval()
        eval_start = time.perf_counter()
        total_loss_sum = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
        total_loss_count = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
        latency_loss_sums: dict[int, torch.Tensor] = {}
        latency_loss_counts: dict[int, torch.Tensor] = {}
        task_loss_sums: dict[str, torch.Tensor] = {}
        task_loss_counts: dict[str, torch.Tensor] = {}
        task_latency_loss_sums: dict[tuple[str, int], torch.Tensor] = {}
        task_latency_loss_counts: dict[tuple[str, int], torch.Tensor] = {}
        num_batches = int(getattr(self.config.trainer, "eval_num_batches", 20))
        per_latency_batches = _optional_positive_int(getattr(self.config.trainer, "per_latency_eval_num_batches", None))
        batch_size = int(self.config.datasets.vla_data.per_device_batch_size)
        per_latency_sample_budget = per_latency_batches * batch_size if per_latency_batches is not None else None
        seen_latency_counts: dict[int, int] = {}
        seen_task_latency_counts: dict[tuple[str, int], int] = {}
        eval_task = str(getattr(getattr(self.config, "rl_games", None), "task", "") or "").lower()
        cross_task_mode = eval_task == "cross_task"
        expected_latencies = (
            _dataset_latency_values(getattr(self.vla_eval_dataloader, "dataset", None))
            if per_latency_sample_budget is not None
            else []
        )
        expected_task_latencies = (
            [
                (task_name, latency)
                for task_name, latencies in _dataset_task_latency_values(
                    getattr(self.vla_eval_dataloader, "dataset", None)
                ).items()
                for latency in latencies
            ]
            if cross_task_mode and per_latency_sample_budget is not None
            else []
        )
        use_per_latency_budget = per_latency_sample_budget is not None and bool(expected_latencies)
        with torch.no_grad():
            for batch_idx, batch_vla in enumerate(self.vla_eval_dataloader):
                if not use_per_latency_budget and batch_idx >= num_batches:
                    break
                batch_examples = list(batch_vla)
                grouped = _group_examples_by_task_latency(batch_examples)
                for (task_name, latency), examples in grouped.items():
                    if latency is not None and use_per_latency_budget:
                        if cross_task_mode and task_name is not None:
                            task_latency_key = (str(task_name), int(latency))
                            remaining = per_latency_sample_budget - seen_task_latency_counts.get(task_latency_key, 0)
                        else:
                            task_latency_key = None
                            remaining = per_latency_sample_budget - seen_latency_counts.get(int(latency), 0)
                        if remaining <= 0:
                            continue
                        examples = examples[:remaining]
                        if task_latency_key is not None:
                            seen_task_latency_counts[task_latency_key] = (
                                seen_task_latency_counts.get(task_latency_key, 0) + len(examples)
                            )
                        else:
                            seen_latency_counts[int(latency)] = seen_latency_counts.get(int(latency), 0) + len(examples)
                    if not examples:
                        continue
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output_dict = self.model.forward(examples)
                        action_loss = output_dict["action_loss"].detach().float()
                    weight = torch.tensor(float(len(examples)), device=self.accelerator.device, dtype=torch.float32)
                    total_loss_sum += action_loss * weight
                    total_loss_count += weight
                    if latency is not None:
                        latency_loss_sums.setdefault(
                            int(latency),
                            torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                        )
                        latency_loss_counts.setdefault(
                            int(latency),
                            torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                        )
                        latency_loss_sums[int(latency)] += action_loss * weight
                        latency_loss_counts[int(latency)] += weight
                    if task_name is not None:
                        task_name = str(task_name)
                        task_loss_sums.setdefault(
                            task_name,
                            torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                        )
                        task_loss_counts.setdefault(
                            task_name,
                            torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                        )
                        task_loss_sums[task_name] += action_loss * weight
                        task_loss_counts[task_name] += weight
                        if latency is not None:
                            task_latency_key = (task_name, int(latency))
                            task_latency_loss_sums.setdefault(
                                task_latency_key,
                                torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                            )
                            task_latency_loss_counts.setdefault(
                                task_latency_key,
                                torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                            )
                            task_latency_loss_sums[task_latency_key] += action_loss * weight
                            task_latency_loss_counts[task_latency_key] += weight
                if (
                    use_per_latency_budget
                    and (
                        (
                            cross_task_mode
                            and expected_task_latencies
                            and all(
                                seen_task_latency_counts.get(task_latency, 0) >= per_latency_sample_budget
                                for task_latency in expected_task_latencies
                            )
                        )
                        or (
                            not cross_task_mode
                            and expected_latencies
                            and all(
                                seen_latency_counts.get(latency, 0) >= per_latency_sample_budget
                                for latency in expected_latencies
                            )
                        )
                    )
                ):
                    break

        if was_training:
            self.model.train()

        if dist.is_initialized():
            dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss_count, op=dist.ReduceOp.SUM)
            latency_keys = sorted(latency_loss_sums)
            gathered_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_keys, latency_keys)
            all_latency_keys = sorted({int(key) for keys in gathered_keys for key in keys})
            for latency in all_latency_keys:
                loss_sum = latency_loss_sums.get(
                    latency,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                loss_count = latency_loss_counts.get(
                    latency,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
                latency_loss_sums[latency] = loss_sum
                latency_loss_counts[latency] = loss_count
            task_keys = sorted(task_loss_sums)
            gathered_task_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_task_keys, task_keys)
            all_task_keys = sorted({str(key) for keys in gathered_task_keys for key in keys})
            for task_name in all_task_keys:
                loss_sum = task_loss_sums.get(
                    task_name,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                loss_count = task_loss_counts.get(
                    task_name,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
                task_loss_sums[task_name] = loss_sum
                task_loss_counts[task_name] = loss_count
            task_latency_keys = sorted(task_latency_loss_sums)
            gathered_task_latency_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_task_latency_keys, task_latency_keys)
            all_task_latency_keys = sorted(
                {(str(task_name), int(latency)) for keys in gathered_task_latency_keys for task_name, latency in keys}
            )
            for task_latency_key in all_task_latency_keys:
                loss_sum = task_latency_loss_sums.get(
                    task_latency_key,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                loss_count = task_latency_loss_counts.get(
                    task_latency_key,
                    torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
                )
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
                task_latency_loss_sums[task_latency_key] = loss_sum
                task_latency_loss_counts[task_latency_key] = loss_count

        if self.accelerator.is_main_process and total_loss_count.item() > 0:
            step_metrics["eval/action_loss/seconds"] = time.perf_counter() - eval_start
            step_metrics["eval/action_loss/samples"] = total_loss_count.item()
            step_metrics["eval/loss"] = (total_loss_sum / total_loss_count.clamp_min(1.0)).item()
            for latency in sorted(latency_loss_sums):
                count = latency_loss_counts[latency]
                if count.item() > 0:
                    step_metrics[f"eval/latency_{latency}/loss"] = (
                        latency_loss_sums[latency] / count.clamp_min(1.0)
                    ).item()
            for task_name in sorted(task_loss_sums):
                count = task_loss_counts[task_name]
                if count.item() > 0:
                    step_metrics[f"eval/{task_name}/loss"] = (
                        task_loss_sums[task_name] / count.clamp_min(1.0)
                    ).item()
            for task_name, latency in sorted(task_latency_loss_sums):
                count = task_latency_loss_counts[(task_name, latency)]
                if count.item() > 0:
                    step_metrics[f"eval/{task_name}/latency_{latency}/loss"] = (
                        task_latency_loss_sums[(task_name, latency)] / count.clamp_min(1.0)
                    ).item()

        _distributed_barrier()
        return step_metrics

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Backward-compatible alias for the held-out action loss evaluator."""
        return self.eval_action_loss(step_metrics)

    def eval_action_cc_f1(self, step_metrics: dict = None) -> dict:
        """Component-based Control-Critical F1 over the held-out teacher set.

        Unlike ``eval_action_loss`` (which samples frames randomly via the
        shuffled eval dataloader), CC-F1 needs *contiguous, fully-covered*
        episodes so the ``±K`` temporal-tolerance matching is meaningful. So this
        runs a dedicated sequential pass: it walks each underlying LeRobot
        dataset's ``all_steps`` in order (bypassing the random mixture sampler),
        shards whole episodes across ranks, runs ``predict_action`` (the same
        decode path as rollout), decodes per-frame action *components* and matches
        teacher vs model events per episode. Counts are all-reduced and reduced to
        CC-F1 on the main process. See ``action_cc_f1`` for the metric definition.
        """
        step_metrics = step_metrics or {}
        if self._refresh_graceful_stop_requested():
            if self.accelerator.is_main_process:
                logger.info(
                    "Skipping held-out action classification eval at step %s due to graceful stop",
                    self.completed_steps,
                )
            return step_metrics
        if self.vla_eval_dataloader is None:
            return step_metrics
        task = str(getattr(getattr(self.config, "rl_games", None), "task", "") or "").lower()
        cross_task_mode = task == "cross_task"
        if not cross_task_mode and task not in action_cc_f1.SUPPORTED_TASKS:
            return step_metrics
        if not bool(getattr(self.config.trainer, "eval_action_classification", True)):
            return step_metrics

        # Resolve spec (deadly_corridor depends on its action layout) and tolerance.
        # Read the layout exactly like action_spec._deadly_action_dim (same default)
        # so the decode matches the action dim the model was actually trained with.
        deadly_layout = action_cc_f1.DEADLY_MULTIBINARY_7
        deadly_cfg = getattr(getattr(getattr(self.config, "rl_games", None), "env_eval", None), "deadly", None)
        if deadly_cfg is not None:
            deadly_layout = str(getattr(deadly_cfg, "action_layout", action_cc_f1.DEADLY_MULTIBINARY_7))

        def _spec_for_task(task_name: str):
            return action_cc_f1.get_spec(task_name, deadly_layout)

        def _tolerance_for_task(task_name: str, spec) -> int:
            k = spec.default_k
            if task_name != "flappy":  # flappy stays per-frame (K=0) to match the shipped flap-F1
                override = getattr(self.config.trainer, "cc_f1_tolerance", None)
                if override is not None:
                    k = int(override)
            return k

        if not cross_task_mode:
            spec = _spec_for_task(task)
            k = _tolerance_for_task(task, spec)
        else:
            spec = None
            k = None

        dataset = getattr(self.vla_eval_dataloader, "dataset", None)
        if dataset is None:
            return step_metrics
        single_datasets = list(getattr(dataset, "datasets", None) or [dataset])

        # Enumerate episodes as contiguous same-trajectory runs of all_steps.
        # episodes: list of (task, dataset_index, episode_key, latency, [(flat_idx, base_index), ...]).
        episodes = []
        for dataset_index, ds in enumerate(single_datasets):
            ds_task = str(getattr(ds, "rl_games_task", task) or task)
            if ds_task not in action_cc_f1.SUPPORTED_TASKS:
                continue
            all_steps = getattr(ds, "all_steps", None)
            if not all_steps:
                continue
            tag = getattr(ds, "tag", "ds")
            cur_traj = None
            cur = []
            for flat_idx, step in enumerate(all_steps):
                traj_id, base_idx = step
                if cur and traj_id != cur_traj:
                    episodes.append((ds_task, dataset_index, f"{ds_task}:{tag}:{cur_traj}", _episode_latency(ds, int(cur_traj)), cur))
                    cur = []
                cur_traj = traj_id
                cur.append((int(flat_idx), int(base_idx)))
            if cur:
                episodes.append((ds_task, dataset_index, f"{ds_task}:{tag}:{cur_traj}", _episode_latency(ds, int(cur_traj)), cur))
        if not episodes:
            return step_metrics

        # Shard whole episodes across ranks, bounded by a per-rank frame budget.
        num_procs = int(self.accelerator.num_processes)
        rank = int(self.accelerator.process_index)
        bs = int(self.config.datasets.vla_data.per_device_batch_size)
        per_latency_batches = _optional_positive_int(getattr(self.config.trainer, "per_latency_eval_num_batches", None))
        shared_frame_budget = max(1, int(getattr(self.config.trainer, "eval_num_batches", 20)) * bs)
        per_latency_frame_budget = per_latency_batches * bs if per_latency_batches is not None else None
        assigned = []
        frame_count = 0
        latency_frame_counts: dict[int, int] = {}
        task_latency_frame_counts: dict[tuple[str, int], int] = {}
        expected_latencies = sorted({int(ep[3]) for ep in episodes if ep[3] is not None})
        expected_task_latencies = (
            sorted({(str(ep[0]), int(ep[3])) for ep in episodes if ep[3] is not None})
            if cross_task_mode
            else []
        )
        use_per_latency_budget = per_latency_frame_budget is not None and bool(expected_latencies)
        for ep_idx, ep in enumerate(episodes):
            if ep_idx % num_procs != rank:
                continue
            ep_task = str(ep[0])
            latency = ep[3]
            if latency is not None and use_per_latency_budget:
                latency = int(latency)
                if cross_task_mode:
                    task_latency_key = (ep_task, latency)
                    if task_latency_frame_counts.get(task_latency_key, 0) >= per_latency_frame_budget:
                        continue
                else:
                    if latency_frame_counts.get(latency, 0) >= per_latency_frame_budget:
                        continue
            assigned.append(ep)
            frame_count += len(ep[4])
            if latency is not None and use_per_latency_budget:
                if cross_task_mode:
                    task_latency_key = (ep_task, int(latency))
                    task_latency_frame_counts[task_latency_key] = (
                        task_latency_frame_counts.get(task_latency_key, 0) + len(ep[4])
                    )
                    if all(
                        task_latency_frame_counts.get(value, 0) >= per_latency_frame_budget
                        for value in expected_task_latencies
                    ):
                        break
                else:
                    latency_frame_counts[int(latency)] = latency_frame_counts.get(int(latency), 0) + len(ep[4])
                    if all(
                        latency_frame_counts.get(value, 0) >= per_latency_frame_budget
                        for value in expected_latencies
                    ):
                        break
            elif not use_per_latency_budget and frame_count >= shared_frame_budget:
                break

        was_training = self.model.training
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        counts = action_cc_f1.new_counts(spec) if spec is not None else None

        per_ep = {}  # episode_key -> list of (base_index, teacher_components, model_components)

        def flush_chunk(chunk):
            normalized = unwrapped.predict_action(examples=[c[4] for c in chunk])["normalized_actions"]
            for i, (sample_task, episode_key, base_idx, latency, sample) in enumerate(chunk):
                sample_task = str(sample_task)
                sample_spec = _spec_for_task(sample_task)
                teacher_vec = np.asarray(sample["action"])[0, :]
                model_vec = normalized[i, 0, :]
                entry = per_ep.setdefault(episode_key, {"task": sample_task, "latency": latency, "items": []})
                entry["items"].append((base_idx, sample_spec.comp_fn(teacher_vec), sample_spec.comp_fn(model_vec)))

        frame_records = []
        for ep_task, dataset_index, episode_key, episode_latency_value, frames in assigned:
            for flat_idx, base_idx in frames:
                frame_records.append((
                    int(dataset_index),
                    int(flat_idx),
                    str(episode_key),
                    int(base_idx),
                    str(ep_task),
                    episode_latency_value,
                ))

        eval_num_workers = getattr(self.config.datasets.vla_data, "eval_num_workers", None)
        if eval_num_workers is None:
            eval_num_workers = self.config.datasets.vla_data.num_workers
        eval_num_workers = int(eval_num_workers)
        dataloader_kwargs = build_cpu_only_dataloader_kwargs(
            eval_num_workers,
            pin_memory=self.config.datasets.vla_data.pin_memory,
            persistent_workers=self.config.datasets.vla_data.persistent_workers,
            prefetch_factor=(
                self.config.datasets.vla_data.prefetch_factor
                if "prefetch_factor" in self.config.datasets.vla_data
                else None
            ),
        )
        frame_loader_key = (tuple(frame_records), bs, eval_num_workers)
        if self._action_cc_f1_frame_loader_key != frame_loader_key:
            if self.accelerator.is_main_process:
                logger.info(
                    "Created action CC-F1 frame dataloader with num_workers=%s, persistent_workers=%s, cpu_only_workers=%s",
                    eval_num_workers,
                    dataloader_kwargs.get("persistent_workers", False),
                    "multiprocessing_context" in dataloader_kwargs,
                )
            self._action_cc_f1_frame_loader = DataLoader(
                _ActionCCF1FrameDataset(single_datasets, frame_records),
                batch_size=bs,
                shuffle=False,
                collate_fn=_identity_collate,
                num_workers=eval_num_workers,
                **dataloader_kwargs,
            )
            self._action_cc_f1_frame_loader_key = frame_loader_key
        frame_loader = self._action_cc_f1_frame_loader

        data_wait_seconds = 0.0
        predict_seconds = 0.0
        processed_frames = 0
        with torch.no_grad():
            t_data_start = time.perf_counter()
            for chunk in frame_loader:
                t_data_end = time.perf_counter()
                data_wait_seconds += t_data_end - t_data_start
                t_predict_start = time.perf_counter()
                flush_chunk(chunk)
                predict_seconds += time.perf_counter() - t_predict_start
                processed_frames += len(chunk)
                t_data_start = time.perf_counter()

        if was_training:
            self.model.train()

        latency_counts: dict[int, np.ndarray] = {}
        task_counts: dict[str, np.ndarray] = {}
        task_latency_counts: dict[tuple[str, int], np.ndarray] = {}
        for entry in per_ep.values():
            entry_task = str(entry.get("task", task))
            entry_spec = _spec_for_task(entry_task)
            entry_k = _tolerance_for_task(entry_task, entry_spec)
            items = entry["items"]
            items.sort(key=lambda r: r[0])
            target_counts = task_counts.setdefault(entry_task, action_cc_f1.new_counts(entry_spec)) if cross_task_mode else counts
            action_cc_f1.accumulate_episode(
                entry_spec,
                frames=[r[0] for r in items],
                teacher_comps=[r[1] for r in items],
                model_comps=[r[2] for r in items],
                k=entry_k,
                counts=target_counts,
            )
            latency = entry.get("latency")
            if latency is not None:
                latency = int(latency)
                if cross_task_mode:
                    task_latency_key = (entry_task, latency)
                    task_latency_counts.setdefault(task_latency_key, action_cc_f1.new_counts(entry_spec))
                    target_latency_counts = task_latency_counts[task_latency_key]
                else:
                    latency_counts.setdefault(latency, action_cc_f1.new_counts(entry_spec))
                    target_latency_counts = latency_counts[latency]
                action_cc_f1.accumulate_episode(
                    entry_spec,
                    frames=[r[0] for r in items],
                    teacher_comps=[r[1] for r in items],
                    model_comps=[r[2] for r in items],
                    k=entry_k,
                    counts=target_latency_counts,
                )

        # Counts are small integers (well within float32's exact range); float32
        # keeps NCCL all_reduce broadly compatible across torch builds.
        if not cross_task_mode:
            counts_t = torch.tensor(counts, device=self.accelerator.device, dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
            if self.accelerator.is_main_process:
                step_metrics.update(action_cc_f1.reduce_metrics(spec, counts_t.detach().cpu().numpy()))

            if dist.is_initialized():
                latency_keys = sorted(latency_counts)
                gathered_keys = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_keys, latency_keys)
                all_latency_keys = sorted({int(key) for keys in gathered_keys for key in keys})
            else:
                all_latency_keys = sorted(latency_counts)

            for latency in all_latency_keys:
                latency_counts_t = torch.tensor(
                    latency_counts.get(latency, action_cc_f1.new_counts(spec)),
                    device=self.accelerator.device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
                    dist.all_reduce(latency_counts_t, op=dist.ReduceOp.SUM)
                if self.accelerator.is_main_process:
                    latency_metrics = action_cc_f1.reduce_metrics(spec, latency_counts_t.detach().cpu().numpy())
                    for key, value in latency_metrics.items():
                        if key.startswith("eval/"):
                            key = f"eval/latency_{int(latency)}/{key[len('eval/'):]}"
                        else:
                            key = f"eval/latency_{int(latency)}/{key}"
                        step_metrics[key] = value
        else:
            if dist.is_initialized():
                task_keys = sorted(task_counts)
                gathered_task_keys = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_task_keys, task_keys)
                all_task_keys = sorted({str(key) for keys in gathered_task_keys for key in keys})
            else:
                all_task_keys = sorted(task_counts)

            for task_name in all_task_keys:
                task_spec = _spec_for_task(task_name)
                counts_t = torch.tensor(
                    task_counts.get(task_name, action_cc_f1.new_counts(task_spec)),
                    device=self.accelerator.device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
                    dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
                if self.accelerator.is_main_process:
                    metrics = action_cc_f1.reduce_metrics(task_spec, counts_t.detach().cpu().numpy())
                    for key, value in metrics.items():
                        if key.startswith("eval/"):
                            key = f"eval/{task_name}/{key[len('eval/'):]}"
                        else:
                            key = f"eval/{task_name}/{key}"
                        step_metrics[key] = value

            if dist.is_initialized():
                task_latency_keys = sorted(task_latency_counts)
                gathered_task_latency_keys = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_task_latency_keys, task_latency_keys)
                all_task_latency_keys = sorted(
                    {(str(task_name), int(latency)) for keys in gathered_task_latency_keys for task_name, latency in keys}
                )
            else:
                all_task_latency_keys = sorted(task_latency_counts)

            for task_name, latency in all_task_latency_keys:
                task_spec = _spec_for_task(task_name)
                counts_t = torch.tensor(
                    task_latency_counts.get((task_name, latency), action_cc_f1.new_counts(task_spec)),
                    device=self.accelerator.device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
                    dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
                if self.accelerator.is_main_process:
                    latency_metrics = action_cc_f1.reduce_metrics(task_spec, counts_t.detach().cpu().numpy())
                    for key, value in latency_metrics.items():
                        if key.startswith("eval/"):
                            key = f"eval/{task_name}/latency_{int(latency)}/{key[len('eval/'):]}"
                        else:
                            key = f"eval/{task_name}/latency_{int(latency)}/{key}"
                        step_metrics[key] = value

        data_wait_t = torch.tensor(data_wait_seconds, device=self.accelerator.device, dtype=torch.float32)
        predict_t = torch.tensor(predict_seconds, device=self.accelerator.device, dtype=torch.float32)
        frames_t = torch.tensor(float(processed_frames), device=self.accelerator.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(data_wait_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(predict_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(frames_t, op=dist.ReduceOp.SUM)
        if self.accelerator.is_main_process:
            step_metrics["eval/action_classification/data_wait_seconds"] = data_wait_t.item()
            step_metrics["eval/action_classification/predict_seconds"] = predict_t.item()
            step_metrics["eval/action_classification/frames"] = frames_t.item()

        _distributed_barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        profile_next_step = self.completed_steps + 1
        profile_log = self._profile_timing_should_log(profile_next_step)
        profile_metrics = {}
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            t_forward = self._profile_start() if profile_log else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                loss_weight = output_dict["loss_weight"]
                total_loss = action_loss
            if profile_log:
                profile_metrics["timing/forward"] = self._profile_elapsed(t_forward)
                profile_metrics.update(output_dict["timing"])
                profile_metrics.update(output_dict["batch_stats"])

            t_backward = self._profile_start() if profile_log else None
            self.model.backward(total_loss)
            self._record_train_loss(action_loss, loss_weight)
            if profile_log:
                profile_metrics["timing/backward"] = self._profile_elapsed(t_backward)

            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            t_optimizer = self._profile_start() if profile_log else None
            self.model.step()
            if profile_log:
                profile_metrics["timing/optimizer_step"] = self._profile_elapsed(t_optimizer)

            t_scheduler = self._profile_start() if profile_log else None
            if optimizer_stepped:
                self.lr_scheduler.step()
            if profile_log:
                profile_metrics["timing/lr_scheduler"] = self._profile_elapsed(t_scheduler)

            if not optimizer_stepped:
                return {"_optimizer_step": False}
            grad_norm = self.model.get_global_grad_norm()
            metrics = {
                "train/loss": self._finalize_train_loss(action_loss.device),
                "_optimizer_step": True,
            }
            metrics.update(profile_metrics)
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.detach().float().item()
            metrics["train/grad_norm_pre_clip"] = float(grad_norm)
            return metrics

        with self.accelerator.accumulate(self.model):
            t_forward = self._profile_start() if profile_log else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                loss_weight = output_dict["loss_weight"]
                total_loss = action_loss
            if profile_log:
                profile_metrics["timing/forward"] = self._profile_elapsed(t_forward)
                profile_metrics.update(output_dict["timing"])
                profile_metrics.update(output_dict["batch_stats"])

            t_backward = self._profile_start() if profile_log else None
            self.accelerator.backward(total_loss)
            self._record_train_loss(action_loss, loss_weight)
            if profile_log:
                profile_metrics["timing/backward"] = self._profile_elapsed(t_backward)

            grad_norm = None
            gradients_synced = self.accelerator.sync_gradients
            t_grad = self._profile_start() if profile_log else None
            if gradients_synced and self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            elif gradients_synced:
                grad_norm = self._total_grad_norm(self.model.parameters())
            if profile_log:
                profile_metrics["timing/grad_clip_or_norm"] = self._profile_elapsed(t_grad)

            t_optimizer = self._profile_start() if profile_log else None
            self.optimizer.step()
            if profile_log:
                profile_metrics["timing/optimizer_step"] = self._profile_elapsed(t_optimizer)
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            t_scheduler = self._profile_start() if profile_log else None
            if gradients_synced:
                self.lr_scheduler.step()
            if profile_log:
                profile_metrics["timing/lr_scheduler"] = self._profile_elapsed(t_scheduler)
            t_zero_grad = self._profile_start() if profile_log else None
            self.optimizer.zero_grad()
            if profile_log:
                profile_metrics["timing/zero_grad"] = self._profile_elapsed(t_zero_grad)

        if not gradients_synced:
            return {"_optimizer_step": False}
        metrics = {
            "train/loss": self._finalize_train_loss(action_loss.device),
            "_optimizer_step": True,
        }
        metrics.update(profile_metrics)
        if grad_norm is not None:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.detach().float().item()
            metrics["train/grad_norm_pre_clip"] = float(grad_norm)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        save_interval = int(getattr(self.config.trainer, "save_interval", 0) or 0)
        final_step_already_saved = (
            self._save_periodic_checkpoints_enabled()
            and self.completed_steps > 0
            and save_interval > 0
            and self.completed_steps % save_interval == 0
        )
        if (
            self._save_final_model_enabled
            and self.completed_steps > 0
            and not final_step_already_saved
        ):
            self._save_checkpoint()

        if self.accelerator.is_main_process:
            if self._save_best_model_enabled:
                logger.info(
                    "Training complete. Best checkpoint step=%s score=%s path=%s",
                    self._best_step,
                    self._best_score,
                    self._best_state_path,
                )
            else:
                logger.info(f"Training complete. Final training state is saved under {self.checkpoint_dir}")

        if (
            self._save_best_model_enabled
            and self._best_state_path
            and os.path.isdir(self._best_state_path)
        ):
            self.accelerator.print(
                f"Loading best checkpoint for post-train eval: "
                f"{self._best_state_path} (step={self._best_step}, score={self._best_score:.6f})"
            )
            self._load_checkpoint(self._best_state_path)
        elif self._save_best_model_enabled:
            logger.warning(
                "checkpoint.save_best_model=true but no best_state checkpoint was found; "
                "post-train eval will use the current in-memory model."
            )
        self.accelerator.wait_for_everyone()

        if self._rl_games_eval_runner is not None and self._rl_games_eval_runner.is_enabled(stage="post_train"):
            if self._distributed_rl_games_eval_enabled():
                eval_result = self._run_distributed_rl_games_eval(stage="post_train")
            elif self.accelerator.is_main_process:
                eval_result = self._run_rl_games_eval_with_model_mode(stage="post_train")
            else:
                eval_result = None

            if self.accelerator.is_main_process and eval_result is not None:
                final_metrics = self._append_rl_games_eval_metrics(
                    step_metrics={},
                    eval_result=eval_result,
                    stage="post_train",
                )
                if self._save_best_model_enabled:
                    final_metrics["checkpoint/best_score"] = float(self._best_score)
                    final_metrics["checkpoint/best_step"] = float(self._best_step)
                self._checkpoint_sync_manager.sync_eval_result(
                    eval_path=eval_result.path,
                    stage="post_train",
                    step=self.completed_steps,
                )
                self._log_wandb(final_metrics)

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def _run_eval_batch_benchmark(trainer) -> None:
    """Benchmark rl_games (latency_bench) eval wall-clock + VRAM across batch sizes.

    Gated by env var LB_EVAL_BENCH=1. The plan is a comma list of
    ``parallel:episodes`` pairs in LB_EVAL_BENCH_PLAN (default
    ``1:4,5:20,10:20,20:20``). Model is loaded ONCE; every config runs against
    the same live model so the comparison is apples-to-apples. The serial 1:4
    row is meant to be linearly extrapolated to 20 eps (4x serial is too slow).
    """
    import os
    import time

    plan_str = os.environ.get("LB_EVAL_BENCH_PLAN", "1:4,5:20,10:20,20:20")
    plan = []
    for item in plan_str.split(","):
        p, e = item.strip().split(":")
        plan.append((int(p), int(e)))

    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception:
        pynvml = None
        _nvml_handle = None

    def _nvml_used_mb():
        if _nvml_handle is None:
            return None
        return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1024 / 1024

    runner = trainer._rl_games_eval_runner
    model = trainer.accelerator.unwrap_model(trainer.model)

    def _set(parallel, episodes):
        # The trainer cfg is an access-tracked wrapper that OmegaConf.update can't
        # touch, so override the runner's per-stage getters directly instead.
        runner._eval_parallel_envs = lambda *a, **k: int(parallel)
        runner._num_episodes = lambda *a, **k: int(episodes)

    import threading

    def _timed_run(parallel, episodes, label):
        _set(parallel, episodes)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        nvml_peak = {"v": 0.0}
        stop = threading.Event()

        def _sample():
            while not stop.is_set():
                u = _nvml_used_mb()
                if u is not None and u > nvml_peak["v"]:
                    nvml_peak["v"] = u
                time.sleep(0.1)

        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        t0 = time.perf_counter()
        result = runner.run(model=model, step=0, stage="mid_train", save=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        stop.set()
        t.join(timeout=1.0)
        torch_peak = torch.cuda.max_memory_reserved() / 1024 / 1024
        mean_r = result.aggregate.get("mean_reward")
        logger.info(
            "[LB_EVAL_BENCH] %s | parallel=%d eps=%d | wall=%.1fs | torch_reserved_peak=%.0fMB | "
            "nvml_used_peak=%sMB | mean_reward=%s",
            label, parallel, episodes, dt, torch_peak,
            f"{nvml_peak['v']:.0f}" if _nvml_handle is not None else "n/a", mean_r,
        )
        return {"parallel": parallel, "episodes": episodes, "wall_s": dt,
                "torch_peak_mb": torch_peak, "nvml_peak_mb": nvml_peak["v"], "mean_reward": mean_r}

    logger.info("[LB_EVAL_BENCH] warmup (parallel=1 eps=2, untimed) to trigger CUDA init/autotune ...")
    _set(1, 2)
    runner.run(model=model, step=0, stage="mid_train", save=False)
    torch.cuda.synchronize()

    rows = []
    for parallel, episodes in plan:
        rows.append(_timed_run(parallel, episodes, label="bench"))

    # Summary table with serial-extrapolation + speedup vs the 1:* serial row.
    serial = next((r for r in rows if r["parallel"] == 1), None)
    serial_per_ep = (serial["wall_s"] / serial["episodes"]) if serial else None
    logger.info("[LB_EVAL_BENCH] ==== SUMMARY ====")
    logger.info("[LB_EVAL_BENCH] parallel | eps | wall_s | s/ep | extrap_20ep_s | speedup_vs_serial20 | torch_peak_MB | nvml_peak_MB | mean_reward")
    serial20 = serial_per_ep * 20 if serial_per_ep else None
    for r in rows:
        s_per_ep = r["wall_s"] / r["episodes"]
        extrap20 = s_per_ep * 20
        speedup = (serial20 / extrap20) if serial20 else float("nan")
        logger.info(
            "[LB_EVAL_BENCH] %8d | %3d | %6.1f | %5.2f | %12.1f | %18.2fx | %12.0f | %11.0f | %s",
            r["parallel"], r["episodes"], r["wall_s"], s_per_ep, extrap20, speedup,
            r["torch_peak_mb"], r["nvml_peak_mb"], r["mean_reward"],
        )
    if serial20 is not None:
        logger.info("[LB_EVAL_BENCH] serial 20-ep estimate (1:4 x5) = %.1fs", serial20)


def main(cfg) -> None:
    previous_signal_handlers = _install_graceful_stop_handlers()
    trainer = None
    normal_completion = False
    _pin_cuda_device_from_local_rank()
    try:
        logger.info("VLA Training :: Warming Up")
        if hasattr(cfg, "rl_games"):
            login_training_services(cfg, workspace_dir=getattr(cfg, "workspace_dir", None))
            validate_rl_games_config(cfg)
        # Keep the obs window tied to the KV-memory rollout length before the config is
        # wrapped/consumed by the dataloader and framework.
        sync_kv_memory_obs_window(cfg)
        apply_model_alias(cfg)
        apply_action_spec(cfg)

        cfg = wrap_config(cfg)
        logger.info("✅ Configuration wrapped for access tracking")

        output_dir = setup_directories(cfg=cfg)
        vla = build_initial_framework(cfg)
        vla = _preload_model_checkpoint_before_accelerator(cfg=cfg, model=vla)

        distributed_backend = str(getattr(cfg.trainer, "distributed_backend", "deepspeed")).lower()
        accelerator = build_accelerator(cfg, use_deepspeed=distributed_backend == "deepspeed")

        vla_train_dataloader, vla_eval_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
        _configure_quota_cumulative_training_steps(cfg=cfg, dataloader=vla_train_dataloader, accelerator=accelerator)
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

        trainer = VLATrainer(
            cfg=cfg,
            model=vla,
            vla_train_dataloader=vla_train_dataloader,
            vla_eval_dataloader=vla_eval_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )

        trainer.prepare_training()

        import os as _os
        if _os.environ.get("LB_EVAL_BENCH"):
            _run_eval_batch_benchmark(trainer)
            logger.info("[LB_EVAL_BENCH] done; skipping training.")
            normal_completion = True
            return

        trainer.train()
        normal_completion = not bool(getattr(trainer, "graceful_stop_requested", False))

        if normal_completion:
            logger.info("... and that's all, folks!")
    finally:
        if trainer is not None:
            trainer.cleanup_runtime()
        if sys.exc_info()[0] is None:
            _destroy_distributed_process_group(use_barrier=normal_completion)
        _restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
