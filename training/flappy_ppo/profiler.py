from __future__ import annotations

import csv
import os
import time
from contextlib import nullcontext
from typing import Dict

import torch


class UpdateProfiler:
    def __init__(self, enabled: bool, sync_cuda: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda)
        self.device = device
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def reset(self) -> None:
        self.times.clear()
        self.counts.clear()

    def synchronize(self) -> None:
        if self.enabled and self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def time(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _ProfileTimer(self, name)

    def add_count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.counts[name] = self.counts.get(name, 0) + int(amount)


class _ProfileTimer:
    def __init__(self, profiler: UpdateProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.profiler.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.synchronize()
        elapsed = time.perf_counter() - self.start
        self.profiler.times[self.name] = self.profiler.times.get(self.name, 0.0) + elapsed
        return False


PROFILE_TIME_KEYS = [
    "rollout_total_s",
    "action_infer_s",
    "action_cpu_copy_s",
    "env_step_s",
    "reward_done_h2d_s",
    "render_obs_s",
    "state_batch_pack_s",
    "direct_obs_s",
    "full_render_s",
    "resize_s",
    "stack_update_s",
    "final_obs_s",
    "gae_s",
    "flatten_s",
    "ppo_update_s",
    "minibatch_forward_loss_s",
    "backward_s",
    "optimizer_s",
]

PROFILE_FIELDS = [
    "update",
    "global_step",
    "included",
    "device",
    "renderer",
    "observation_render_mode",
    "resize_backend",
    "n_envs",
    "n_steps",
    "rollout_size",
    "batch_size",
    "n_epochs",
    "num_minibatches",
    "total_s",
    "end_to_end_sps",
    "rollout_sps",
    "update_samples_s",
    "rollout_pct",
    "ppo_update_pct",
    "gae_pct",
    "avg_minibatch_ms",
] + PROFILE_TIME_KEYS


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def init_profile_csv(path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writeheader()


def append_profile_csv(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writerow({key: row.get(key, "") for key in PROFILE_FIELDS})


def build_profile_row(
    args,
    device: torch.device,
    profiler: UpdateProfiler,
    update: int,
    global_step: int,
    total_s: float,
    rollout_size: int,
    minibatch_size: int,
    included: bool,
    end_to_end_sps: int,
) -> Dict[str, object]:
    times = profiler.times
    num_minibatches = int(profiler.counts.get("num_minibatches", 0))
    minibatch_work_s = (
        times.get("minibatch_forward_loss_s", 0.0)
        + times.get("backward_s", 0.0)
        + times.get("optimizer_s", 0.0)
    )
    row: Dict[str, object] = {
        "update": int(update),
        "global_step": int(global_step),
        "included": int(included),
        "device": str(device),
        "renderer": args.renderer,
        "observation_render_mode": args.observation_render_mode,
        "resize_backend": args.resize_backend,
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "rollout_size": int(rollout_size),
        "batch_size": int(minibatch_size),
        "n_epochs": int(args.n_epochs),
        "num_minibatches": num_minibatches,
        "total_s": total_s,
        "end_to_end_sps": int(end_to_end_sps),
        "rollout_sps": safe_div(rollout_size, times.get("rollout_total_s", 0.0)),
        "update_samples_s": safe_div(rollout_size, times.get("ppo_update_s", 0.0)),
        "rollout_pct": 100.0 * safe_div(times.get("rollout_total_s", 0.0), total_s),
        "ppo_update_pct": 100.0 * safe_div(times.get("ppo_update_s", 0.0), total_s),
        "gae_pct": 100.0 * safe_div(times.get("gae_s", 0.0), total_s),
        "avg_minibatch_ms": 1000.0 * safe_div(minibatch_work_s, num_minibatches),
    }
    for key in PROFILE_TIME_KEYS:
        row[key] = times.get(key, 0.0)
    return row


def print_profile_row(row: Dict[str, object]) -> None:
    print(
        "[profile] "
        f"update={row['update']} included={row['included']} "
        f"total_s={float(row['total_s']):.3f} "
        f"rollout_s={float(row['rollout_total_s']):.3f} "
        f"ppo_s={float(row['ppo_update_s']):.3f} "
        f"gae_s={float(row['gae_s']):.3f} "
        f"rollout_pct={float(row['rollout_pct']):.1f}% "
        f"ppo_pct={float(row['ppo_update_pct']):.1f}% "
        f"env_step_s={float(row['env_step_s']):.3f} "
        f"render_obs_s={float(row['render_obs_s']):.3f} "
        f"direct_obs_s={float(row['direct_obs_s']):.3f} "
        f"full_render_s={float(row['full_render_s']):.3f} "
        f"resize_s={float(row['resize_s']):.3f} "
        f"action_s={float(row['action_infer_s']):.3f} "
        f"avg_mb_ms={float(row['avg_minibatch_ms']):.3f}",
        flush=True,
    )
