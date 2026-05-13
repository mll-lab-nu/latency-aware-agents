from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from bisect import insort_right
from collections import deque
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
FLAPPY_ROOT = os.path.join(ROOT, "flappy-bird-gymnasium")
if FLAPPY_ROOT not in sys.path:
    sys.path.insert(0, FLAPPY_ROOT)

import gymnasium as gym  # noqa: E402
import flappy_bird_gymnasium  # noqa: E402,F401
from flappy_bird_gymnasium.envs.render_state import (  # noqa: E402
    BatchedFlappyRenderState,
)
from flappy_bird_gymnasium.gpu_render import (  # noqa: E402
    FlappyRenderAssets,
    TorchFlappyRenderer,
)
from flappy_bird_gymnasium.gpu_render.oracle import resize_frame_pil  # noqa: E402
from latency_bench.logging.wandb import WandbMetricLogger, init_wandb  # noqa: E402


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, in_channels: int, image_size: int, n_actions: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, image_size, image_size)
            flatten_dim = int(self.network(sample).shape[1])
        self.network = nn.Sequential(
            self.network,
            layer_init(nn.Linear(flatten_dim, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x.float() / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.float() / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


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


class QueueLatencyWrapper(gym.Wrapper):
    """True input-delay wrapper matching the SB3 queue-latency scripts."""

    def __init__(self, env: gym.Env, frame_skip_values: Sequence[int], seed: int):
        super().__init__(env)
        self.frame_skip_values = [int(x) for x in frame_skip_values]
        self.rng = np.random.RandomState(seed)
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _sample_frame_skip(self) -> int:
        return int(self.rng.choice(self.frame_skip_values))

    def _schedule_action(self, action: int) -> int:
        latency_for_action = int(self.current_frame_skip)
        apply_step = self.env_step_idx + latency_for_action
        self._schedule_seq += 1
        insort_right(
            self.pending_actions,
            (apply_step, self._schedule_seq, int(action), latency_for_action),
        )
        return apply_step

    def _advance_pending_actions(self) -> None:
        while self.pending_actions and self.pending_actions[0][0] <= self.env_step_idx:
            _, _, matured_action, _ = self.pending_actions.pop(0)
            self.current_applied_action = int(matured_action)

    def reset(self, **kwargs):
        self.current_applied_action = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0
        self.current_frame_skip = int(self.frame_skip_values[0])
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["applied_action"] = int(self.current_applied_action)
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self.current_frame_skip = self._sample_frame_skip()
        scheduled_delay = int(self.current_frame_skip)
        scheduled_apply_step = self._schedule_action(int(action))
        self._advance_pending_actions()
        obs, reward, terminated, truncated, info = self.env.step(
            int(self.current_applied_action)
        )
        info = dict(info or {})
        info["frame_skip"] = scheduled_delay
        info["decision_step_idx"] = int(self.decision_step_idx)
        info["env_step_idx"] = int(self.env_step_idx)
        info["applied_action"] = int(self.current_applied_action)
        info["queued_action"] = int(action)
        info["scheduled_apply_step"] = int(scheduled_apply_step)
        info["pending_queue_len"] = int(len(self.pending_actions))
        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


class SyncFlappyBatch:
    def __init__(
        self,
        n_envs: int,
        seed: int,
        frame_skip_values: Sequence[int],
        render_mode: Optional[str],
    ) -> None:
        self.envs = []
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        for idx in range(n_envs):
            env = gym.make(
                "FlappyBird-v0",
                use_lidar=False,
                normalize_obs=True,
                render_mode=render_mode,
            )
            env = QueueLatencyWrapper(env, frame_skip_values, seed=seed + idx)
            env.reset(seed=seed + idx)
            self.envs.append(env)

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    def reset(self):
        for idx, env in enumerate(self.envs):
            env.reset()
            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0
        return self.render_states()

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)
        infos: List[Dict] = []
        for idx, (env, action) in enumerate(zip(self.envs, actions)):
            _, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            self.episode_returns[idx] += float(reward)
            self.episode_lengths[idx] += 1
            if done:
                info["episodic_return"] = float(self.episode_returns[idx])
                info["episodic_length"] = int(self.episode_lengths[idx])
                env.reset()
                self.episode_returns[idx] = 0.0
                self.episode_lengths[idx] = 0
            rewards[idx] = float(reward)
            dones[idx] = done
            infos.append(info)
        return rewards, dones, infos, self.render_states()

    def render_states(self):
        return [env.unwrapped.get_render_state() for env in self.envs]

    def render_frames(self):
        return [env.render() for env in self.envs]

    def frame_skips(self) -> np.ndarray:
        return np.asarray(
            [env.current_frame_skip for env in self.envs], dtype=np.int64
        )

    def close(self):
        for env in self.envs:
            env.close()


@dataclass
class ObservationPipeline:
    args: argparse.Namespace
    envs: SyncFlappyBatch
    device: torch.device
    renderer: Optional[TorchFlappyRenderer] = None

    def __post_init__(self):
        self.base_channels = 1 if self.args.grayscale else 3
        self.stacked_channels = self.base_channels * int(self.args.stack_frames)
        self.in_channels = self.stacked_channels + (
            1 if self.args.obs_mode == "latency_channel" else 0
        )
        self.stack = torch.empty(
            (
                self.envs.n_envs,
                self.args.image_size,
                self.args.image_size,
                self.stacked_channels,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

    def reset(self, states, profiler: Optional[UpdateProfiler] = None) -> torch.Tensor:
        frame_obs = self._render_frame_obs(states, profiler=profiler)
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            for stack_idx in range(int(self.args.stack_frames)):
                c0 = stack_idx * self.base_channels
                c1 = c0 + self.base_channels
                self.stack[:, :, :, c0:c1] = frame_obs
        return self._final_obs(self.envs.frame_skips(), profiler=profiler)

    def step(
        self,
        states,
        dones: np.ndarray,
        profiler: Optional[UpdateProfiler] = None,
    ) -> torch.Tensor:
        frame_obs = self._render_frame_obs(states, profiler=profiler)
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            if self.args.stack_frames <= 1:
                self.stack[:, :, :, :] = frame_obs
            else:
                self.stack[:, :, :, : -self.base_channels] = self.stack[
                    :, :, :, self.base_channels :
                ].clone()
                self.stack[:, :, :, -self.base_channels :] = frame_obs
            done_indices = np.flatnonzero(dones)
            for idx in done_indices:
                for stack_idx in range(int(self.args.stack_frames)):
                    c0 = stack_idx * self.base_channels
                    c1 = c0 + self.base_channels
                    self.stack[idx, :, :, c0:c1] = frame_obs[idx]
        return self._final_obs(self.envs.frame_skips(), profiler=profiler)

    def _render_frame_obs(self, states, profiler: Optional[UpdateProfiler] = None) -> torch.Tensor:
        if self.args.renderer == "cpu_oracle":
            if self.args.observation_render_mode != "full_then_resize":
                raise ValueError("cpu_oracle only supports full_then_resize observations")
            timer = profiler.time("resize_s") if profiler is not None else nullcontext()
            with timer:
                obs = [
                    resize_frame_pil(
                        frame,
                        image_size=self.args.image_size,
                        grayscale=self.args.grayscale,
                    )
                    for frame in self.envs.render_frames()
                ]
                return torch.as_tensor(np.stack(obs, axis=0), device=self.device)

        assert self.renderer is not None
        timer = (
            profiler.time("state_batch_pack_s")
            if profiler is not None
            else nullcontext()
        )
        with timer:
            batch = BatchedFlappyRenderState.from_states(states)
        if self.args.observation_render_mode == "direct_84":
            return self.renderer.render_observation_direct(
                batch,
                image_size=self.args.image_size,
                grayscale=self.args.grayscale,
                profiler=profiler,
            )
        if self.args.observation_render_mode != "full_then_resize":
            raise ValueError(
                f"unsupported observation_render_mode: {self.args.observation_render_mode}"
            )
        return self.renderer.render_observation(
            batch,
            image_size=self.args.image_size,
            grayscale=self.args.grayscale,
            resize_backend=self.args.resize_backend,
            profiler=profiler,
        )

    def _final_obs(
        self,
        frame_skips: np.ndarray,
        profiler: Optional[UpdateProfiler] = None,
    ) -> torch.Tensor:
        timer = profiler.time("final_obs_s") if profiler is not None else nullcontext()
        with timer:
            obs = self.stack
            if self.args.obs_mode == "latency_channel":
                max_skip = max(1, max(parse_int_list(self.args.frame_skip_values)))
                values = np.minimum(frame_skips / float(max_skip), 1.0)
                values = np.rint(values * 255.0).astype(np.uint8)
                latency = torch.as_tensor(values, dtype=torch.uint8, device=self.device)
                latency = latency.view(-1, 1, 1, 1).expand(
                    -1, self.args.image_size, self.args.image_size, 1
                )
                obs = torch.cat([obs, latency], dim=3)
            return obs.permute(0, 3, 1, 2).contiguous()


def save_checkpoint(path: str, agent: Agent, optimizer, args, global_step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "global_step": int(global_step),
        },
        path,
    )



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


def train(args, wandb_module) -> str:
    frame_skip_values = parse_int_list(args.frame_skip_values)
    if args.observation_render_mode == "direct_84" and args.renderer != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --renderer torch")
    if args.observation_render_mode == "direct_84" and args.resize_backend != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --resize-backend torch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    render_mode = "rgb_array" if args.renderer == "cpu_oracle" else None
    envs = SyncFlappyBatch(
        n_envs=args.n_envs,
        seed=args.seed,
        frame_skip_values=frame_skip_values,
        render_mode=render_mode,
    )
    states = envs.reset()

    renderer = None
    if args.renderer == "torch":
        assets = FlappyRenderAssets.for_state(states[0])
        renderer = TorchFlappyRenderer(assets, device=device)

    pipeline = ObservationPipeline(args=args, envs=envs, device=device, renderer=renderer)
    next_obs = pipeline.reset(states)
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)

    agent = Agent(
        in_channels=pipeline.in_channels,
        image_size=args.image_size,
        n_actions=2,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))

    rollout_size = int(args.n_envs * args.n_steps)
    minibatch_size = min(int(args.batch_size), rollout_size)
    num_updates = max(1, int(args.train_steps) // rollout_size)
    logger = WandbMetricLogger(wandb_module)
    stats_window_size = max(1, int(getattr(args, "stats_window_size", 100)))
    run_name = (
        getattr(args, "run_name", "")
        or getattr(args, "wandb_run_name", "")
        or "n/a"
    )
    profiler = UpdateProfiler(
        enabled=args.profile,
        sync_cuda=args.profile_sync_cuda,
        device=device,
    )
    if args.profile and args.profile_csv:
        init_profile_csv(args.profile_csv)

    print(
        "[train] "
        f"device={device} renderer={args.renderer} "
        f"observation_render_mode={args.observation_render_mode} "
        f"resize_backend={args.resize_backend} "
        f"obs_mode={args.obs_mode} n_envs={args.n_envs} n_steps={args.n_steps} "
        f"rollout_size={rollout_size} batch_size={minibatch_size} "
        f"updates={num_updates} frame_skip_values={args.frame_skip_values} "
        f"stats_window_size={stats_window_size} "
        f"wandb=True run_name={run_name} profile={args.profile} save={args.save}",
        flush=True,
    )

    obs = torch.zeros(
        (args.n_steps, args.n_envs, pipeline.in_channels, args.image_size, args.image_size),
        dtype=torch.uint8,
        device=device,
    )
    actions = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.n_steps, args.n_envs), device=device)
    rewards = torch.zeros((args.n_steps, args.n_envs), device=device)
    dones = torch.zeros((args.n_steps, args.n_envs), device=device)
    values = torch.zeros((args.n_steps, args.n_envs), device=device)

    recent_returns = deque(maxlen=stats_window_size)
    recent_lengths = deque(maxlen=stats_window_size)
    episode_count = 0
    last_ep_ret = float("nan")
    last_ep_len = float("nan")
    ep_rew_mean = float("nan")
    ep_len_mean = float("nan")
    start_time = time.time()
    last_progress_log = start_time
    try:
        for update in range(1, num_updates + 1):
            profiler.reset()
            profiler.synchronize()
            update_start = time.perf_counter()
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / float(num_updates)
                lr_now = args.learning_rate_final + frac * (
                    args.learning_rate - args.learning_rate_final
                )
                optimizer.param_groups[0]["lr"] = lr_now
                clip_coef_now = args.clip_coef_final + frac * (
                    args.clip_coef - args.clip_coef_final
                )
            else:
                clip_coef_now = args.clip_coef

            with profiler.time("rollout_total_s"):
                for step in range(args.n_steps):
                    global_step += args.n_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    with profiler.time("action_infer_s"):
                        with torch.no_grad():
                            action, logprob, _, value = agent.get_action_and_value(next_obs)
                            values[step] = value.flatten()
                        actions[step] = action
                        logprobs[step] = logprob

                    with profiler.time("action_cpu_copy_s"):
                        action_np = action.detach().cpu().numpy()

                    with profiler.time("env_step_s"):
                        rewards_np, dones_np, infos, states = envs.step(action_np)

                    with profiler.time("reward_done_h2d_s"):
                        rewards[step] = torch.as_tensor(
                            rewards_np, dtype=torch.float32, device=device
                        )
                        next_done = torch.as_tensor(
                            dones_np.astype(np.float32),
                            dtype=torch.float32,
                            device=device,
                        )

                    with profiler.time("render_obs_s"):
                        next_obs = pipeline.step(states, dones_np, profiler=profiler)

                    now = time.time()
                    if args.log_interval_seconds > 0 and (
                        now - last_progress_log >= args.log_interval_seconds
                    ):
                        elapsed = max(now - start_time, 1e-9)
                        print(
                            "[rollout] "
                            f"update={update}/{num_updates} "
                            f"step={step + 1}/{args.n_steps} "
                            f"global_step={global_step} "
                            f"SPS={int(global_step / elapsed)} "
                            f"episodes={episode_count} "
                            f"last_ep_rew={last_ep_ret:.3f} last_ep_len={last_ep_len:.1f} "
                            f"ep_rew_mean={ep_rew_mean:.3f} ep_len_mean={ep_len_mean:.1f} ",
                            flush=True,
                        )
                        last_progress_log = now

                    for info in infos:
                        if "episodic_return" not in info:
                            continue
                        ep_ret = float(info["episodic_return"])
                        ep_len = int(info["episodic_length"])
                        episode_count += 1
                        last_ep_ret = ep_ret
                        last_ep_len = float(ep_len)
                        recent_returns.append(ep_ret)
                        recent_lengths.append(ep_len)
                        ep_rew_mean = float(np.mean(recent_returns))
                        ep_len_mean = float(np.mean(recent_lengths))
                        logger.add_scalar("charts/episodic_return", ep_ret, global_step)
                        logger.add_scalar("charts/episodic_length", ep_len, global_step)
                        logger.add_scalar(
                            "charts/avg_episodic_return",
                            ep_rew_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "charts/avg_episodic_length",
                            ep_len_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "rollout/ep_rew_mean",
                            ep_rew_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "rollout/ep_len_mean",
                            ep_len_mean,
                            global_step,
                        )
                        logger.add_scalar(
                            "rollout/episodes",
                            episode_count,
                            global_step,
                        )

            with profiler.time("gae_s"):
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards, device=device)
                    lastgaelam = 0
                    for t in reversed(range(args.n_steps)):
                        if t == args.n_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + args.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + args.gamma
                            * args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values

            with profiler.time("flatten_s"):
                b_obs = obs.reshape(
                    (-1, pipeline.in_channels, args.image_size, args.image_size)
                )
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

            b_inds = np.arange(rollout_size)
            clipfracs = []
            with profiler.time("ppo_update_s"):
                for _ in range(args.n_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, rollout_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        profiler.add_count("num_minibatches")

                        with profiler.time("minibatch_forward_loss_s"):
                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                                b_obs[mb_inds], b_actions.long()[mb_inds]
                            )
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs.append(
                                    ((ratio - 1.0).abs() > clip_coef_now)
                                    .float()
                                    .mean()
                                    .item()
                                )

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (
                                    mb_advantages - mb_advantages.mean()
                                ) / (mb_advantages.std() + 1e-8)

                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(
                                ratio, 1 - clip_coef_now, 1 + clip_coef_now
                            )
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -clip_coef_now,
                                    clip_coef_now,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss = 0.5 * torch.max(
                                    v_loss_unclipped, v_loss_clipped
                                ).mean()
                            else:
                                v_loss = 0.5 * (
                                    (newvalue - b_returns[mb_inds]) ** 2
                                ).mean()

                            entropy_loss = entropy.mean()
                            loss = (
                                pg_loss
                                - args.ent_coef * entropy_loss
                                + v_loss * args.vf_coef
                            )

                        with profiler.time("backward_s"):
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                agent.parameters(), args.max_grad_norm
                            )

                        with profiler.time("optimizer_s"):
                            optimizer.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = int(global_step / (time.time() - start_time))
            current_lr = optimizer.param_groups[0]["lr"]
            clipfrac_mean = float(np.mean(clipfracs)) if clipfracs else float("nan")

            logger.add_scalar("charts/learning_rate", current_lr, global_step)
            logger.add_scalar("charts/clip_coef", clip_coef_now, global_step)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            print(
                "[update] "
                f"global_step={global_step} update={update}/{num_updates} SPS={sps} "
                f"episodes={episode_count} ep_rew_mean={ep_rew_mean:.3f} "
                f"ep_len_mean={ep_len_mean:.1f} loss={loss.item():.6f} "
                f"value_loss={v_loss.item():.6f} policy_loss={pg_loss.item():.6f} "
                f"entropy={entropy_loss.item():.6f} approx_kl={approx_kl.item():.6f} "
                f"clipfrac={clipfrac_mean:.3f} explained_var={explained_var:.3f} "
                f"lr={current_lr:.6g} clip_coef={clip_coef_now:.6g} ",
                flush=True,
            )

            if args.profile:
                profiler.synchronize()
                total_s = time.perf_counter() - update_start
                included = update > args.profile_warmup_updates
                row = build_profile_row(
                    args=args,
                    device=device,
                    profiler=profiler,
                    update=update,
                    global_step=global_step,
                    total_s=total_s,
                    rollout_size=rollout_size,
                    minibatch_size=minibatch_size,
                    included=included,
                    end_to_end_sps=sps,
                )
                print_profile_row(row)
                if included and args.profile_csv:
                    append_profile_csv(args.profile_csv, row)

            if args.checkpoint_freq > 0 and global_step % args.checkpoint_freq < args.n_envs:
                save_checkpoint(args.save, agent, optimizer, args, global_step)
    finally:
        envs.close()
        logger.close()

    save_checkpoint(args.save, agent, optimizer, args, global_step)
    return args.save


def eval_model(args, model_path: str, frame_skip: int):
    if args.observation_render_mode == "direct_84" and args.renderer != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --renderer torch")
    if args.observation_render_mode == "direct_84" and args.resize_backend != "torch":
        raise ValueError("--observation-render-mode direct_84 requires --resize-backend torch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.n_envs = 1
    eval_args.frame_skip_values = str(int(frame_skip))

    render_mode = "rgb_array" if eval_args.renderer == "cpu_oracle" else None
    envs = SyncFlappyBatch(
        n_envs=1,
        seed=eval_args.seed,
        frame_skip_values=[int(frame_skip)],
        render_mode=render_mode,
    )
    states = envs.reset()

    renderer = None
    if eval_args.renderer == "torch":
        assets = FlappyRenderAssets.for_state(states[0])
        renderer = TorchFlappyRenderer(assets, device=device)

    pipeline = ObservationPipeline(
        args=eval_args,
        envs=envs,
        device=device,
        renderer=renderer,
    )
    obs = pipeline.reset(states)

    agent = Agent(
        in_channels=pipeline.in_channels,
        image_size=eval_args.image_size,
        n_actions=2,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    returns: List[float] = []
    lengths: List[int] = []
    try:
        for _ in range(eval_args.eval_episodes):
            states = envs.reset()
            obs = pipeline.reset(states)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while (not done) and (
                eval_args.eval_max_steps <= 0 or ep_len < eval_args.eval_max_steps
            ):
                with torch.no_grad():
                    hidden = agent.network(obs.float() / 255.0)
                    logits = agent.actor(hidden)
                    action = torch.argmax(logits, dim=1)

                rewards_np, dones_np, _, states = envs.step(
                    action.detach().cpu().numpy()
                )
                done = bool(dones_np[0])
                ep_ret += float(rewards_np[0])
                ep_len += 1
                if not done:
                    obs = pipeline.step(states, dones_np)

            returns.append(ep_ret)
            lengths.append(ep_len)
    finally:
        envs.close()

    return returns, lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CleanRL-style PPO for Flappy Bird with render-state Torch observations."
    )
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--save", type=str, default="./models/flappy_cleanrl_gpu_render.pt")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    parser.add_argument("--stats-window-size", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-frame-skip-values", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--eval-max-steps", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-sync-cuda", action="store_true")
    parser.add_argument("--profile-csv", type=str, default="")
    parser.add_argument("--profile-warmup-updates", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--learning-rate-final", type=float, default=2.5e-6)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--clip-coef-final", type=float, default=0.025)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--stack-frames", type=int, default=4)
    parser.add_argument(
        "--obs-mode",
        choices=["image_only", "latency_channel"],
        default="latency_channel",
    )
    parser.add_argument("--frame-skip-values", type=str, default="0,1,2,3,4,5")
    parser.add_argument(
        "--renderer",
        choices=["torch", "cpu_oracle"],
        default="torch",
        help="cpu_oracle uses env.render()+PIL for algorithm parity checks.",
    )
    parser.add_argument(
        "--resize-backend",
        choices=["pil", "torch"],
        default="pil",
        help="pil is pixel-exact with the SB3 oracle; torch keeps resize on device but is approximate.",
    )
    parser.add_argument(
        "--observation-render-mode",
        choices=["full_then_resize", "direct_84"],
        default="full_then_resize",
        help=(
            "full_then_resize draws the full rgb frame before resizing; direct_84 "
            "skips the full frame and directly emits CNN-sized observations."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="flappy_rl_image_queue_latency")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    args = parser.parse_args()

    if args.train_steps < 0:
        raise ValueError("--train-steps must be non-negative")
    if args.stats_window_size <= 0:
        raise ValueError("--stats-window-size must be positive")
    if not args.run_name and not args.wandb_run_name:
        args.run_name = f"flappy_gpu_render__{args.seed}__{int(time.time())}"
    elif not args.run_name:
        args.run_name = args.wandb_run_name

    model_path = args.load or args.save
    wandb_module = init_wandb(
        args,
        default_run_name=f"flappy_gpu_render_{int(time.time())}",
    )
    try:
        if args.train_steps > 0:
            model_path = train(args, wandb_module=wandb_module)
            print(f"[train] saved => {model_path}", flush=True)

        if args.eval_episodes > 0:
            for fs in parse_int_list(args.eval_frame_skip_values):
                returns, lengths = eval_model(args, model_path=model_path, frame_skip=fs)
                print(
                    f"[eval] frame_skip={fs} "
                    f"mean_return={float(np.mean(returns)):.3f} "
                    f"std_return={float(np.std(returns)):.3f} "
                    f"mean_len={float(np.mean(lengths)):.1f}",
                    flush=True,
                )
    finally:
        wandb_module.finish()


if __name__ == "__main__":
    main()
