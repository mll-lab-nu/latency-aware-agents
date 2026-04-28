#!/usr/bin/env python3
"""Offline shifted-label distillation for the queue-latency ViZDoom PPO code.

Workflow:
- collect latency-0 teacher actions on synced envs
- store a latency-agnostic base image rollout view
- for each target train latency k, build an offline shifted dataset
      x_t -> teacher_action_{t+k}
- save each shifted dataset to its own folder for later reuse
- train one BC student per training latency
- evaluate each student across a user-provided list of eval latencies

This script matches the queue-latency ViZDoom PPO module the user shared.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym as gym  # type: ignore

from stable_baselines3 import PPO


# =========================
# Repro helpers
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Optional wandb
# =========================


def maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        return wandb
    except Exception as exc:
        print(f"[WARN] wandb requested but failed to initialize: {exc}")
        return None


# =========================
# Module loading / env build
# =========================


def load_module_from_path(module_path: str):
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("user_ppo_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return accepted

def _materialize_env(maybe_env_or_factory):
    if callable(maybe_env_or_factory) and not hasattr(maybe_env_or_factory, "reset"):
        return maybe_env_or_factory()
    return maybe_env_or_factory


def build_single_env(
    module,
    obs_mode: str,
    seed: int,
    fixed_latency: int,
    latency_norm_max: int,
    env_id: str,
    vizdoom_frame_skip: int,
    max_buttons_pressed: int,
    use_multi_binary_action_space: bool,
    treat_episode_timeout_as_truncation: bool,
    screen_size: Tuple[int, int],
    grayscale: bool,
    clip_reward: bool,
    stack_frames: int,
    history_len: int,
    memory_use_latency: bool,
    memory_latency_repr: str,
    render_mode: Optional[str],
    worker_tmp_root: str,
):
    """
    Build a single ViZDoom env from the user's queue-latency PPO module.

    For latency-aware obs modes, keep the normalization denominator consistent
    via `latency_norm_max`. We do that by passing frame_skip_values as
    [fixed_latency, latency_norm_max] and curriculum sampling with a giant stage,
    so the actual sampled latency stays fixed at `fixed_latency` while wrappers
    still compute max(frame_skip_values) == latency_norm_max.
    """
    candidates = [
        "make_env",
        "build_env",
        "create_env",
        "get_env",
        "make_vizdoom_env",
        "make_vizdoom_basic_env",
    ]

    fixed_latency = int(fixed_latency)
    latency_norm_max = max(int(latency_norm_max), fixed_latency, 1)

    if latency_norm_max == fixed_latency:
        frame_skip_values = [fixed_latency]
        latency_sampling_mode = "uniform_all"
    else:
        frame_skip_values = [fixed_latency, latency_norm_max]
        latency_sampling_mode = "curriculum"

    # If the training module exposes registration helpers, use them first so gym.make works reliably.
    prewarm_fn = getattr(module, "prewarm_vizdoom_registration", None)
    if callable(prewarm_fn):
        try:
            prewarm_fn(
                env_id=env_id,
                vizdoom_frame_skip=vizdoom_frame_skip,
                max_buttons_pressed=max_buttons_pressed,
                use_multi_binary_action_space=use_multi_binary_action_space,
                treat_episode_timeout_as_truncation=treat_episode_timeout_as_truncation,
            )
        except Exception as exc:
            print(f"[WARN] prewarm_vizdoom_registration failed: {exc}")
    ensure_fn = getattr(module, "ensure_vizdoom_registered", None)
    if callable(ensure_fn):
        try:
            ensure_fn()
        except Exception as exc:
            print(f"[WARN] ensure_vizdoom_registered failed: {exc}")

    common_kwargs = dict(
        seed=seed,
        worker_idx=0,
        env_id=env_id,
        obs_mode=obs_mode,
        frame_skip_values=frame_skip_values,
        episodes_per_stage=10**9,
        randomize_within_stage=False,
        latency_mode="episode_fixed",
        latency_sampling_mode=latency_sampling_mode,
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        vizdoom_frame_skip=vizdoom_frame_skip,
        max_buttons_pressed=max_buttons_pressed,
        use_multi_binary_action_space=use_multi_binary_action_space,
        treat_episode_timeout_as_truncation=treat_episode_timeout_as_truncation,
        screen_size=screen_size,
        grayscale=grayscale,
        clip_reward=clip_reward,
        stack_frames=stack_frames,
        history_len=history_len,
        memory_use_latency=memory_use_latency,
        memory_latency_repr=memory_latency_repr,
        render_mode=render_mode,
        worker_tmp_root=worker_tmp_root,
    )

    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                env_or_factory = fn(**_filter_kwargs_for_callable(fn, common_kwargs))
                env = _materialize_env(env_or_factory)
                print(f"[INFO] Built env using module factory: {name}")
                return env
            except Exception as exc:
                print(f"[WARN] Failed env factory {name}: {exc}")

    raise RuntimeError(
        "Could not build environment from the provided ViZDoom PPO module. Expected a callable like make_env(...)."
    )


# =========================
# Observation helpers

# =========================


def parse_int_pair(s: str) -> Tuple[int, int]:
    values = [int(x.strip()) for x in str(s).split(',') if x.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected exactly two integers, got: {s}")
    return int(values[0]), int(values[1])


def hwc_to_chw(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs)
    if obs.ndim != 3:
        raise ValueError(f"Expected 3D image observation, got shape={obs.shape}")
    if obs.shape[0] <= 16 and obs.shape[1] >= 16 and obs.shape[2] >= 16:
        return obs
    return np.transpose(obs, (2, 0, 1))



def image_with_latency_channel(chw_image: np.ndarray, latency: int, latency_norm_max: int) -> np.ndarray:
    chw_image = np.asarray(chw_image, dtype=np.uint8)
    latency_norm_max = max(int(latency_norm_max), 1)
    latency = max(0, int(latency))
    skip_norm = min(latency / float(latency_norm_max), 1.0)
    plane_value = int(round(skip_norm * 255.0))
    plane = np.full((1, chw_image.shape[1], chw_image.shape[2]), plane_value, dtype=np.uint8)
    return np.concatenate([chw_image, plane], axis=0)



def latency_scalar(latency: int, latency_norm_max: int) -> np.ndarray:
    latency_norm_max = max(int(latency_norm_max), 1)
    latency = max(0, int(latency))
    return np.array([min(latency / float(latency_norm_max), 1.0)], dtype=np.float32)



def recursive_stack(items: Sequence[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {k: recursive_stack([item[k] for item in items]) for k in first.keys()}
    return np.stack(items, axis=0)



def recursive_to_tensor(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: recursive_to_tensor(v) for k, v in x.items()}
    return torch.from_numpy(np.asarray(x))



def recursive_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    return x.to(device)



def recursive_unsqueeze(x: Any, dim: int = 0) -> Any:
    if isinstance(x, dict):
        return {k: recursive_unsqueeze(v, dim=dim) for k, v in x.items()}
    return np.expand_dims(np.asarray(x), axis=dim)



def recursive_squeeze(x: Any, dim: int = 0) -> Any:
    if isinstance(x, dict):
        return {k: recursive_squeeze(v, dim=dim) for k, v in x.items()}
    return np.squeeze(np.asarray(x), axis=dim)


def live_obs_to_student_format(x: Any) -> Any:
    """Convert live env observations to the same layout used in the offline dataset.

    ViZDoom env wrappers emit HWC images (and THWC history images), while the
    distilled student is trained on CHW / TCHW tensors. This helper mirrors the
    offline dataset layout at evaluation time.
    """
    if isinstance(x, dict):
        return {k: live_obs_to_student_format(v) for k, v in x.items()}

    arr = np.asarray(x)

    if arr.ndim == 3:
        if arr.shape[0] <= 16 and arr.shape[1] >= 16 and arr.shape[2] >= 16:
            return arr
        if arr.shape[-1] <= 16 and arr.shape[0] >= 16 and arr.shape[1] >= 16:
            return np.transpose(arr, (2, 0, 1))
        return arr

    if arr.ndim == 4:
        if arr.shape[1] <= 16 and arr.shape[2] >= 16 and arr.shape[3] >= 16:
            return arr
        if arr.shape[-1] <= 16 and arr.shape[1] >= 16 and arr.shape[2] >= 16:
            return np.transpose(arr, (0, 3, 1, 2))
        return arr

    return arr


# =========================
# Student models
# =========================


class ImageBCPolicy(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            feat_dim = self.net(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.head(self.net(x))


class LateFusionBCPolicy(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], num_actions: int, latency_hidden_dims: Sequence[int]):
        super().__init__()
        c, h, w = image_shape
        self.image_backbone = ImageBCPolicy(c, h, w, num_actions=512).net
        with torch.no_grad():
            feat_dim = self.image_backbone(torch.zeros(1, c, h, w)).shape[1]

        dims = [1] + [int(x) for x in latency_hidden_dims]
        latency_layers: List[nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            latency_layers.append(nn.Linear(din, dout))
            latency_layers.append(nn.ReLU(inplace=True))
        self.latency_mlp = nn.Sequential(*latency_layers) if latency_layers else nn.Identity()
        latency_dim = dims[-1] if len(dims) > 1 else 1

        self.head = nn.Sequential(
            nn.Linear(feat_dim + latency_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"].float() / 255.0
        latency = obs["latency"].float()
        img_feat = self.image_backbone(image)
        lat_feat = self.latency_mlp(latency)
        return self.head(torch.cat([img_feat, lat_feat], dim=1))


class MemoryBCPolicy(nn.Module):
    def __init__(
        self,
        current_image_shape: Tuple[int, int, int],
        history_len: int,
        num_actions: int,
        action_embed_dim: int,
        memory_hidden_dim: int,
        memory_num_layers: int,
        memory_model: str,
        include_latency: bool,
        latency_hidden_dims: Sequence[int],
        transformer_heads: int,
        transformer_ff_dim: int,
        transformer_dropout: float,
    ):
        super().__init__()
        c, h, w = current_image_shape
        self.num_actions = int(num_actions)
        self.pad_action = int(num_actions)
        self.history_len = int(history_len)
        self.memory_model = str(memory_model)
        self.include_latency = bool(include_latency)
        self.memory_hidden_dim = int(memory_hidden_dim)

        self.image_backbone = ImageBCPolicy(c, h, w, num_actions=512).net
        with torch.no_grad():
            image_feat_dim = self.image_backbone(torch.zeros(1, c, h, w)).shape[1]

        self.current_proj = nn.Linear(image_feat_dim, self.memory_hidden_dim)
        self.action_embed = nn.Embedding(self.num_actions + 1, int(action_embed_dim), padding_idx=self.pad_action)
        seq_input_dim = int(image_feat_dim + action_embed_dim)

        if self.memory_model == "rnn":
            self.sequence_model = nn.RNN(
                input_size=seq_input_dim,
                hidden_size=self.memory_hidden_dim,
                num_layers=int(memory_num_layers),
                batch_first=True,
            )
        elif self.memory_model == "gru":
            self.sequence_model = nn.GRU(
                input_size=seq_input_dim,
                hidden_size=self.memory_hidden_dim,
                num_layers=int(memory_num_layers),
                batch_first=True,
            )
        elif self.memory_model == "lstm":
            self.sequence_model = nn.LSTM(
                input_size=seq_input_dim,
                hidden_size=self.memory_hidden_dim,
                num_layers=int(memory_num_layers),
                batch_first=True,
            )
        elif self.memory_model == "transformer":
            self.sequence_proj = nn.Linear(seq_input_dim, self.memory_hidden_dim)
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, self.memory_hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=self.memory_hidden_dim,
                nhead=int(transformer_heads),
                dim_feedforward=int(transformer_ff_dim),
                dropout=float(transformer_dropout),
                batch_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(layer, num_layers=int(memory_num_layers), enable_nested_tensor=False)
        else:
            raise ValueError(f"Unsupported memory_model: {self.memory_model}")

        latency_dim = 0
        if self.include_latency:
            dims = [1] + [int(x) for x in latency_hidden_dims]
            latency_layers: List[nn.Module] = []
            for din, dout in zip(dims[:-1], dims[1:]):
                latency_layers.append(nn.Linear(din, dout))
                latency_layers.append(nn.ReLU(inplace=True))
            self.latency_mlp = nn.Sequential(*latency_layers) if latency_layers else nn.Identity()
            latency_dim = dims[-1] if len(dims) > 1 else 1
        else:
            self.latency_mlp = None

        self.head = nn.Sequential(
            nn.Linear(self.memory_hidden_dim * 2 + latency_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_actions),
        )

    def _encode_images(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.image_backbone(x.float() / 255.0)
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            flat = x.reshape(b * t, c, h, w).float() / 255.0
            enc = self.image_backbone(flat)
            return enc.view(b, t, -1)
        raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")

    def _gather_last_valid(self, seq_out: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        lengths = valid_mask.long().sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
        gathered = seq_out[torch.arange(seq_out.shape[0], device=seq_out.device), last_idx]
        zero_mask = (lengths == 0).unsqueeze(1)
        return torch.where(zero_mask, torch.zeros_like(gathered), gathered)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_image = obs["current_image"]
        history_images = obs["history_images"]
        history_actions = obs["history_actions"].long()

        current_feat = self._encode_images(current_image)
        current_feat = self.current_proj(current_feat)

        hist_img_feat = self._encode_images(history_images)
        hist_action_feat = self.action_embed(history_actions)
        seq_inputs = torch.cat([hist_img_feat, hist_action_feat], dim=-1)
        valid_mask = history_actions != self.pad_action

        if self.memory_model in {"rnn", "gru", "lstm"}:
            seq_out, _ = self.sequence_model(seq_inputs)
            memory_feat = self._gather_last_valid(seq_out, valid_mask)
        else:
            seq_proj = self.sequence_proj(seq_inputs) + self.pos_embedding[:, : seq_inputs.shape[1], :]
            key_padding_mask = ~valid_mask
            all_pad_rows = key_padding_mask.all(dim=1)
            if all_pad_rows.any():
                seq_proj = seq_proj.clone()
                key_padding_mask = key_padding_mask.clone()
                seq_proj[all_pad_rows, 0, :] = 0.0
                key_padding_mask[all_pad_rows, 0] = False
            seq_out = self.sequence_model(seq_proj, src_key_padding_mask=key_padding_mask)
            memory_feat = self._gather_last_valid(seq_out, valid_mask)

        pieces = [current_feat, memory_feat]
        if self.include_latency and self.latency_mlp is not None:
            pieces.append(self.latency_mlp(obs["latency"].float()))
        return self.head(torch.cat(pieces, dim=1))


# =========================
# Dataset helpers
# =========================


@dataclass
class EpisodeData:
    base_observations: List[np.ndarray]  # CHW stacked image
    actions: List[int]  # latency-0 teacher actions
    rewards: List[float]
    dones: List[bool]


class ShiftedTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, observations: Sequence[Any], actions: np.ndarray):
        self.observations = list(observations)
        self.actions = np.asarray(actions, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        return self.observations[idx], int(self.actions[idx])



def shifted_history_actions(actions: Sequence[int], t: int, latency: int, history_len: int, pad_action: int) -> np.ndarray:
    vals: List[int] = []
    start = t - history_len
    for j in range(start, t):
        if j < 0:
            vals.append(pad_action)
        else:
            vals.append(int(actions[j + latency]))
    return np.asarray(vals, dtype=np.int64)



def history_images(base_obs: Sequence[np.ndarray], t: int, history_len: int) -> np.ndarray:
    current = np.asarray(base_obs[t], dtype=np.uint8)
    vals: List[np.ndarray] = []
    start = t - history_len
    for j in range(start, t):
        if j < 0:
            vals.append(np.zeros_like(current, dtype=np.uint8))
        else:
            vals.append(np.asarray(base_obs[j], dtype=np.uint8))
    return np.stack(vals, axis=0).astype(np.uint8)



def build_student_observation(
    episode: EpisodeData,
    t: int,
    latency: int,
    obs_mode: str,
    history_len: int,
    latency_norm_max: int,
    num_actions: int,
    memory_use_latency: bool,
    memory_latency_repr: str,
) -> Any:
    base = np.asarray(episode.base_observations[t], dtype=np.uint8)

    if obs_mode == "image_only":
        return base

    if obs_mode == "latency_channel":
        return image_with_latency_channel(base, latency=latency, latency_norm_max=latency_norm_max)

    if obs_mode == "late_fusion":
        return {
            "image": base,
            "latency": latency_scalar(latency=latency, latency_norm_max=latency_norm_max),
        }

    if obs_mode == "memory":
        include_latency = bool(memory_use_latency and memory_latency_repr == "separate")
        current_image = base
        if memory_latency_repr == "channel":
            current_image = image_with_latency_channel(current_image, latency=latency, latency_norm_max=latency_norm_max)
            hist_imgs = history_images(episode.base_observations, t=t, history_len=history_len)
            hist_imgs = np.stack(
                [image_with_latency_channel(img, latency=latency, latency_norm_max=latency_norm_max) for img in hist_imgs],
                axis=0,
            ).astype(np.uint8)
        else:
            hist_imgs = history_images(episode.base_observations, t=t, history_len=history_len)

        obs: Dict[str, Any] = {
            "current_image": current_image.astype(np.uint8),
            "history_images": hist_imgs.astype(np.uint8),
            "history_actions": shifted_history_actions(
                episode.actions,
                t=t,
                latency=latency,
                history_len=history_len,
                pad_action=int(num_actions),
            ),
        }
        if include_latency:
            obs["latency"] = latency_scalar(latency=latency, latency_norm_max=latency_norm_max)
        return obs

    raise ValueError(f"Unsupported student obs_mode: {obs_mode}")



def flatten_shifted_dataset(
    episodes: Sequence[EpisodeData],
    latency: int,
    obs_mode: str,
    history_len: int,
    latency_norm_max: int,
    num_actions: int,
    memory_use_latency: bool,
    memory_latency_repr: str,
):
    xs: List[Any] = []
    ys: List[int] = []
    for ep in episodes:
        T = len(ep.actions)
        if T <= latency:
            continue
        for t in range(T - latency):
            xs.append(
                build_student_observation(
                    episode=ep,
                    t=t,
                    latency=latency,
                    obs_mode=obs_mode,
                    history_len=history_len,
                    latency_norm_max=latency_norm_max,
                    num_actions=num_actions,
                    memory_use_latency=memory_use_latency,
                    memory_latency_repr=memory_latency_repr,
                )
            )
            ys.append(int(ep.actions[t + latency]))
    if not xs:
        raise ValueError(f"No shifted examples created for latency={latency}")
    return xs, np.asarray(ys, dtype=np.int64)


# =========================
# Teacher rollout collection
# =========================


def infer_num_actions_from_env(env: gym.Env) -> int:
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError(f"Expected Discrete action space, got {action_space}")
    return int(action_space.n)



def teacher_predict(teacher, obs: Any) -> int:
    action, _ = teacher.predict(obs, deterministic=True)
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)



def collect_teacher_episodes(
    teacher,
    actor_env: gym.Env,
    record_env: gym.Env,
    num_episodes: int,
    max_steps: int,
) -> List[EpisodeData]:
    episodes: List[EpisodeData] = []

    for ep_idx in range(num_episodes):
        actor_obs, _ = actor_env.reset()
        record_obs, _ = record_env.reset()

        ep_base_obs: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_dones: List[bool] = []
        ep_return = 0.0

        for _ in range(max_steps):
            action = teacher_predict(teacher, actor_obs)

            next_actor_obs, actor_reward, actor_term, actor_trunc, _ = actor_env.step(action)
            next_record_obs, record_reward, record_term, record_trunc, _ = record_env.step(action)

            actor_done = bool(actor_term or actor_trunc)
            record_done = bool(record_term or record_trunc)
            if actor_done != record_done:
                raise RuntimeError(
                    f"Actor env and record env desynced on done flag: actor_done={actor_done} record_done={record_done}"
                )

            ep_base_obs.append(hwc_to_chw(np.asarray(record_obs, dtype=np.uint8)))
            ep_actions.append(int(action))
            ep_rewards.append(float(record_reward))
            ep_dones.append(record_done)
            ep_return += float(record_reward)

            actor_obs = next_actor_obs
            record_obs = next_record_obs
            if record_done:
                break

        episodes.append(
            EpisodeData(
                base_observations=ep_base_obs,
                actions=ep_actions,
                rewards=ep_rewards,
                dones=ep_dones,
            )
        )
        print(f"[collect] episode={ep_idx+1}/{num_episodes} len={len(ep_actions)} return={ep_return:.3f}")

    return episodes


# =========================
# Training / eval helpers
# =========================


def collate_batch(batch: Sequence[Tuple[Any, int]]):
    obs_items = [item[0] for item in batch]
    act_items = [item[1] for item in batch]
    obs = recursive_to_tensor(recursive_stack(obs_items))
    actions = torch.tensor(act_items, dtype=torch.long)
    return obs, actions



def make_student_model(
    obs_mode: str,
    sample_obs: Any,
    num_actions: int,
    args: argparse.Namespace,
) -> nn.Module:
    if obs_mode in {"image_only", "latency_channel"}:
        sample_obs = np.asarray(sample_obs)
        if sample_obs.ndim != 3:
            raise ValueError(f"Expected CHW image observation, got {sample_obs.shape}")
        c, h, w = sample_obs.shape
        return ImageBCPolicy(in_channels=c, height=h, width=w, num_actions=num_actions)

    if obs_mode == "late_fusion":
        image = np.asarray(sample_obs["image"])
        c, h, w = image.shape
        latency_hidden_dims = [int(x) for x in args.latency_hidden_dims.split(",") if x.strip()]
        return LateFusionBCPolicy(
            image_shape=(c, h, w),
            num_actions=num_actions,
            latency_hidden_dims=latency_hidden_dims,
        )

    if obs_mode == "memory":
        current = np.asarray(sample_obs["current_image"])
        c, h, w = current.shape
        latency_hidden_dims = [int(x) for x in args.latency_hidden_dims.split(",") if x.strip()]
        include_latency = bool(args.memory_use_latency and args.memory_latency_repr == "separate")
        return MemoryBCPolicy(
            current_image_shape=(c, h, w),
            history_len=args.history_len,
            num_actions=num_actions,
            action_embed_dim=args.action_embed_dim,
            memory_hidden_dim=args.memory_hidden_dim,
            memory_num_layers=args.memory_num_layers,
            memory_model=args.memory_model,
            include_latency=include_latency,
            latency_hidden_dims=latency_hidden_dims,
            transformer_heads=args.transformer_heads,
            transformer_ff_dim=args.transformer_ff_dim,
            transformer_dropout=args.transformer_dropout,
        )

    raise ValueError(f"Unsupported student obs_mode: {obs_mode}")



def train_student(
    model: nn.Module,
    train_dataset: ShiftedTrajectoryDataset,
    args: argparse.Namespace,
    wandb_run=None,
    log_prefix: str = "",
):
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )
    device = torch.device(args.device)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_weights = None
    if args.balance_loss:
        acts = train_dataset.actions
        counts = np.bincount(acts, minlength=int(acts.max()) + 1).astype(np.float64)
        counts[counts == 0] = 1.0
        inv = counts.sum() / counts
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        print(f"[INFO] Using class-weighted CE loss: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_obs, batch_act in loader:
            batch_obs = recursive_to_device(batch_obs, device)
            batch_act = batch_act.to(device)

            logits = model(batch_obs)
            loss = criterion(logits, batch_act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * batch_act.shape[0]
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == batch_act).sum().item())
            total_count += int(batch_act.shape[0])

        mean_loss = total_loss / max(total_count, 1)
        mean_acc = total_correct / max(total_count, 1)
        print(f"[train]{log_prefix} epoch={epoch+1}/{args.epochs} loss={mean_loss:.6f} acc={mean_acc:.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{log_prefix}train/loss": mean_loss,
                    f"{log_prefix}train/acc": mean_acc,
                    f"{log_prefix}train/epoch": epoch + 1,
                }
            )

    return model


@torch.no_grad()
def student_act(model: nn.Module, obs: Any, device: torch.device) -> int:
    obs = live_obs_to_student_format(obs)
    batch_obs = recursive_to_tensor(recursive_unsqueeze(obs, dim=0))
    batch_obs = recursive_to_device(batch_obs, device)
    logits = model(batch_obs)
    return int(logits.argmax(dim=-1).item())



def evaluate_student(
    model: nn.Module,
    env: gym.Env,
    num_episodes: int,
    max_steps: int,
    device: torch.device,
):
    model.eval()
    returns = []
    lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_len = 0
        for _ in range(max_steps):
            action = student_act(model, obs, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if terminated or truncated:
                break
        returns.append(ep_return)
        lengths.append(ep_len)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "returns": [float(x) for x in returns],
        "lengths": [int(x) for x in lengths],
    }



# =========================
# Save/load
# =========================


def save_rollouts(path: str, episodes: Sequence[EpisodeData]) -> None:
    payload = []
    for ep in episodes:
        payload.append(
            {
                "base_observations": ep.base_observations,
                "actions": ep.actions,
                "rewards": ep.rewards,
                "dones": ep.dones,
            }
        )
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[INFO] Saved teacher rollouts to {path}")


def load_rollouts(path: str) -> List[EpisodeData]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    episodes = []
    for item in payload:
        if "base_observations" in item:
            base_obs = item["base_observations"]
        elif "observations" in item:
            base_obs = item["observations"]
        else:
            raise KeyError("Rollout file missing base_observations/observations")
        episodes.append(
            EpisodeData(
                base_observations=base_obs,
                actions=item["actions"],
                rewards=item["rewards"],
                dones=item["dones"],
            )
        )
    print(f"[INFO] Loaded teacher rollouts from {path}")
    return episodes


def save_student(path: str, model: nn.Module, metadata: Dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    print(f"[INFO] Saved student to {path}")


def load_student_checkpoint(path: str, sample_obs: Any, num_actions: int, args: argparse.Namespace) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    model = make_student_model(
        obs_mode=args.student_obs_mode,
        sample_obs=sample_obs,
        num_actions=num_actions,
        args=args,
    )
    model.load_state_dict(payload["state_dict"])
    print(f"[INFO] Loaded student checkpoint from {path}")
    return model, metadata


def shifted_dataset_dir(root: str, latency: int) -> str:
    return os.path.join(root, f"latency_{int(latency)}")


def shifted_dataset_exists(root: str, latency: int) -> bool:
    indir = shifted_dataset_dir(root, latency)
    return (
        os.path.exists(os.path.join(indir, "observations.pkl"))
        and os.path.exists(os.path.join(indir, "actions.npy"))
        and os.path.exists(os.path.join(indir, "metadata.json"))
    )


def save_shifted_dataset(root: str, latency: int, observations: Sequence[Any], actions: np.ndarray, metadata: Dict[str, Any]) -> str:
    outdir = shifted_dataset_dir(root, latency)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "observations.pkl"), "wb") as f:
        pickle.dump(list(observations), f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(outdir, "actions.npy"), np.asarray(actions, dtype=np.int64))
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved shifted dataset for latency={latency} to {outdir}")
    return outdir


def load_shifted_dataset(root: str, latency: int) -> Tuple[List[Any], np.ndarray, Dict[str, Any]]:
    indir = shifted_dataset_dir(root, latency)
    with open(os.path.join(indir, "observations.pkl"), "rb") as f:
        observations = pickle.load(f)
    actions = np.load(os.path.join(indir, "actions.npy"))
    with open(os.path.join(indir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f"[INFO] Loaded shifted dataset for latency={latency} from {indir}")
    return list(observations), np.asarray(actions, dtype=np.int64), metadata


# =========================
# CLI
# =========================


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ppo-module-path", type=str, required=True)
    p.add_argument("--teacher-model-path", type=str, required=True)
    p.add_argument("--teacher-rollout-path", type=str, required=True)
    p.add_argument("--shifted-dataset-root", type=str, required=True)
    p.add_argument("--student-save-dir", type=str, default="")
    p.add_argument("--student-save-path", type=str, default="")
    p.add_argument("--eval-json-path", type=str, required=True)

    p.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "collect_only", "build_only", "train_only", "eval_only"],
    )

    p.add_argument("--teacher-episodes", type=int, default=500)
    p.add_argument("--teacher-max-steps", type=int, default=300)
    p.add_argument(
        "--teacher-obs-mode",
        type=str,
        default="image_only",
        choices=["image_only", "latency_channel", "late_fusion", "memory"],
    )
    p.add_argument(
        "--student-obs-mode",
        type=str,
        default="image_only",
        choices=["image_only", "latency_channel", "late_fusion", "memory"],
    )
    p.add_argument("--train-latencies", type=int, nargs="+", required=True)
    p.add_argument("--eval-latencies", type=int, nargs="+", required=True)
    p.add_argument(
        "--latency-normalization-max",
        type=int,
        default=-1,
        help="Max latency used to normalize latency-aware observations. Default: max(train/eval latencies).",
    )
    p.add_argument("--reuse-teacher-rollouts", action="store_true")
    p.add_argument("--reuse-shifted-datasets", action="store_true")

    p.add_argument("--env-id", type=str, default="VizdoomBasic-v1")
    p.add_argument("--vizdoom-frame-skip", type=int, default=1)
    p.add_argument("--max-buttons-pressed", type=int, default=1)
    p.add_argument("--use-multi-binary-action-space", action="store_true")
    p.add_argument("--treat-episode-timeout-as-truncation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--screen-size", type=str, default="84,84")
    p.add_argument("--worker-tmp-root", type=str, default="")
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--clip-reward", action="store_true")
    p.add_argument("--stack-frames", type=int, default=4)

    p.add_argument("--history-len", type=int, default=8)
    p.add_argument("--memory-use-latency", action="store_true")
    p.add_argument(
        "--memory-latency-repr",
        type=str,
        default="separate",
        choices=["none", "separate", "channel"],
    )
    p.add_argument("--memory-model", type=str, default="lstm", choices=["rnn", "gru", "lstm", "transformer"])
    p.add_argument("--action-embed-dim", type=int, default=32)
    p.add_argument("--memory-hidden-dim", type=int, default=256)
    p.add_argument("--memory-num-layers", type=int, default=1)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-ff-dim", type=int, default=512)
    p.add_argument("--transformer-dropout", type=float, default=0.1)
    p.add_argument("--latency-hidden-dims", type=str, default="32,32")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--balance-loss", action="store_true")

    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-max-steps", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="vizdoom_queue_distill")
    p.add_argument("--wandb-run-name", type=str, default="vizdoom_queue_shifted_bc")

    return p.parse_args()


def resolve_student_path(args: argparse.Namespace, train_latency: int) -> str:
    if args.student_save_path:
        base = Path(args.student_save_path)
        if len(args.train_latencies) == 1:
            return str(base)
        suffix = base.suffix or ".pt"
        return str(base.with_name(f"{base.stem}_latency_{train_latency}{suffix}"))

    if args.student_save_dir:
        os.makedirs(args.student_save_dir, exist_ok=True)
        return os.path.join(args.student_save_dir, f"vizdoom_queue_student_latency_{train_latency}.pt")

    raise ValueError("Provide either --student-save-dir or --student-save-path")


# =========================
# Main
# =========================


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.latency_normalization_max < 0:
        args.latency_normalization_max = max([int(x) for x in args.train_latencies + args.eval_latencies])

    rollout_dir = os.path.dirname(os.path.abspath(args.teacher_rollout_path))
    if rollout_dir:
        os.makedirs(rollout_dir, exist_ok=True)

    os.makedirs(args.shifted_dataset_root, exist_ok=True)

    eval_json_dir = os.path.dirname(os.path.abspath(args.eval_json_path))
    if eval_json_dir:
        os.makedirs(eval_json_dir, exist_ok=True)

    if args.student_save_dir:
        os.makedirs(args.student_save_dir, exist_ok=True)
    elif args.student_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.student_save_path)), exist_ok=True)

    wandb_run = maybe_init_wandb(args)
    module = load_module_from_path(args.ppo_module_path)

    need_teacher_rollouts = args.mode in {"all", "collect_only", "build_only"} or (
        args.mode == "train_only" and not args.reuse_shifted_datasets
    )

    episodes = None
    if need_teacher_rollouts:
        if args.reuse_teacher_rollouts and os.path.exists(args.teacher_rollout_path):
            episodes = load_rollouts(args.teacher_rollout_path)
        else:
            print(f"[INFO] Loading PPO teacher from {args.teacher_model_path}")
            teacher = PPO.load(args.teacher_model_path, device=args.device)

            actor_env = build_single_env(
                module=module,
                obs_mode=args.teacher_obs_mode,
                seed=args.seed,
                fixed_latency=0,
                latency_norm_max=max(args.latency_normalization_max, 1),
                env_id=args.env_id,
                vizdoom_frame_skip=args.vizdoom_frame_skip,
                max_buttons_pressed=args.max_buttons_pressed,
                use_multi_binary_action_space=args.use_multi_binary_action_space,
                treat_episode_timeout_as_truncation=args.treat_episode_timeout_as_truncation,
                screen_size=parse_int_pair(args.screen_size),
                grayscale=args.grayscale,
                clip_reward=args.clip_reward,
                stack_frames=args.stack_frames,
                history_len=args.history_len,
                memory_use_latency=args.memory_use_latency,
                memory_latency_repr=args.memory_latency_repr,
                render_mode=None,
                worker_tmp_root=args.worker_tmp_root,
            )
            record_env = build_single_env(
                module=module,
                obs_mode="image_only",
                seed=args.seed,
                fixed_latency=0,
                latency_norm_max=max(args.latency_normalization_max, 1),
                env_id=args.env_id,
                vizdoom_frame_skip=args.vizdoom_frame_skip,
                max_buttons_pressed=args.max_buttons_pressed,
                use_multi_binary_action_space=args.use_multi_binary_action_space,
                treat_episode_timeout_as_truncation=args.treat_episode_timeout_as_truncation,
                screen_size=parse_int_pair(args.screen_size),
                grayscale=False,
                clip_reward=args.clip_reward,
                stack_frames=args.stack_frames,
                history_len=args.history_len,
                memory_use_latency=args.memory_use_latency,
                memory_latency_repr=args.memory_latency_repr,
                render_mode=None,
                worker_tmp_root=args.worker_tmp_root,
            )

            episodes = collect_teacher_episodes(
                teacher=teacher,
                actor_env=actor_env,
                record_env=record_env,
                num_episodes=args.teacher_episodes,
                max_steps=args.teacher_max_steps,
            )
            save_rollouts(args.teacher_rollout_path, episodes)

            returns = [sum(ep.rewards) for ep in episodes]
            lengths = [len(ep.actions) for ep in episodes]
            print(
                json.dumps(
                    {
                        "source_teacher_latency": 0,
                        "num_episodes": len(episodes),
                        "mean_return": float(np.mean(returns)),
                        "std_return": float(np.std(returns)),
                        "mean_length": float(np.mean(lengths)),
                        "std_length": float(np.std(lengths)),
                    },
                    indent=2,
                )
            )

            for env in [actor_env, record_env]:
                try:
                    env.close()
                except Exception:
                    pass

    if args.mode == "collect_only":
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        return

    if episodes is None and (args.mode in {"build_only", "train_only"} and not args.reuse_shifted_datasets):
        if os.path.exists(args.teacher_rollout_path):
            episodes = load_rollouts(args.teacher_rollout_path)
        else:
            raise FileNotFoundError(
                f"Teacher rollout file not found: {args.teacher_rollout_path}. Run with --mode all or --mode collect_only first."
            )

    sample_num_actions_env = build_single_env(
        module=module,
        obs_mode="image_only",
        seed=args.seed,
        fixed_latency=0,
        latency_norm_max=max(args.latency_normalization_max, 1),
        env_id=args.env_id,
        vizdoom_frame_skip=args.vizdoom_frame_skip,
        max_buttons_pressed=args.max_buttons_pressed,
        use_multi_binary_action_space=args.use_multi_binary_action_space,
        treat_episode_timeout_as_truncation=args.treat_episode_timeout_as_truncation,
        screen_size=parse_int_pair(args.screen_size),
        grayscale=args.grayscale,
        clip_reward=args.clip_reward,
        stack_frames=args.stack_frames,
        history_len=args.history_len,
        memory_use_latency=args.memory_use_latency,
        memory_latency_repr=args.memory_latency_repr,
        render_mode=None,
        worker_tmp_root=args.worker_tmp_root,
    )
    num_actions = infer_num_actions_from_env(sample_num_actions_env)
    sample_num_actions_env.close()

    all_eval_results: Dict[str, Any] = {}

    for train_latency in args.train_latencies:
        print(f"\n[INFO] ===== Train latency={train_latency} =====")

        if args.reuse_shifted_datasets and shifted_dataset_exists(args.shifted_dataset_root, train_latency):
            xs, ys, dataset_meta = load_shifted_dataset(args.shifted_dataset_root, train_latency)
        else:
            if episodes is None:
                if os.path.exists(args.teacher_rollout_path):
                    episodes = load_rollouts(args.teacher_rollout_path)
                else:
                    raise FileNotFoundError(
                        f"Teacher rollout file not found: {args.teacher_rollout_path}. Cannot build shifted dataset."
                    )
            xs, ys = flatten_shifted_dataset(
                episodes=episodes,
                latency=train_latency,
                obs_mode=args.student_obs_mode,
                history_len=args.history_len,
                latency_norm_max=args.latency_normalization_max,
                num_actions=num_actions,
                memory_use_latency=args.memory_use_latency,
                memory_latency_repr=args.memory_latency_repr,
            )
            uniq, counts = np.unique(ys, return_counts=True)
            action_hist_tmp = {int(k): int(v) for k, v in zip(uniq, counts)}
            dataset_meta = {
                "train_latency": int(train_latency),
                "student_obs_mode": args.student_obs_mode,
                "teacher_rollout_path": args.teacher_rollout_path,
                "num_examples": int(len(ys)),
                "action_hist": action_hist_tmp,
                "latency_normalization_max": int(args.latency_normalization_max),
                "history_len": int(args.history_len),
                "memory_use_latency": bool(args.memory_use_latency),
                "memory_latency_repr": args.memory_latency_repr,
            }
            save_shifted_dataset(args.shifted_dataset_root, train_latency, xs, ys, dataset_meta)

        dataset = ShiftedTrajectoryDataset(xs, ys)
        uniq, counts = np.unique(ys, return_counts=True)
        action_hist = {int(k): int(v) for k, v in zip(uniq, counts)}
        print(f"[INFO] Shifted dataset size for train_latency={train_latency}: {len(dataset)}")
        print(f"[INFO] Shifted action histogram for train_latency={train_latency}: {action_hist}")

        if args.mode == "build_only":
            all_eval_results[str(train_latency)] = {
                "dataset_size": int(len(dataset)),
                "action_hist": action_hist,
                "dataset_dir": shifted_dataset_dir(args.shifted_dataset_root, train_latency),
            }
            continue

        sample_obs = xs[0]
        student_path = resolve_student_path(args=args, train_latency=train_latency)

        model = None
        if args.mode in {"all", "train_only"}:
            model = make_student_model(
                obs_mode=args.student_obs_mode,
                sample_obs=sample_obs,
                num_actions=num_actions,
                args=args,
            )
            model = train_student(
                model=model,
                train_dataset=dataset,
                args=args,
                wandb_run=wandb_run,
                log_prefix=f"train_latency_{train_latency}/",
            )
            save_student(
                path=student_path,
                model=model,
                metadata={
                    "train_latency": int(train_latency),
                    "teacher_model_path": args.teacher_model_path,
                    "teacher_rollout_path": args.teacher_rollout_path,
                    "teacher_obs_mode": args.teacher_obs_mode,
                    "student_obs_mode": args.student_obs_mode,
                    "num_actions": int(num_actions),
                    "action_hist": action_hist,
                    "seed": int(args.seed),
                    "env_id": args.env_id,
                    "vizdoom_frame_skip": int(args.vizdoom_frame_skip),
                    "max_buttons_pressed": int(args.max_buttons_pressed),
                    "use_multi_binary_action_space": bool(args.use_multi_binary_action_space),
                    "treat_episode_timeout_as_truncation": bool(args.treat_episode_timeout_as_truncation),
                    "screen_size": list(parse_int_pair(args.screen_size)),
                    "grayscale": bool(args.grayscale),
                    "clip_reward": bool(args.clip_reward),
                    "stack_frames": int(args.stack_frames),
                    "latency_normalization_max": int(args.latency_normalization_max),
                    "history_len": int(args.history_len),
                    "memory_use_latency": bool(args.memory_use_latency),
                    "memory_latency_repr": args.memory_latency_repr,
                    "memory_model": args.memory_model,
                    "dataset_dir": shifted_dataset_dir(args.shifted_dataset_root, train_latency),
                },
            )

        if args.mode in {"all", "train_only", "eval_only"}:
            if model is None:
                if not os.path.exists(student_path):
                    raise FileNotFoundError(
                        f"Student checkpoint for train_latency={train_latency} not found: {student_path}"
                    )
                model, _ = load_student_checkpoint(
                    path=student_path,
                    sample_obs=sample_obs,
                    num_actions=num_actions,
                    args=args,
                )

            train_entry: Dict[str, Any] = {
                "dataset_size": int(len(dataset)),
                "action_hist": action_hist,
                "dataset_dir": shifted_dataset_dir(args.shifted_dataset_root, train_latency),
                "eval": {},
            }

            for eval_latency in args.eval_latencies:
                eval_env = build_single_env(
                    module=module,
                    obs_mode=args.student_obs_mode,
                    seed=args.seed + int(train_latency) * 1000 + int(eval_latency),
                    fixed_latency=eval_latency,
                    latency_norm_max=args.latency_normalization_max,
                    env_id=args.env_id,
                    vizdoom_frame_skip=args.vizdoom_frame_skip,
                    max_buttons_pressed=args.max_buttons_pressed,
                    use_multi_binary_action_space=args.use_multi_binary_action_space,
                    treat_episode_timeout_as_truncation=args.treat_episode_timeout_as_truncation,
                    screen_size=parse_int_pair(args.screen_size),
                    grayscale=args.grayscale,
                    clip_reward=args.clip_reward,
                    stack_frames=args.stack_frames,
                    history_len=args.history_len,
                    memory_use_latency=args.memory_use_latency,
                    memory_latency_repr=args.memory_latency_repr,
                    render_mode=None,
                    worker_tmp_root=args.worker_tmp_root,
                )
                metrics = evaluate_student(
                    model=model,
                    env=eval_env,
                    num_episodes=args.eval_episodes,
                    max_steps=args.eval_max_steps,
                    device=torch.device(args.device),
                )
                train_entry["eval"][str(eval_latency)] = metrics
                print(
                    json.dumps(
                        {
                            "train_latency": int(train_latency),
                            "eval_latency": int(eval_latency),
                            "metrics": metrics,
                        },
                        indent=2,
                    )
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"eval/train_latency_{train_latency}/eval_latency_{eval_latency}/mean_return": metrics["mean_return"],
                            f"eval/train_latency_{train_latency}/eval_latency_{eval_latency}/std_return": metrics["std_return"],
                            f"eval/train_latency_{train_latency}/eval_latency_{eval_latency}/mean_length": metrics["mean_length"],
                            f"eval/train_latency_{train_latency}/eval_latency_{eval_latency}/std_length": metrics["std_length"],
                            f"eval/train_latency_{train_latency}/dataset_size": int(len(dataset)),
                        }
                    )
                try:
                    eval_env.close()
                except Exception:
                    pass

            all_eval_results[str(train_latency)] = train_entry

    with open(args.eval_json_path, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"[INFO] Wrote eval results to {args.eval_json_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()