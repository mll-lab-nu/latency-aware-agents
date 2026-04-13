from __future__ import annotations

import argparse
import json
import os
import time
from bisect import insort_right
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict as DictSpace

try:
    import ale_py  # noqa: F401  # Registers Atari envs for Gymnasium installs that need it
except Exception:
    ale_py = None  # type: ignore

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


# =========================
# Latency wrappers
# =========================


class QueueLatencyWrapper(gym.Wrapper):
    """
    Queue/scheduled-action latency wrapper for true input delay.

    Every step the policy outputs an action, but that action is scheduled to
    become active `current_frame_skip` frames later. The environment executes the
    most recent matured action, starting from NOOP (= 0).
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip_values: List[int],
        latency_mode: str = "episode_fixed",
        latency_sampling_mode: str = "uniform_all",
        episodes_per_stage: int = 10,
        randomize_within_stage: bool = False,
        within_episode_change_interval: int = 1,
        within_episode_change_prob: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(env)
        assert len(frame_skip_values) > 0, "frame_skip_values cannot be empty"

        if latency_mode == "within_episode_change":
            latency_mode = "within_episode_random"
        assert latency_mode in {"episode_fixed", "within_episode_random", "within_episode_curriculum"}
        assert latency_sampling_mode in {"uniform_all", "curriculum"}

        self.frame_skip_values = [max(0, int(x)) for x in frame_skip_values]
        self.latency_mode = str(latency_mode)
        self.latency_sampling_mode = str(latency_sampling_mode)
        self.episodes_per_stage = max(1, int(episodes_per_stage))
        self.randomize_within_stage = bool(randomize_within_stage)
        self.within_episode_change_interval = max(1, int(within_episode_change_interval))
        self.within_episode_change_prob = float(within_episode_change_prob)

        self.rng = np.random.RandomState(seed)

        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _get_stage_index(self) -> int:
        stage = self.episode_idx // self.episodes_per_stage
        return min(stage, len(self.frame_skip_values) - 1)

    def _sample_frame_skip(self) -> int:
        if self.latency_sampling_mode == "uniform_all":
            return int(self.rng.choice(self.frame_skip_values))

        stage = self._get_stage_index()
        if self.randomize_within_stage:
            candidates = self.frame_skip_values[: stage + 1]
            return int(self.rng.choice(candidates))
        return int(self.frame_skip_values[stage])

    def _current_curriculum_frame_skip(self) -> int:
        stage = self.decision_step_idx // self.within_episode_change_interval
        stage = min(stage, len(self.frame_skip_values) - 1)
        return int(self.frame_skip_values[stage])

    def _maybe_update_frame_skip_for_this_decision(self) -> None:
        if self.latency_mode == "episode_fixed":
            return
        if self.latency_mode == "within_episode_random":
            if self.decision_step_idx == 0:
                return
            if self.decision_step_idx % self.within_episode_change_interval != 0:
                return
            if self.rng.rand() <= self.within_episode_change_prob:
                self.current_frame_skip = self._sample_frame_skip()
            return
        if self.latency_mode == "within_episode_curriculum":
            self.current_frame_skip = self._current_curriculum_frame_skip()
            return

    def _schedule_action(self, action: int) -> int:
        latency_for_action = int(self.current_frame_skip)
        apply_step = self.env_step_idx + latency_for_action
        self._schedule_seq += 1
        insort_right(self.pending_actions, (apply_step, self._schedule_seq, int(action), latency_for_action))
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

        if self.latency_mode == "within_episode_curriculum":
            self.current_frame_skip = int(self.frame_skip_values[0])
        else:
            self.current_frame_skip = self._sample_frame_skip()

        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["latency_mode"] = self.latency_mode
        info["applied_action"] = int(self.current_applied_action)
        info["pending_queue_len"] = 0
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self._maybe_update_frame_skip_for_this_decision()
        scheduled_delay = int(self.current_frame_skip)
        scheduled_apply_step = self._schedule_action(int(action))
        self._advance_pending_actions()

        obs, reward, terminated, truncated, info = self.env.step(int(self.current_applied_action))

        if not isinstance(info, dict):
            info = {}
        info["frame_skip"] = scheduled_delay
        info["decision_step_idx"] = int(self.decision_step_idx)
        info["env_step_idx"] = int(self.env_step_idx)
        info["latency_mode"] = self.latency_mode
        info["applied_action"] = int(self.current_applied_action)
        info["queued_action"] = int(action)
        info["scheduled_apply_step"] = int(scheduled_apply_step)
        info["pending_queue_len"] = int(len(self.pending_actions))

        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


# =========================
# Observation wrappers
# =========================


class EnvFrameStack(gym.Wrapper):
    """
    Channel-wise frame stack at the single-env level so it works for both plain image
    observations and later Dict-based wrappers.
    """

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "EnvFrameStack expects Box obs"
        assert len(self.observation_space.shape) == 3, "EnvFrameStack expects HWC obs"
        assert self.observation_space.dtype == np.uint8, "EnvFrameStack expects uint8 image obs"

        self.n_stack = max(1, int(n_stack))
        h, w, c = self.observation_space.shape
        self.frames: deque[np.ndarray] = deque(maxlen=self.n_stack)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(h, w, c * self.n_stack),
            dtype=np.uint8,
        )

    def _get_stacked(self) -> np.ndarray:
        assert len(self.frames) == self.n_stack
        return np.concatenate(list(self.frames), axis=2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_arr = np.asarray(obs, dtype=np.uint8)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(obs_arr.copy())
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.uint8)
        self.frames.append(obs_arr.copy())
        return self._get_stacked(), reward, terminated, truncated, info


class LatencyImageObsWrapper(ObservationWrapper):
    """
    Appends one grayscale-like channel containing the normalized latency value.
    The channel is filled with a constant uint8 intensity in [0, 255].
    """

    def __init__(self, env: gym.Env, max_frame_skip: int = 15):
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "LatencyImageObsWrapper expects Box obs"
        assert self.observation_space.dtype == np.uint8, "Expected uint8 image observations"
        assert len(self.observation_space.shape) == 3, "Expected HWC image observations"

        h, w, c = self.observation_space.shape
        self.max_frame_skip = max(1, int(max_frame_skip))
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(h, w, c + 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs_arr = np.asarray(obs, dtype=np.uint8)
        frame_skip = int(_unwrap_attr(self.env, "current_frame_skip", 0))
        skip_norm = min(frame_skip / float(self.max_frame_skip), 1.0)
        skip_value = int(round(skip_norm * 255.0))
        latency_plane = np.full(obs_arr.shape[:2] + (1,), skip_value, dtype=np.uint8)
        return np.concatenate([obs_arr, latency_plane], axis=2)


class LatencyScalarDictObsWrapper(ObservationWrapper):
    """
    Dict observation with current image and a separate normalized latency scalar.
    """

    def __init__(self, env: gym.Env, max_frame_skip: int = 15):
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "LatencyScalarDictObsWrapper expects Box obs"
        assert len(self.observation_space.shape) == 3, "Expected HWC image observations"
        image_space = self.observation_space
        self.max_frame_skip = max(1, int(max_frame_skip))
        self.observation_space = DictSpace(
            {
                "image": image_space,
                "latency": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def observation(self, obs):
        frame_skip = int(_unwrap_attr(self.env, "current_frame_skip", 0))
        skip_norm = min(frame_skip / float(self.max_frame_skip), 1.0)
        return {
            "image": np.asarray(obs, dtype=np.uint8),
            "latency": np.array([skip_norm], dtype=np.float32),
        }


class MemoryObservationWrapper(gym.Wrapper):
    """
    Returns a Dict observation containing:
      - current_image: current stacked image observation
      - history_images: fixed-length sequence of previous observations
      - history_actions: fixed-length sequence of previous actions (with padding token)
      - optional latency: normalized current latency scalar

    The sequence dimension is explicit so a custom features extractor can apply
    RNN/LSTM/Transformer over the past observations/actions.
    """

    def __init__(
        self,
        env: gym.Env,
        history_len: int,
        action_dim: int,
        include_latency: bool,
        max_frame_skip: int = 15,
    ):
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "MemoryObservationWrapper expects Box obs"
        assert len(self.observation_space.shape) == 3, "Expected HWC image observations"

        self.history_len = max(1, int(history_len))
        self.action_dim = int(action_dim)
        self.include_latency = bool(include_latency)
        self.max_frame_skip = max(1, int(max_frame_skip))
        self.pad_action = self.action_dim

        image_space = self.observation_space
        h, w, c = image_space.shape
        self._current_image: Optional[np.ndarray] = None
        self.history_images: deque[np.ndarray] = deque(maxlen=self.history_len)
        self.history_actions: deque[int] = deque(maxlen=self.history_len)

        space_dict: Dict[str, gym.Space] = {
            "current_image": image_space,
            "history_images": Box(
                low=0,
                high=255,
                shape=(self.history_len, h, w, c),
                dtype=np.uint8,
            ),
            "history_actions": Box(
                low=0,
                high=self.action_dim,
                shape=(self.history_len,),
                dtype=np.int64,
            ),
        }
        if self.include_latency:
            space_dict["latency"] = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = DictSpace(space_dict)

    def _latency_value(self) -> np.ndarray:
        frame_skip = int(_unwrap_attr(self.env, "current_frame_skip", 0))
        skip_norm = min(frame_skip / float(self.max_frame_skip), 1.0)
        return np.array([skip_norm], dtype=np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        assert self._current_image is not None
        h_imgs = list(self.history_images)
        h_actions = list(self.history_actions)
        while len(h_imgs) < self.history_len:
            h_imgs.insert(0, np.zeros_like(self._current_image, dtype=np.uint8))
        while len(h_actions) < self.history_len:
            h_actions.insert(0, self.pad_action)
        obs: Dict[str, np.ndarray] = {
            "current_image": np.asarray(self._current_image, dtype=np.uint8),
            "history_images": np.stack(h_imgs, axis=0).astype(np.uint8),
            "history_actions": np.asarray(h_actions, dtype=np.int64),
        }
        if self.include_latency:
            obs["latency"] = self._latency_value()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_image = np.asarray(obs, dtype=np.uint8)
        self.history_images.clear()
        self.history_actions.clear()
        return self._build_obs(), info

    def step(self, action):
        assert self._current_image is not None
        prev_current = self._current_image.copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history_images.append(prev_current)
        self.history_actions.append(int(action))
        self._current_image = np.asarray(obs, dtype=np.uint8)
        return self._build_obs(), reward, terminated, truncated, info


# =========================
# Helpers
# =========================


def _unwrap_attr(env: gym.Env, attr_name: str, default):
    cur = env
    while cur is not None:
        if hasattr(cur, attr_name):
            return getattr(cur, attr_name)
        cur = getattr(cur, "env", None)
    return default


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_pair(s: str) -> Tuple[int, int]:
    values = parse_int_list(s)
    if len(values) != 2:
        raise ValueError(f"Expected exactly two integers, got: {s}")
    return int(values[0]), int(values[1])


def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def linear_schedule(initial_value: float, final_value: Optional[float] = None):
    if final_value is None:
        return initial_value

    def scheduler(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return scheduler


def infer_image_space_layout(image_space: Box) -> tuple[str, int, int, int]:
    """
    Infer whether a 3D image space is HWC or CHW.

    Heuristic:
      - if the first dimension looks like a channel count (small) and the
        last two look like spatial dimensions, treat it as CHW
      - otherwise treat it as HWC
    """
    assert isinstance(image_space, Box)
    assert len(image_space.shape) == 3

    d0, d1, d2 = [int(x) for x in image_space.shape]
    if d0 <= 16 and d1 >= 16 and d2 >= 16:
        return "CHW", d0, d1, d2
    return "HWC", d2, d0, d1


def maybe_image_to_chw(x: torch.Tensor, image_space: gym.Space) -> torch.Tensor:
    """
    Converts image tensors to channel-first when needed.
    Supports both HWC/CHW for 4D batched tensors and 5D sequence tensors.
    """
    if not isinstance(image_space, Box) or len(image_space.shape) != 3:
        return x

    layout, c, h, w = infer_image_space_layout(image_space)

    if x.ndim == 4:
        if x.shape[1] == c and x.shape[2] == h and x.shape[3] == w:
            return x
        if x.shape[-1] == c and x.shape[1] == h and x.shape[2] == w:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    if x.ndim == 5:
        if x.shape[2] == c and x.shape[3] == h and x.shape[4] == w:
            return x
        if x.shape[-1] == c and x.shape[2] == h and x.shape[3] == w:
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x

    return x


def compute_cnn_out_dim(conv_net: nn.Module, image_space: Box) -> int:
    layout, c, h, w = infer_image_space_layout(image_space)
    with torch.no_grad():
        sample = torch.zeros((1, c, h, w), dtype=torch.float32)
        return int(conv_net(sample).shape[1])


def build_small_cnn_from_space(image_space: Box, feature_dim: int) -> nn.Module:
    assert isinstance(image_space, Box)
    assert len(image_space.shape) == 3

    _, in_channels, _, _ = infer_image_space_layout(image_space)
    conv_net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    conv_out_dim = compute_cnn_out_dim(conv_net, image_space)
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(conv_out_dim, feature_dim),
        nn.ReLU(),
    )


def serialize_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): serialize_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serialize_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# =========================
# Custom feature extractors
# =========================


class LateFusionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: DictSpace,
        image_feature_dim: int = 512,
        latency_hidden_dims: Sequence[int] = (32, 32),
    ):
        assert isinstance(observation_space, DictSpace)
        self.image_space = observation_space.spaces["image"]
        assert isinstance(self.image_space, Box)
        super().__init__(observation_space, features_dim=1)

        self.image_cnn = build_small_cnn_from_space(self.image_space, feature_dim=image_feature_dim)

        latency_layers: List[nn.Module] = []
        last_dim = 1
        for dim in latency_hidden_dims:
            latency_layers.append(nn.Linear(last_dim, int(dim)))
            latency_layers.append(nn.ReLU())
            last_dim = int(dim)
        self.latency_mlp = nn.Sequential(*latency_layers) if latency_layers else nn.Identity()
        self.latency_out_dim = last_dim
        self._features_dim = int(image_feature_dim + self.latency_out_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["image"].float()
        image = maybe_image_to_chw(image, self.image_space)
        image = image / 255.0
        image_features = self.image_cnn(image)

        latency = observations["latency"].float()
        latency_features = self.latency_mlp(latency)
        return torch.cat([image_features, latency_features], dim=1)


class MemoryExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: DictSpace,
        action_dim: int,
        history_len: int,
        memory_model: str = "lstm",
        image_feature_dim: int = 256,
        action_embed_dim: int = 32,
        memory_hidden_dim: int = 256,
        memory_num_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
        latency_hidden_dims: Sequence[int] = (32, 32),
        use_latency: bool = False,
    ):
        assert isinstance(observation_space, DictSpace)
        self.current_image_space = observation_space.spaces["current_image"]
        self.history_image_space = observation_space.spaces["history_images"]
        assert isinstance(self.current_image_space, Box)
        assert isinstance(self.history_image_space, Box)

        super().__init__(observation_space, features_dim=1)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.memory_model = str(memory_model)
        self.memory_hidden_dim = int(memory_hidden_dim)
        self.use_latency = bool(use_latency and ("latency" in observation_space.spaces))
        self.pad_action = self.action_dim

        self.image_cnn = build_small_cnn_from_space(self.current_image_space, feature_dim=image_feature_dim)
        self.action_embed = nn.Embedding(self.action_dim + 1, int(action_embed_dim), padding_idx=self.pad_action)
        seq_input_dim = int(image_feature_dim + action_embed_dim)

        if self.memory_model == "rnn":
            self.sequence_model = nn.RNN(
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
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.memory_hidden_dim,
                nhead=int(transformer_heads),
                dim_feedforward=int(transformer_ff_dim),
                dropout=float(transformer_dropout),
                batch_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(memory_num_layers),
                enable_nested_tensor=False,
            )
        else:
            raise ValueError(f"Unsupported memory_model: {self.memory_model}")

        self.current_proj = nn.Linear(image_feature_dim, self.memory_hidden_dim)

        latency_out_dim = 0
        if self.use_latency:
            latency_layers: List[nn.Module] = []
            last_dim = 1
            for dim in latency_hidden_dims:
                latency_layers.append(nn.Linear(last_dim, int(dim)))
                latency_layers.append(nn.ReLU())
                last_dim = int(dim)
            self.latency_mlp = nn.Sequential(*latency_layers) if latency_layers else nn.Identity()
            latency_out_dim = last_dim
        else:
            self.latency_mlp = None

        self._features_dim = int(self.memory_hidden_dim * 2 + latency_out_dim)

    def _encode_images(self, images: torch.Tensor, image_space: Box) -> torch.Tensor:
        images = images.float()
        if images.ndim == 4:
            images = maybe_image_to_chw(images, image_space)
            images = images / 255.0
            return self.image_cnn(images)
        if images.ndim == 5:
            images = maybe_image_to_chw(images, image_space)
            if images.ndim != 5:
                raise ValueError(f"Unexpected sequence image tensor shape after conversion: {tuple(images.shape)}")
            b, t, c, h, w = images.shape
            flat = images.reshape(b * t, c, h, w).contiguous().float() / 255.0
            enc = self.image_cnn(flat)
            return enc.view(b, t, -1)
        raise ValueError(f"Unexpected image tensor shape: {tuple(images.shape)}")

    def _gather_last_valid(self, seq_out: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        lengths = valid_mask.long().sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
        gathered = seq_out[torch.arange(seq_out.shape[0], device=seq_out.device), last_idx]
        zero_mask = (lengths == 0).unsqueeze(1)
        return torch.where(zero_mask, torch.zeros_like(gathered), gathered)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        current_image = observations["current_image"]
        history_images = observations["history_images"]
        history_actions = observations["history_actions"].long()

        current_feat = self._encode_images(current_image, self.current_image_space)
        current_feat = self.current_proj(current_feat)

        history_img_feat = self._encode_images(history_images, self.current_image_space)
        action_feat = self.action_embed(history_actions)
        seq_inputs = torch.cat([history_img_feat, action_feat], dim=-1)

        valid_mask = history_actions != self.pad_action

        if self.memory_model in {"rnn", "lstm"}:
            seq_out, _ = self.sequence_model(seq_inputs)
            memory_feat = self._gather_last_valid(seq_out, valid_mask)
        else:
            seq_proj = self.sequence_proj(seq_inputs) + self.pos_embedding[:, : seq_inputs.shape[1], :]
            key_padding_mask = ~valid_mask

            # PyTorch TransformerEncoder can fail when an entire sequence is padding.
            # This happens at the beginning of episodes before any past actions exist.
            # For those rows, force the first token to be attendable with a zero vector.
            all_pad_rows = key_padding_mask.all(dim=1)
            if all_pad_rows.any():
                seq_proj = seq_proj.clone()
                key_padding_mask = key_padding_mask.clone()
                seq_proj[all_pad_rows, 0, :] = 0.0
                key_padding_mask[all_pad_rows, 0] = False

            seq_out = self.sequence_model(seq_proj, src_key_padding_mask=key_padding_mask)
            memory_feat = self._gather_last_valid(seq_out, valid_mask)

        pieces = [current_feat, memory_feat]
        if self.use_latency and self.latency_mlp is not None:
            latency_feat = self.latency_mlp(observations["latency"].float())
            pieces.append(latency_feat)
        return torch.cat(pieces, dim=1)


# =========================
# Env builders
# =========================


def maybe_apply_obs_mode(
    env: gym.Env,
    obs_mode: str,
    max_frame_skip: int,
    history_len: int,
    memory_use_latency: bool,
    memory_latency_repr: str = "separate",
) -> gym.Env:
    if obs_mode == "latency_channel":
        return LatencyImageObsWrapper(env, max_frame_skip=max_frame_skip)
    if obs_mode == "image_only":
        return env
    if obs_mode == "late_fusion":
        return LatencyScalarDictObsWrapper(env, max_frame_skip=max_frame_skip)
    if obs_mode == "memory":
        assert isinstance(env.action_space, gym.spaces.Discrete), "Memory mode expects discrete action space"
        if memory_latency_repr == "channel":
            env = LatencyImageObsWrapper(env, max_frame_skip=max_frame_skip)
            return MemoryObservationWrapper(
                env,
                history_len=history_len,
                action_dim=int(env.action_space.n),
                include_latency=False,
                max_frame_skip=max_frame_skip,
            )
        if memory_latency_repr == "none":
            return MemoryObservationWrapper(
                env,
                history_len=history_len,
                action_dim=int(env.action_space.n),
                include_latency=False,
                max_frame_skip=max_frame_skip,
            )
        return MemoryObservationWrapper(
            env,
            history_len=history_len,
            action_dim=int(env.action_space.n),
            include_latency=memory_use_latency,
            max_frame_skip=max_frame_skip,
        )
    raise ValueError(f"Unsupported obs_mode: {obs_mode}")


def make_env(
    seed: int,
    env_id: str = "ALE/DemonAttack-v5",
    obs_mode: str = "latency_channel",
    frame_skip_values: Optional[List[int]] = None,
    episodes_per_stage: int = 10,
    randomize_within_stage: bool = False,
    latency_mode: str = "episode_fixed",
    latency_sampling_mode: str = "uniform_all",
    within_episode_change_interval: int = 1,
    within_episode_change_prob: float = 1.0,
    screen_size: int = 84,
    noop_max: int = 30,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    stack_frames: int = 4,
    history_len: int = 8,
    memory_use_latency: bool = False,
    memory_latency_repr: str = "separate",
):
    def _init():
        env = gym.make(
            env_id,
            obs_type="rgb",
            frameskip=1,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            render_mode=None,
        )

        env = AtariWrapper(
            env,
            noop_max=noop_max,
            frame_skip=1,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            clip_reward=clip_reward,
            action_repeat_probability=0.0,
        )

        latency_values = [0] if frame_skip_values is None else frame_skip_values
        env = QueueLatencyWrapper(
            env,
            frame_skip_values=latency_values,
            latency_mode=latency_mode,
            latency_sampling_mode=latency_sampling_mode,
            episodes_per_stage=episodes_per_stage,
            randomize_within_stage=randomize_within_stage,
            within_episode_change_interval=within_episode_change_interval,
            within_episode_change_prob=within_episode_change_prob,
            seed=seed,
        )

        if stack_frames > 1:
            env = EnvFrameStack(env, n_stack=stack_frames)

        env = maybe_apply_obs_mode(
            env,
            obs_mode=obs_mode,
            max_frame_skip=max(latency_values),
            history_len=history_len,
            memory_use_latency=memory_use_latency,
            memory_latency_repr=memory_latency_repr,
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_demon_attack(
    n_envs: int,
    seed: int,
    env_id: str = "ALE/DemonAttack-v5",
    obs_mode: str = "latency_channel",
    frame_skip_values: Optional[List[int]] = None,
    episodes_per_stage: int = 10,
    randomize_within_stage: bool = False,
    latency_mode: str = "episode_fixed",
    latency_sampling_mode: str = "uniform_all",
    within_episode_change_interval: int = 1,
    within_episode_change_prob: float = 1.0,
    screen_size: int = 84,
    noop_max: int = 30,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    stack_frames: int = 4,
    history_len: int = 8,
    memory_use_latency: bool = False,
    memory_latency_repr: str = "separate",
):
    env_fns = [
        make_env(
            seed=seed + i,
            env_id=env_id,
            obs_mode=obs_mode,
            frame_skip_values=frame_skip_values,
            episodes_per_stage=episodes_per_stage,
            randomize_within_stage=randomize_within_stage,
            latency_mode=latency_mode,
            latency_sampling_mode=latency_sampling_mode,
            within_episode_change_interval=within_episode_change_interval,
            within_episode_change_prob=within_episode_change_prob,
            screen_size=screen_size,
            noop_max=noop_max,
            clip_reward=clip_reward,
            terminal_on_life_loss=terminal_on_life_loss,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            stack_frames=stack_frames,
            history_len=history_len,
            memory_use_latency=memory_use_latency,
            memory_latency_repr=memory_latency_repr,
        )
        for i in range(n_envs)
    ]
    return SubprocVecEnv(env_fns, start_method="fork")


# =========================
# Train / eval / export
# =========================


def build_policy_config(args, sample_env: gym.Env):
    obs_space = sample_env.observation_space
    if args.include_latency_obs and args.obs_mode == "image_only":
        obs_mode = "latency_channel"
    else:
        obs_mode = args.obs_mode

    if obs_mode in {"latency_channel", "image_only"}:
        policy = "CnnPolicy"
        policy_kwargs = dict(normalize_images=True)
        return policy, policy_kwargs, obs_mode

    if obs_mode == "late_fusion":
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            normalize_images=True,
            features_extractor_class=LateFusionExtractor,
            features_extractor_kwargs=dict(
                image_feature_dim=args.image_feature_dim,
                latency_hidden_dims=parse_int_list(args.latency_hidden_dims),
            ),
        )
        return policy, policy_kwargs, obs_mode

    if obs_mode == "memory":
        action_dim = int(sample_env.action_space.n)
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            normalize_images=True,
            features_extractor_class=MemoryExtractor,
            features_extractor_kwargs=dict(
                action_dim=action_dim,
                history_len=args.history_len,
                memory_model=args.memory_model,
                image_feature_dim=args.image_feature_dim,
                action_embed_dim=args.action_embed_dim,
                memory_hidden_dim=args.memory_hidden_dim,
                memory_num_layers=args.memory_num_layers,
                transformer_heads=args.transformer_heads,
                transformer_ff_dim=args.transformer_ff_dim,
                transformer_dropout=args.transformer_dropout,
                latency_hidden_dims=parse_int_list(args.latency_hidden_dims),
                use_latency=(args.memory_use_latency and args.memory_latency_repr == "separate"),
            ),
        )
        return policy, policy_kwargs, obs_mode

    raise ValueError(f"Unsupported obs_mode: {obs_mode}")


def train(args, use_wandb: bool, tb_log: Optional[str]):
    set_random_seed(args.seed)
    frame_skip_values = parse_int_list(args.frame_skip_values)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    venv = make_vec_demon_attack(
        n_envs=args.n_envs,
        seed=args.seed,
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        frame_skip_values=frame_skip_values,
        episodes_per_stage=args.episodes_per_stage,
        randomize_within_stage=args.randomize_within_stage,
        latency_mode=args.latency_mode,
        latency_sampling_mode=args.latency_sampling_mode,
        within_episode_change_interval=args.within_episode_change_interval,
        within_episode_change_prob=args.within_episode_change_prob,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        clip_reward=args.clip_reward,
        terminal_on_life_loss=args.terminal_on_life_loss,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=args.full_action_space,
        stack_frames=args.stack_frames,
        history_len=args.history_len,
        memory_use_latency=args.memory_use_latency,
        memory_latency_repr=args.memory_latency_repr,
    )

    # Build policy config from a single non-vectorized env to inspect spaces/action dim.
    sample_env = make_env(
        seed=args.seed,
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        frame_skip_values=frame_skip_values,
        episodes_per_stage=args.episodes_per_stage,
        randomize_within_stage=args.randomize_within_stage,
        latency_mode=args.latency_mode,
        latency_sampling_mode=args.latency_sampling_mode,
        within_episode_change_interval=args.within_episode_change_interval,
        within_episode_change_prob=args.within_episode_change_prob,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        clip_reward=args.clip_reward,
        terminal_on_life_loss=args.terminal_on_life_loss,
        repeat_action_probability=args.repeat_action_probability,
        full_action_space=args.full_action_space,
        stack_frames=args.stack_frames,
        history_len=args.history_len,
        memory_use_latency=args.memory_use_latency,
        memory_latency_repr=args.memory_latency_repr,
    )()
    policy, policy_kwargs, resolved_obs_mode = build_policy_config(args, sample_env)
    sample_env.close()

    learning_rate = linear_schedule(args.learning_rate, args.learning_rate_final)
    clip_range = linear_schedule(args.clip_range, args.clip_range_final)

    if args.load:
        print(f"[train] loading from {args.load}")
        model = PPO.load(
            args.load,
            env=venv,
            device=args.device,
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": learning_rate,
                "clip_range": clip_range,
            },
        )
    else:
        model = PPO(
            policy,
            venv,
            verbose=1,
            device=args.device,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            clip_range=clip_range,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=(tb_log if (use_wandb and tb_log) else None),
            policy_kwargs=policy_kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=os.path.dirname(args.save) or ".",
        name_prefix=os.path.basename(args.save).replace(".zip", ""),
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=int(args.train_steps),
        callback=checkpoint_callback,
        log_interval=1,
        tb_log_name=(args.wandb_run_name or f"demon_attack_{resolved_obs_mode}"),
    )

    model.save(args.save)
    venv.close()
    return args.save


def eval_model(
    model_path: str,
    episodes: int,
    seed: int,
    env_id: str = "ALE/DemonAttack-v5",
    eval_frame_skip: int = 0,
    obs_mode: str = "latency_channel",
    history_len: int = 8,
    memory_use_latency: bool = False,
    memory_latency_repr: str = "separate",
    screen_size: int = 84,
    noop_max: int = 30,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    stack_frames: int = 4,
    max_steps_per_episode: int = 0,
):
    model = PPO.load(model_path, device="cpu")

    venv = make_vec_demon_attack(
        n_envs=1,
        seed=seed,
        env_id=env_id,
        obs_mode=obs_mode,
        frame_skip_values=[eval_frame_skip],
        episodes_per_stage=10**9,
        randomize_within_stage=False,
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        screen_size=screen_size,
        noop_max=noop_max,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        stack_frames=stack_frames,
        history_len=history_len,
        memory_use_latency=memory_use_latency,
        memory_latency_repr=memory_latency_repr,
    )

    returns: List[float] = []
    lengths: List[int] = []
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while (not done) and (max_steps_per_episode <= 0 or ep_len < max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, infos = venv.step(action)
            done = bool(done_vec[0])
            ep_ret += float(reward[0])
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    venv.close()
    return returns, lengths


def export_rollouts(
    model_path: str,
    outdir: str,
    episodes: int,
    max_steps: int,
    seed: int,
    env_id: str = "ALE/DemonAttack-v5",
    export_frame_skip: int = 0,
    obs_mode: str = "latency_channel",
    history_len: int = 8,
    memory_use_latency: bool = False,
    memory_latency_repr: str = "separate",
    screen_size: int = 84,
    noop_max: int = 30,
    clip_reward: bool = True,
    terminal_on_life_loss: bool = False,
    repeat_action_probability: float = 0.0,
    full_action_space: bool = False,
    stack_frames: int = 4,
):
    os.makedirs(outdir, exist_ok=True)
    meta_root = os.path.join(outdir, "metadata")
    os.makedirs(meta_root, exist_ok=True)

    model = PPO.load(model_path, device="cpu")

    venv = make_vec_demon_attack(
        n_envs=1,
        seed=seed,
        env_id=env_id,
        obs_mode=obs_mode,
        frame_skip_values=[export_frame_skip],
        episodes_per_stage=10**9,
        randomize_within_stage=False,
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        screen_size=screen_size,
        noop_max=noop_max,
        clip_reward=clip_reward,
        terminal_on_life_loss=terminal_on_life_loss,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        stack_frames=stack_frames,
        history_len=history_len,
        memory_use_latency=memory_use_latency,
        memory_latency_repr=memory_latency_repr,
    )

    for ep in range(episodes):
        meta_path = os.path.join(meta_root, f"episode_{ep:04d}.jsonl")
        obs = venv.reset()
        done = False
        t = 0

        with open(meta_path, "w", encoding="utf-8") as mf:
            while (not done) and t < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward_vec, done_vec, info_vec = venv.step(action)
                done = bool(done_vec[0])
                reward = float(reward_vec[0])
                info = info_vec[0] if isinstance(info_vec, (list, tuple)) else {}

                record = {
                    "t": int(t),
                    "obs": serialize_jsonable(obs[0]),
                    "action": int(action[0]) if isinstance(action, np.ndarray) else int(action),
                    "reward": reward,
                    "done": done,
                    "info": serialize_jsonable(info if isinstance(info, dict) else {}),
                }
                mf.write(json.dumps(record) + "\n")
                obs = next_obs
                t += 1

    venv.close()
    return outdir


# =========================
# Main
# =========================


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--env-id", type=str, default="ALE/DemonAttack-v5")
    ap.add_argument("--train-steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="./models/ppo_demon_attack_queue_latency.zip")
    ap.add_argument("--load", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--checkpoint-freq", type=int, default=25_000)

    ap.add_argument("--eval-episodes", type=int, default=0)
    ap.add_argument(
        "--eval-frame-skip-values",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated latency values to evaluate on",
    )

    ap.add_argument("--export-episodes", type=int, default=0)
    ap.add_argument("--export-max-steps", type=int, default=5000)
    ap.add_argument("--export-frame-skip", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./demon_attack_rollouts")

    # PPO hyperparameters
    ap.add_argument("--learning-rate", type=float, default=2.5e-4)
    ap.add_argument("--learning-rate-final", type=float, default=2.5e-5)
    ap.add_argument("--clip-range", type=float, default=0.1)
    ap.add_argument("--clip-range-final", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)

    # Atari preprocessing
    ap.add_argument("--screen-size", type=int, default=84)
    ap.add_argument("--noop-max", type=int, default=30)
    ap.add_argument("--stack-frames", type=int, default=4)
    ap.add_argument("--clip-reward", action="store_true")
    ap.add_argument("--terminal-on-life-loss", action="store_true")
    ap.add_argument("--repeat-action-probability", type=float, default=0.0)
    ap.add_argument("--full-action-space", action="store_true")

    # Observation/model mode
    ap.add_argument(
        "--obs-mode",
        type=str,
        default="latency_channel",
        choices=["latency_channel", "image_only", "late_fusion", "memory"],
        help=(
            "latency_channel = append latency as image channel; "
            "image_only = no explicit latency; "
            "late_fusion = separate CNN image + latency MLP; "
            "memory = current image + history of obs/actions with RNN/LSTM/Transformer"
        ),
    )
    ap.add_argument(
        "--include-latency-obs",
        action="store_true",
        help="Legacy alias. If used with --obs-mode image_only, it switches to latency_channel.",
    )
    ap.add_argument("--image-feature-dim", type=int, default=256)
    ap.add_argument("--latency-hidden-dims", type=str, default="32,32")

    # Memory mode
    ap.add_argument("--history-len", type=int, default=8)
    ap.add_argument("--memory-model", type=str, default="lstm", choices=["rnn", "lstm", "transformer"])
    ap.add_argument("--memory-use-latency", action="store_true")
    ap.add_argument(
        "--memory-latency-repr",
        type=str,
        default="separate",
        choices=["none", "separate", "channel"],
        help=(
            "How latency is represented in memory mode: "
            "none = no explicit latency, "
            "separate = latency scalar/MLP fusion, "
            "channel = append latency as an image channel before building memory history"
        ),
    )
    ap.add_argument("--action-embed-dim", type=int, default=32)
    ap.add_argument("--memory-hidden-dim", type=int, default=256)
    ap.add_argument("--memory-num-layers", type=int, default=1)
    ap.add_argument("--transformer-heads", type=int, default=4)
    ap.add_argument("--transformer-ff-dim", type=int, default=512)
    ap.add_argument("--transformer-dropout", type=float, default=0.1)

    # Latency settings
    ap.add_argument(
        "--frame-skip-values",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated action-delay values in frames. For delay K, action chosen at t is applied at t+K.",
    )
    ap.add_argument("--episodes-per-stage", type=int, default=10)
    ap.add_argument("--randomize-within-stage", action="store_true")
    ap.add_argument(
        "--latency-mode",
        type=str,
        default="episode_fixed",
        choices=["episode_fixed", "within_episode_random", "within_episode_curriculum", "within_episode_change"],
        help=(
            "episode_fixed: one latency per episode; "
            "within_episode_random: resample latency inside episode; "
            "within_episode_curriculum: increase latency step-by-step through provided range"
        ),
    )
    ap.add_argument(
        "--latency-sampling-mode",
        type=str,
        default="uniform_all",
        choices=["uniform_all", "curriculum"],
        help="For episode_fixed / within_episode_random sampling across the provided latency values",
    )
    ap.add_argument(
        "--within-episode-change-interval",
        type=int,
        default=1,
        help=(
            "For within_episode_random: number of decision steps between possible resamples. "
            "For within_episode_curriculum: number of decision steps spent at each latency before moving to the next one."
        ),
    )
    ap.add_argument("--within-episode-change-prob", type=float, default=1.0)

    # Logging
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="demon_attack_rl_queue_latency")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--eval-max-steps", type=int, default=0)

    args = ap.parse_args()

    if args.device == "auto":
        try:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    if args.include_latency_obs and args.obs_mode == "image_only":
        args.obs_mode = "latency_channel"

    set_random_seed(args.seed)

    use_wandb = bool(args.wandb)
    tb_log = "./tb_logs" if use_wandb else None
    if use_wandb:
        try:
            import wandb
        except Exception as e:
            print(f"[wandb] wandb not installed or failed to import ({e}). Disabling wandb.")
            use_wandb = False
            tb_log = None
        else:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=(args.wandb_run_name or f"demon_attack_queue_{int(time.time())}"),
                tags=tags,
                config=vars(args),
                sync_tensorboard=True,
                save_code=True,
            )

    model_path = args.load or args.save
    if args.train_steps > 0:
        print(
            f"[train] steps={args.train_steps} n_envs={args.n_envs} save={args.save} "
            f"obs_mode={args.obs_mode} latency_mode={args.latency_mode}"
        )
        model_path = train(args, use_wandb, tb_log)
        print(f"[train] saved => {model_path}")

    if args.eval_episodes > 0:
        eval_frame_skips = parse_int_list(args.eval_frame_skip_values)
        for fs in eval_frame_skips:
            returns, lengths = eval_model(
                model_path=model_path,
                episodes=args.eval_episodes,
                seed=args.seed,
                env_id=args.env_id,
                eval_frame_skip=fs,
                obs_mode=args.obs_mode,
                history_len=args.history_len,
                memory_use_latency=args.memory_use_latency,
                memory_latency_repr=args.memory_latency_repr,
                screen_size=args.screen_size,
                noop_max=args.noop_max,
                clip_reward=args.clip_reward,
                terminal_on_life_loss=args.terminal_on_life_loss,
                repeat_action_probability=args.repeat_action_probability,
                full_action_space=args.full_action_space,
                stack_frames=args.stack_frames,
                max_steps_per_episode=args.eval_max_steps,
            )
            print(
                f"[eval] frame_skip={fs} "
                f"mean_return={float(np.mean(returns)):.3f} std_return={float(np.std(returns)):.3f} "
                f"mean_len={float(np.mean(lengths)):.1f}"
            )

    if args.export_episodes > 0:
        print(f"[export] episodes={args.export_episodes} outdir={args.outdir}")
        out = export_rollouts(
            model_path=model_path,
            outdir=args.outdir,
            episodes=int(args.export_episodes),
            max_steps=int(args.export_max_steps),
            seed=args.seed,
            env_id=args.env_id,
            export_frame_skip=args.export_frame_skip,
            obs_mode=args.obs_mode,
            history_len=args.history_len,
            memory_use_latency=args.memory_use_latency,
            memory_latency_repr=args.memory_latency_repr,
            screen_size=args.screen_size,
            noop_max=args.noop_max,
            clip_reward=args.clip_reward,
            terminal_on_life_loss=args.terminal_on_life_loss,
            repeat_action_probability=args.repeat_action_probability,
            full_action_space=args.full_action_space,
            stack_frames=args.stack_frames,
        )
        print(f"[export] wrote => {out}")

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
