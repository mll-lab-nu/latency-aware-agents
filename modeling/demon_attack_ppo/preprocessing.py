from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from modeling.demon_attack_ppo.observation import DemonAttackPpoObservationConfig


def latency_channel_value(frame_skip: int | float, max_frame_skip: int) -> np.uint8:
    max_skip = max(1, int(max_frame_skip))
    value = min(max(float(frame_skip), 0.0) / float(max_skip), 1.0)
    return np.uint8(np.rint(value * 255.0))


def latency_channel(
    height: int,
    width: int,
    frame_skip: int | float,
    max_frame_skip: int,
) -> np.ndarray:
    value = latency_channel_value(frame_skip, max_frame_skip)
    return np.full((int(height), int(width), 1), value, dtype=np.uint8)


def append_latency_channel(
    stacked_obs: np.ndarray,
    frame_skip: int | float,
    max_frame_skip: int,
) -> np.ndarray:
    obs = np.asarray(stacked_obs, dtype=np.uint8)
    return np.concatenate(
        [
            obs,
            latency_channel(
                obs.shape[0],
                obs.shape[1],
                frame_skip=frame_skip,
                max_frame_skip=max_frame_skip,
            ),
        ],
        axis=2,
    )


def resize_atari_batch_torch(
    raw_batch: torch.Tensor,
    image_size: int,
    base_channels: int = 1,
) -> torch.Tensor:
    """Resize NHW or NHWC uint8 Atari frames into NHWC uint8 model frames."""

    if not isinstance(raw_batch, torch.Tensor):
        raise TypeError("raw_batch must be a torch.Tensor")
    if raw_batch.dtype != torch.uint8:
        raise ValueError("raw_batch must have dtype torch.uint8")
    if raw_batch.ndim == 3:
        raw_batch = raw_batch.unsqueeze(-1)
    if raw_batch.ndim != 4:
        raise ValueError("raw_batch must have shape (N, H, W) or (N, H, W, C)")

    n_envs, _, _, channels = raw_batch.shape
    image_size = int(image_size)
    base_channels = int(base_channels)
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if base_channels not in {1, 3}:
        raise ValueError("base_channels must be 1 or 3")
    if int(channels) not in {1, 3}:
        raise ValueError("raw_batch channel count must be 1 or 3")

    nchw = raw_batch.permute(0, 3, 1, 2).to(torch.float32)
    if int(channels) != base_channels:
        if base_channels == 1 and int(channels) == 3:
            r = nchw[:, 0:1]
            g = nchw[:, 1:2]
            b = nchw[:, 2:3]
            nchw = torch.floor((r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0)
        elif base_channels == 3 and int(channels) == 1:
            nchw = nchw.expand(int(n_envs), 3, nchw.shape[2], nchw.shape[3])
        else:
            raise ValueError("unsupported channel conversion")

    resized = F.interpolate(
        nchw,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        resized.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def append_latency_channel_torch(
    stacked_obs: torch.Tensor,
    frame_skips: np.ndarray | torch.Tensor,
    max_frame_skip: int,
) -> torch.Tensor:
    """Append a latency channel to a NHWC uint8 stacked observation batch."""

    if not isinstance(stacked_obs, torch.Tensor):
        raise TypeError("stacked_obs must be a torch.Tensor")
    if stacked_obs.ndim != 4:
        raise ValueError("stacked_obs must have shape (N, H, W, C)")
    if stacked_obs.dtype != torch.uint8:
        raise ValueError("stacked_obs must have dtype torch.uint8")

    n_envs, height, width, _ = stacked_obs.shape
    skips = torch.as_tensor(frame_skips, device=stacked_obs.device).reshape(-1)
    if int(skips.numel()) != int(n_envs):
        raise ValueError("frame_skips must have one value per environment")

    max_skip = max(1, int(max_frame_skip))
    values = torch.clamp(skips.to(torch.float32) / float(max_skip), 0.0, 1.0)
    values = torch.round(values * 255.0).to(torch.uint8)
    latency = values.view(n_envs, 1, 1, 1).expand(n_envs, height, width, 1)
    return torch.cat([stacked_obs, latency], dim=3)


def reset_frame_stack(frame_obs: np.ndarray, stack_frames: int) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
    if frame.ndim == 2:
        frame = frame[..., None]
    return np.concatenate([frame.copy() for _ in range(max(1, int(stack_frames)))], axis=2)


def step_frame_stack(
    stacked_obs: np.ndarray,
    frame_obs: np.ndarray,
    base_channels: int | None = None,
    reset: bool = False,
) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
    if frame.ndim == 2:
        frame = frame[..., None]
    if reset:
        if base_channels is None:
            channels = int(np.asarray(stacked_obs).shape[2])
            base_channels = frame.shape[2]
            stack_frames = max(1, channels // base_channels)
        else:
            stack_frames = max(1, int(np.asarray(stacked_obs).shape[2]) // int(base_channels))
        return reset_frame_stack(frame, stack_frames)

    stack = np.asarray(stacked_obs, dtype=np.uint8)
    channels = int(base_channels or frame.shape[2])
    if stack.shape[:2] != frame.shape[:2]:
        raise ValueError("stacked_obs and frame_obs must have matching height/width")
    if stack.shape[2] % channels != 0:
        raise ValueError("stacked_obs channel count must be divisible by base_channels")
    if stack.shape[2] == channels:
        return frame.copy()
    return np.concatenate([stack[:, :, channels:].copy(), frame.copy()], axis=2)


def build_stacked_observation(
    stacked_obs: np.ndarray,
    observation_config: DemonAttackPpoObservationConfig,
    frame_skip: int | float = 0,
) -> np.ndarray:
    if observation_config.obs_mode == "image_only":
        return np.asarray(stacked_obs, dtype=np.uint8)
    return append_latency_channel(
        stacked_obs,
        frame_skip=frame_skip,
        max_frame_skip=observation_config.max_frame_skip,
    )


def hwc_to_nchw_tensor(
    obs: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    arr = np.asarray(obs, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError("obs must be HWC")
    tensor = torch.as_tensor(arr, dtype=torch.uint8, device=device)
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


class DemonAttackPpoTorchObservationPipeline:
    """Batch Atari-frame preprocessing pipeline shared by training and policy evaluation."""

    def __init__(
        self,
        observation_config: DemonAttackPpoObservationConfig,
        n_envs: int,
        device: str | torch.device,
    ) -> None:
        self.observation_config = observation_config
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.device = torch.device(device)
        self.base_channels = observation_config.base_channels
        self.stacked_channels = observation_config.stacked_channels
        self._stack = torch.empty(
            (
                self.n_envs,
                observation_config.image_size,
                observation_config.image_size,
                self.stacked_channels,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self._last_observation: torch.Tensor | None = None

    @property
    def current_frame_stack(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return self._stack.detach().cpu().numpy().copy()

    @property
    def current_observation(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return self._last_observation.detach().cpu().numpy().copy()

    def reset(
        self,
        raw_batch: torch.Tensor,
        frame_skips: np.ndarray | torch.Tensor | None = None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_batch, profiler=profiler)
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            for stack_idx in range(self.observation_config.stack_frames):
                c0 = stack_idx * self.base_channels
                c1 = c0 + self.base_channels
                self._stack[:, :, :, c0:c1] = frame_obs
        return self._final_obs(frame_skips, profiler=profiler)

    def step(
        self,
        raw_batch: torch.Tensor,
        dones: np.ndarray,
        frame_skips: np.ndarray | torch.Tensor | None = None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_batch, profiler=profiler)
        done_indices = np.flatnonzero(np.asarray(dones, dtype=np.bool_))
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            if self.observation_config.stack_frames <= 1:
                self._stack[:, :, :, :] = frame_obs
            else:
                self._stack[:, :, :, : -self.base_channels] = self._stack[
                    :, :, :, self.base_channels :
                ].clone()
                self._stack[:, :, :, -self.base_channels :] = frame_obs
            for idx in done_indices:
                for stack_idx in range(self.observation_config.stack_frames):
                    c0 = stack_idx * self.base_channels
                    c1 = c0 + self.base_channels
                    self._stack[idx, :, :, c0:c1] = frame_obs[idx]
        return self._final_obs(frame_skips, profiler=profiler)

    def _resize(self, raw_batch: torch.Tensor, profiler: Any | None = None) -> torch.Tensor:
        if not isinstance(raw_batch, torch.Tensor):
            raise TypeError("raw_batch must be a torch.Tensor")
        raw_batch = raw_batch.to(device=self.device, non_blocking=True)
        if int(raw_batch.shape[0]) != self.n_envs:
            raise ValueError("raw_batch must have one frame per environment")
        timer = profiler.time("torch_resize_s") if profiler is not None else nullcontext()
        with timer:
            return resize_atari_batch_torch(
                raw_batch,
                image_size=self.observation_config.image_size,
                base_channels=self.observation_config.base_channels,
            )

    def _final_obs(
        self,
        frame_skips: np.ndarray | torch.Tensor | None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        timer = profiler.time("final_obs_s") if profiler is not None else nullcontext()
        with timer:
            if frame_skips is None:
                frame_skips = torch.zeros(self.n_envs, dtype=torch.int64, device=self.device)
            obs = self._stack
            if self.observation_config.obs_mode == "latency_channel":
                obs = append_latency_channel_torch(
                    obs,
                    frame_skips=frame_skips,
                    max_frame_skip=self.observation_config.max_frame_skip,
                )
            self._last_observation = obs
            return obs.permute(0, 3, 1, 2).contiguous()


def _extract_frame(frame_or_observation: Any) -> Any:
    if hasattr(frame_or_observation, "data"):
        return getattr(frame_or_observation, "data")
    return frame_or_observation


def extract_frame_skip(frame_or_observation: Any, default: int | float = 0) -> int | float:
    metadata = getattr(frame_or_observation, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("frame_skip", metadata.get("latency_frame_skip", default))
    return default
