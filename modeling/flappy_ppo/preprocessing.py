from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modeling.flappy_ppo.observation import FlappyPpoObservationConfig


def resize_frame_pil(
    frame: np.ndarray,
    image_size: int = 84,
    grayscale: bool = False,
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr)
    image = image.convert("L" if grayscale else "RGB")
    target_size = (int(image_size), int(image_size))
    if image.size != target_size:
        image = image.resize(target_size, Image.BILINEAR)

    out = np.asarray(image, dtype=np.uint8)
    if grayscale:
        out = out[..., None]
    return out


def resize_rgb_batch_torch(
    raw_rgb_batch: torch.Tensor,
    image_size: int,
    grayscale: bool = False,
) -> torch.Tensor:
    """Resize NHWC uint8 raw RGB frames into NHWC uint8 model frames."""

    if not isinstance(raw_rgb_batch, torch.Tensor):
        raise TypeError("raw_rgb_batch must be a torch.Tensor")
    if raw_rgb_batch.ndim != 4 or int(raw_rgb_batch.shape[-1]) != 3:
        raise ValueError("raw_rgb_batch must have shape (N, H, W, 3)")
    if raw_rgb_batch.dtype != torch.uint8:
        raise ValueError("raw_rgb_batch must have dtype torch.uint8")

    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    nchw = _resize_rgb_batch_torch_nchw(
        raw_rgb_batch,
        image_size=image_size,
        grayscale=grayscale,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


def _resize_rgb_batch_torch_nchw(
    raw_rgb_batch: torch.Tensor,
    image_size: int,
    grayscale: bool = False,
) -> torch.Tensor:
    nchw = raw_rgb_batch.permute(0, 3, 1, 2).to(torch.float32)
    if grayscale:
        r = nchw[:, 0:1]
        g = nchw[:, 1:2]
        b = nchw[:, 2:3]
        nchw = torch.floor((r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0)

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
        .contiguous()
    )


def reset_frame_stack(frame_obs: np.ndarray, stack_frames: int) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
    return np.concatenate([frame.copy() for _ in range(max(1, int(stack_frames)))], axis=2)


def step_frame_stack(
    stacked_obs: np.ndarray,
    frame_obs: np.ndarray,
    base_channels: int | None = None,
    reset: bool = False,
) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
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
    observation_config: FlappyPpoObservationConfig,
    frame_skip: int | float = 0,
) -> np.ndarray:
    del observation_config, frame_skip
    return np.asarray(stacked_obs, dtype=np.uint8)


def hwc_to_nchw_tensor(
    obs: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    arr = np.asarray(obs, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError("obs must be HWC")
    tensor = torch.as_tensor(arr, dtype=torch.uint8, device=device)
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


class FlappyPpoTorchObservationPipeline:
    """Batch raw-RGB preprocessing pipeline shared by training and evaluation."""

    def __init__(
        self,
        observation_config: FlappyPpoObservationConfig,
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
                self.stacked_channels,
                observation_config.image_size,
                observation_config.image_size,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self._next_stack = torch.empty_like(self._stack)
        self._last_observation: torch.Tensor | None = None

    @property
    def current_frame_stack(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return self._stack.permute(0, 2, 3, 1).detach().cpu().numpy().copy()

    @property
    def current_observation(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return self._last_observation.detach().cpu().numpy().copy()

    def reset(
        self,
        raw_rgb_batch: torch.Tensor,
        frame_skips: np.ndarray | torch.Tensor | None = None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch, profiler=profiler)
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            for stack_idx in range(self.observation_config.stack_frames):
                c0 = stack_idx * self.base_channels
                c1 = c0 + self.base_channels
                self._stack[:, c0:c1, :, :] = frame_obs
        return self._final_obs(frame_skips, profiler=profiler)

    def step(
        self,
        raw_rgb_batch: torch.Tensor,
        dones: np.ndarray,
        frame_skips: np.ndarray | torch.Tensor | None = None,
        profiler: Any | None = None,
        active_mask: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch, profiler=profiler)
        active = self._active_mask(active_mask)
        active_indices = np.flatnonzero(active)
        inactive_indices = np.flatnonzero(~active)
        done_indices = np.flatnonzero(np.asarray(dones, dtype=np.bool_) & active)
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            if self.observation_config.stack_frames <= 1:
                if inactive_indices.size:
                    self._next_stack[inactive_indices] = self._stack[inactive_indices]
                self._next_stack[active_indices] = frame_obs[active_indices]
            else:
                if inactive_indices.size:
                    self._next_stack[inactive_indices] = self._stack[inactive_indices]
                self._next_stack[active_indices, : -self.base_channels, :, :] = self._stack[
                    active_indices, self.base_channels :, :, :
                ]
                self._next_stack[active_indices, -self.base_channels :, :, :] = frame_obs[active_indices]
            for idx in done_indices:
                for stack_idx in range(self.observation_config.stack_frames):
                    c0 = stack_idx * self.base_channels
                    c1 = c0 + self.base_channels
                    self._next_stack[idx, c0:c1, :, :] = frame_obs[idx]
            self._stack, self._next_stack = self._next_stack, self._stack
        return self._final_obs(frame_skips, profiler=profiler)

    def reset_indices(
        self,
        raw_rgb_batch: torch.Tensor,
        reset_mask: np.ndarray | torch.Tensor,
        frame_skips: np.ndarray | torch.Tensor | None = None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        if self._last_observation is None:
            return self.reset(raw_rgb_batch, frame_skips=frame_skips, profiler=profiler)

        frame_obs = self._resize(raw_rgb_batch, profiler=profiler)
        mask = self._active_mask(reset_mask)
        reset_indices = [int(item) for item in np.flatnonzero(mask)]
        timer = profiler.time("stack_update_s") if profiler is not None else nullcontext()
        with timer:
            for idx in reset_indices:
                for stack_idx in range(self.observation_config.stack_frames):
                    c0 = stack_idx * self.base_channels
                    c1 = c0 + self.base_channels
                    self._stack[idx, c0:c1, :, :] = frame_obs[idx]
        return self._final_obs(frame_skips, profiler=profiler)

    def _active_mask(self, value: np.ndarray | torch.Tensor | None) -> np.ndarray:
        if value is None:
            return np.ones(self.n_envs, dtype=np.bool_)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        mask = np.asarray(value, dtype=np.bool_).reshape(-1)
        if int(mask.shape[0]) != self.n_envs:
            raise ValueError("mask must have one value per environment")
        return mask

    def _resize(self, raw_rgb_batch: torch.Tensor, profiler: Any | None = None) -> torch.Tensor:
        if not isinstance(raw_rgb_batch, torch.Tensor):
            raise TypeError("raw_rgb_batch must be a torch.Tensor")
        if raw_rgb_batch.ndim != 4 or int(raw_rgb_batch.shape[-1]) != 3:
            raise ValueError("raw_rgb_batch must have shape (N, H, W, 3)")
        if raw_rgb_batch.dtype != torch.uint8:
            raise ValueError("raw_rgb_batch must have dtype torch.uint8")
        raw_rgb_batch = raw_rgb_batch.to(device=self.device, non_blocking=True)
        if int(raw_rgb_batch.shape[0]) != self.n_envs:
            raise ValueError("raw_rgb_batch must have one frame per environment")
        timer = profiler.time("torch_resize_s") if profiler is not None else nullcontext()
        with timer:
            return _resize_rgb_batch_torch_nchw(
                raw_rgb_batch,
                image_size=self.observation_config.image_size,
                grayscale=self.observation_config.grayscale,
            )

    def _final_obs(
        self,
        frame_skips: np.ndarray | torch.Tensor | None,
        profiler: Any | None = None,
    ) -> torch.Tensor:
        del frame_skips
        timer = profiler.time("final_obs_s") if profiler is not None else nullcontext()
        with timer:
            obs = self._stack
            self._last_observation = obs
            return obs


class FlappyPpoEvalObservationPipeline:
    """Single-environment HWC frame stacker for latency_bench policy evaluation."""

    def __init__(self, observation_config: FlappyPpoObservationConfig):
        self.observation_config = observation_config
        self._stack: np.ndarray | None = None
        self._last_observation: np.ndarray | None = None

    @property
    def current_frame_stack(self) -> np.ndarray | None:
        return None if self._stack is None else self._stack.copy()

    @property
    def current_observation(self) -> np.ndarray | None:
        return None if self._last_observation is None else self._last_observation.copy()

    def reset(self, frame_or_observation: Any, frame_skip: int | float = 0) -> np.ndarray:
        frame = self._frame_obs(frame_or_observation)
        self._stack = reset_frame_stack(frame, self.observation_config.stack_frames)
        self._last_observation = build_stacked_observation(
            self._stack,
            self.observation_config,
            frame_skip=frame_skip,
        )
        return self._last_observation.copy()

    def step(self, frame_or_observation: Any, frame_skip: int | float = 0) -> np.ndarray:
        frame = self._frame_obs(frame_or_observation)
        if self._stack is None:
            return self.reset(frame, frame_skip=frame_skip)
        self._stack = step_frame_stack(
            self._stack,
            frame,
            base_channels=self.observation_config.base_channels,
        )
        self._last_observation = build_stacked_observation(
            self._stack,
            self.observation_config,
            frame_skip=frame_skip,
        )
        return self._last_observation.copy()

    def build_observation(
        self,
        frame_or_observation: Any,
        env_step: int | None = None,
        frame_skip: int | float | None = None,
    ) -> np.ndarray:
        frame, inferred_env_step, inferred_frame_skip = self._extract_inputs(
            frame_or_observation,
            env_step=env_step,
            frame_skip=frame_skip,
        )
        if inferred_env_step == 0 or self._stack is None:
            return self.reset(frame, frame_skip=inferred_frame_skip)
        return self.step(frame, frame_skip=inferred_frame_skip)

    def build_tensor(
        self,
        frame_or_observation: Any,
        env_step: int | None = None,
        frame_skip: int | float | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        obs = self.build_observation(
            frame_or_observation,
            env_step=env_step,
            frame_skip=frame_skip,
        )
        return hwc_to_nchw_tensor(obs, device=device)

    def _frame_obs(self, frame_or_observation: Any) -> np.ndarray:
        frame = _extract_frame(frame_or_observation)
        return resize_frame_pil(
            frame,
            image_size=self.observation_config.image_size,
            grayscale=self.observation_config.grayscale,
        )

    def _extract_inputs(
        self,
        frame_or_observation: Any,
        env_step: int | None,
        frame_skip: int | float | None,
    ) -> tuple[Any, int, int | float]:
        frame = _extract_frame(frame_or_observation)
        inferred_env_step = env_step
        inferred_frame_skip = frame_skip
        metadata = getattr(frame_or_observation, "metadata", None)
        if inferred_env_step is None and hasattr(frame_or_observation, "env_step"):
            inferred_env_step = int(getattr(frame_or_observation, "env_step"))
        if inferred_env_step is None:
            inferred_env_step = 0 if self._stack is None else 1
        if inferred_frame_skip is None and isinstance(metadata, Mapping):
            inferred_frame_skip = metadata.get("frame_skip", metadata.get("latency_frame_skip", 0))
        if inferred_frame_skip is None:
            inferred_frame_skip = 0
        return frame, int(inferred_env_step), inferred_frame_skip


def _extract_frame(frame_or_observation: Any) -> Any:
    if hasattr(frame_or_observation, "data"):
        return getattr(frame_or_observation, "data")
    return frame_or_observation
