from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer


_PROFILE_FIELDS = (
    "env_step_s",
    "get_render_state_s",
    "state_batch_from_states_s",
    "render_into_s",
    "pipeline_s",
    "reset_obs_total_s",
    "step_obs_total_s",
)


@dataclass(frozen=True)
class FlappyObservationConfig:
    image_size: int
    grayscale: bool
    stack_frames: int

    @property
    def base_channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def in_channels(self) -> int:
        return self.base_channels * self.stack_frames


class GpuRenderProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        interval: int,
        device: torch.device,
        worker_index: int,
        vector_index: int,
        env_id: int,
    ):
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.device = torch.device(device)
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.total_steps = 0
        self.totals = {field: 0.0 for field in _PROFILE_FIELDS}
        self.counts = {field: 0 for field in _PROFILE_FIELDS}

    def time(self, field: str):
        if not self.enabled:
            return nullcontext()
        return _GpuRenderProfileTimer(self, field)

    def record(self, field: str, elapsed: float) -> None:
        self.totals[field] = self.totals.get(field, 0.0) + float(elapsed)
        self.counts[field] = self.counts.get(field, 0) + 1

    def synchronize(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def observe_step(self) -> None:
        if not self.enabled:
            return
        self.total_steps += 1
        if self.total_steps % self.interval == 0:
            self.log_and_reset()

    def log_and_reset(self) -> None:
        parts = [
            "[FlappyGpuRenderProfile]",
            f"pid={os.getpid()}",
            f"worker={self.worker_index}",
            f"vector={self.vector_index}",
            f"env_id={self.env_id}",
            f"steps={self.total_steps}",
        ]
        for field in _PROFILE_FIELDS:
            count = self.counts.get(field, 0)
            avg_ms = 0.0 if count <= 0 else 1000.0 * self.totals[field] / float(count)
            parts.append(f"{field}_avg_ms={avg_ms:.3f}")
            parts.append(f"{field}_n={count}")
        print(" ".join(parts), flush=True)
        for field in _PROFILE_FIELDS:
            self.totals[field] = 0.0
            self.counts[field] = 0


class _GpuRenderProfileTimer:
    def __init__(self, profiler: GpuRenderProfiler, field: str):
        self.profiler = profiler
        self.field = field
        self.start = 0.0

    def __enter__(self):
        self.profiler.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.synchronize()
        self.profiler.record(self.field, time.perf_counter() - self.start)
        return False


def resolve_gpu_render_device(device: str | torch.device) -> torch.device:
    requested = str(device).strip().lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Flappy GPU rendering requested gpu_render_device=auto, but CUDA is unavailable. "
                "Use gpu_render_device=cpu for CPU-compatible tests."
            )
        return torch.device("cuda")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Flappy GPU rendering requested gpu_render_device={device!s}, but CUDA is unavailable. "
                "Use gpu_render_device=cpu for CPU-compatible tests."
            )
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("gpu_render_device must be 'auto', 'cpu', 'cuda', or a cuda device string")


def resize_rgb_batch_torch_nchw(
    raw_rgb_batch: torch.Tensor,
    *,
    image_size: int,
    grayscale: bool,
) -> torch.Tensor:
    nchw = raw_rgb_batch.permute(0, 3, 1, 2).to(torch.float32)
    if grayscale:
        r = nchw[:, 0:1]
        g = nchw[:, 1:2]
        b = nchw[:, 2:3]
        nchw = torch.floor((r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0)
    resized = F.interpolate(
        nchw,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.round().clamp_(0, 255).to(torch.uint8).contiguous()


class FlappyTorchObservationPipeline:
    def __init__(
        self,
        observation_config: FlappyObservationConfig,
        n_envs: int,
        device: str | torch.device,
    ) -> None:
        self.observation_config = observation_config
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self.base_channels = observation_config.base_channels
        self._stack = torch.empty(
            (
                self.n_envs,
                observation_config.in_channels,
                observation_config.image_size,
                observation_config.image_size,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self._next_stack = torch.empty_like(self._stack)
        self._last_observation: torch.Tensor | None = None

    def reset(self, raw_rgb_batch: torch.Tensor) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch)
        for stack_idx in range(self.observation_config.stack_frames):
            c0 = stack_idx * self.base_channels
            c1 = c0 + self.base_channels
            self._stack[:, c0:c1, :, :] = frame_obs
        return self._final_obs()

    def step(
        self,
        raw_rgb_batch: torch.Tensor,
        *,
        dones: np.ndarray,
        active_mask: np.ndarray,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch)
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        active_indices = np.flatnonzero(active)
        inactive_indices = np.flatnonzero(~active)
        done_indices = np.flatnonzero(np.asarray(dones, dtype=np.bool_).reshape(-1) & active)
        if self.observation_config.stack_frames <= 1:
            self._next_stack[inactive_indices] = self._stack[inactive_indices]
            self._next_stack[active_indices] = frame_obs[active_indices]
        else:
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
        return self._final_obs()

    def reset_indices(self, raw_rgb_batch: torch.Tensor, reset_mask: np.ndarray) -> torch.Tensor:
        if self._last_observation is None:
            return self.reset(raw_rgb_batch)
        frame_obs = self._resize(raw_rgb_batch)
        for idx in np.flatnonzero(np.asarray(reset_mask, dtype=np.bool_).reshape(-1)):
            for stack_idx in range(self.observation_config.stack_frames):
                c0 = stack_idx * self.base_channels
                c1 = c0 + self.base_channels
                self._stack[idx, c0:c1, :, :] = frame_obs[idx]
        return self._final_obs()

    def _resize(self, raw_rgb_batch: torch.Tensor) -> torch.Tensor:
        raw_rgb_batch = raw_rgb_batch.to(device=self.device, non_blocking=True)
        return resize_rgb_batch_torch_nchw(
            raw_rgb_batch,
            image_size=self.observation_config.image_size,
            grayscale=self.observation_config.grayscale,
        )

    def _final_obs(self) -> torch.Tensor:
        self._last_observation = self._stack
        return self._stack


class FlappyTorchRenderPipeline:
    def __init__(self, *, device: torch.device):
        self.device = torch.device(device)
        self.renderer: TorchFlappyRenderer | None = None
        self._raw_rgb_buffer: torch.Tensor | None = None

    def render(self, batch: BatchedFlappyRenderState) -> torch.Tensor:
        if self.renderer is None:
            assets = FlappyRenderAssets.load(
                bird_color=batch.bird_color,
                pipe_color=batch.pipe_color,
                background=batch.background,
            )
            self.renderer = TorchFlappyRenderer(assets, device=self.device)
        return self.renderer.render_into(batch, self._raw_rgb_output_buffer(batch))

    def _raw_rgb_output_buffer(self, batch: BatchedFlappyRenderState) -> torch.Tensor:
        expected_shape = (int(batch.batch_size), int(batch.screen_height), int(batch.screen_width), 3)
        expected_device = self.device if self.renderer is None else self.renderer.device
        if (
            self._raw_rgb_buffer is None
            or tuple(self._raw_rgb_buffer.shape) != expected_shape
            or self._raw_rgb_buffer.device != expected_device
        ):
            self._raw_rgb_buffer = torch.empty(expected_shape, dtype=torch.uint8, device=expected_device)
        return self._raw_rgb_buffer
