from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from modeling.flappy_ppo import (
    latency_channel_value,
    observation_config_from_args,
    resize_frame_pil,
)
from training.flappy_ppo.env_batch import SyncFlappyBatch
from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path
from training.flappy_ppo.profiler import UpdateProfiler


ensure_flappy_gymnasium_on_path()

from flappy_bird_gymnasium.envs.render_state import (  # noqa: E402
    BatchedFlappyRenderState,
)
from flappy_bird_gymnasium.gpu_render import TorchFlappyRenderer  # noqa: E402


@dataclass
class ObservationPipeline:
    args: argparse.Namespace
    envs: SyncFlappyBatch
    device: torch.device
    renderer: Optional[TorchFlappyRenderer] = None

    def __post_init__(self):
        self.observation_config = observation_config_from_args(self.args)
        self.base_channels = self.observation_config.base_channels
        self.stacked_channels = self.observation_config.stacked_channels
        self.in_channels = self.observation_config.in_channels
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
                max_skip = self.observation_config.max_frame_skip
                values = np.asarray(
                    [latency_channel_value(skip, max_skip) for skip in frame_skips],
                    dtype=np.uint8,
                )
                latency = torch.as_tensor(values, dtype=torch.uint8, device=self.device)
                latency = latency.view(-1, 1, 1, 1).expand(
                    -1, self.args.image_size, self.args.image_size, 1
                )
                obs = torch.cat([obs, latency], dim=3)
            return obs.permute(0, 3, 1, 2).contiguous()
