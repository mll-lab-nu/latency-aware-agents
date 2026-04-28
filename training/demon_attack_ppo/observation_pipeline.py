from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from modeling.demon_attack_ppo import (
    DemonAttackPpoTorchObservationPipeline,
    observation_config_from_args,
)
from training.demon_attack_ppo.profiler import UpdateProfiler


class ObservationPipeline:
    """Training wrapper around the shared Atari-frame Torch preprocessor."""

    def __init__(
        self,
        args: argparse.Namespace,
        n_envs: int | None = None,
        device: torch.device | str = "cpu",
        envs: Any | None = None,
        renderer: Any | None = None,
    ) -> None:
        del renderer
        if n_envs is None:
            if envs is None:
                raise ValueError("n_envs is required when envs is not provided")
            n_envs = int(envs.n_envs)
        self.args = args
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self.observation_config = observation_config_from_args(args)
        self._pipeline = DemonAttackPpoTorchObservationPipeline(
            observation_config=self.observation_config,
            n_envs=self.n_envs,
            device=self.device,
        )

    @property
    def in_channels(self) -> int:
        return self.observation_config.in_channels

    @property
    def current_observation(self) -> np.ndarray | None:
        return self._pipeline.current_observation

    def reset(
        self,
        raw_batch: torch.Tensor,
        frame_skips: np.ndarray | torch.Tensor,
        profiler: UpdateProfiler | None = None,
    ) -> torch.Tensor:
        return self._pipeline.reset(
            raw_batch,
            frame_skips=frame_skips,
            profiler=profiler,
        )

    def step(
        self,
        raw_batch: torch.Tensor,
        dones: np.ndarray,
        frame_skips: np.ndarray | torch.Tensor,
        profiler: UpdateProfiler | None = None,
    ) -> torch.Tensor:
        return self._pipeline.step(
            raw_batch,
            dones=dones,
            frame_skips=frame_skips,
            profiler=profiler,
        )
