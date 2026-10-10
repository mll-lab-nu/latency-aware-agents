from __future__ import annotations

from importlib import resources
import torch

from training.demon_attack_sf.gpu_core import CudaAtariCore


def atlantis_rom_path() -> str:
    return str(resources.files("AutoROM.roms").joinpath("atlantis.bin"))


class CudaAtlantisCore(CudaAtariCore):
    """Configure the shared CuLE CUDA core for Atlantis full-game semantics."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cuda",
        noop_max: int = 30,
        terminal_on_life_loss: bool = False,
        obs_stride: int = 1,
        frame_stack: int = 1,
    ) -> None:
        super().__init__(
            batch_size,
            rom_path=atlantis_rom_path(),
            reset_actions=(),
            life_loss_reset_actions=(),
            device=device,
            noop_max=noop_max,
            terminal_on_life_loss=terminal_on_life_loss,
            obs_stride=obs_stride,
            frame_stack=frame_stack,
        )
