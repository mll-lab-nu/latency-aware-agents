from __future__ import annotations

from importlib import resources
from typing import Sequence

import torch

from training.demon_attack_sf.gpu_core import CudaAtariCore


# ALE reference RAM bytes after NoopResetEnv(noops=1..30).
_RESET_RAM_ADDRESSES = (
    0x0F,
    0x16,
    0x17,
    0x18,
    0x1E,
    0x20,
    0x27,
    0x30,
    0x31,
    0x38,
    0x39,
    0x45,
    0x46,
    0x4A,
    0x4E,
    0x52,
    0x70,
    0x7A,
)
_RESET_RAM_VALUES_BY_CACHE_INDEX = (
    (1, 112, 96, 112, 152, 162, 0, 0, 96, 0, 96, 238, 82, 7, 1, 64, 10, 72),
    (1, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 221, 37, 7, 1, 64, 10, 73),
    (0, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 187, 75, 7, 1, 64, 10, 74),
    (1, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 118, 23, 7, 1, 64, 10, 75),
    (0, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 236, 46, 7, 1, 64, 10, 76),
    (0, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 216, 93, 7, 1, 64, 10, 77),
    (1, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 177, 59, 7, 1, 64, 10, 78),
    (0, 112, 96, 112, 152, 166, 0, 0, 96, 0, 96, 99, 119, 7, 1, 64, 10, 79),
    (1, 112, 96, 112, 152, 166, 0, 0, 96, 0, 96, 198, 110, 7, 1, 64, 10, 80),
    (1, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 141, 93, 7, 1, 64, 10, 81),
    (1, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 26, 59, 7, 1, 64, 10, 82),
    (0, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 52, 118, 7, 1, 64, 10, 83),
    (1, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 105, 108, 7, 1, 64, 10, 84),
    (1, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 210, 88, 7, 1, 64, 10, 85),
    (1, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 165, 49, 7, 1, 64, 10, 86),
    (0, 112, 96, 112, 152, 166, 0, 0, 96, 0, 96, 75, 99, 7, 1, 64, 10, 87),
    (1, 112, 96, 112, 152, 166, 0, 0, 96, 0, 96, 150, 70, 7, 1, 64, 10, 88),
    (1, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 45, 13, 7, 1, 64, 10, 89),
    (0, 120, 104, 120, 152, 166, 0, 0, 96, 0, 96, 91, 26, 7, 1, 64, 10, 90),
    (0, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 183, 52, 7, 1, 64, 10, 91),
    (0, 120, 96, 120, 152, 166, 0, 0, 96, 0, 96, 111, 105, 7, 1, 64, 10, 92),
    (1, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 222, 82, 7, 1, 64, 10, 93),
    (1, 112, 104, 112, 152, 166, 0, 0, 96, 0, 96, 189, 37, 7, 1, 64, 10, 94),
    (0, 112, 96, 112, 152, 166, 0, 0, 96, 0, 96, 123, 75, 7, 1, 64, 10, 95),
    (1, 112, 96, 112, 255, 166, 152, 0, 96, 0, 96, 246, 22, 3, 255, 64, 9, 96),
    (0, 120, 104, 120, 152, 166, 151, 9, 208, 10, 64, 236, 45, 3, 255, 64, 9, 97),
    (0, 120, 104, 120, 152, 166, 150, 9, 224, 10, 80, 216, 91, 3, 255, 64, 9, 98),
    (1, 120, 96, 120, 152, 166, 149, 9, 240, 10, 96, 177, 55, 3, 255, 64, 9, 99),
    (0, 120, 96, 120, 152, 166, 148, 9, 0, 10, 112, 99, 111, 3, 255, 64, 9, 100),
    (1, 112, 104, 112, 152, 166, 147, 9, 16, 9, 144, 198, 94, 3, 255, 64, 9, 101),
)


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
            reset_actions=(0, 0),
            reset_action_frame_count=0,
            life_loss_reset_actions=(),
            device=device,
            noop_max=noop_max,
            terminal_on_life_loss=terminal_on_life_loss,
            obs_stride=obs_stride,
            frame_stack=frame_stack,
        )
        self._reset_ram_addresses = torch.tensor(_RESET_RAM_ADDRESSES, dtype=torch.int64, device=self.device)
        self._reset_ram_values = torch.tensor(
            _RESET_RAM_VALUES_BY_CACHE_INDEX,
            dtype=torch.uint8,
            device=self.device,
        )

    def _patch_reset_ram(self, target_indices: Sequence[int]) -> None:
        index_t = torch.as_tensor(target_indices, dtype=torch.int64, device=self.device)
        values = self._reset_ram_values[self.cache_index[index_t].to(torch.int64)]
        self.ram[index_t[:, None], self._reset_ram_addresses[None, :]] = values
