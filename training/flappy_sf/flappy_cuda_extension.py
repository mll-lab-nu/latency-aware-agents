from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.cpp_extension import load

from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets
from training.common.cuda_extension_build import ensure_compatible_cuda_extension_toolchain


@lru_cache(maxsize=1)
def load_flappy_cuda_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("simulator=gpu requires CUDA, but torch.cuda.is_available() is false")
    ensure_compatible_cuda_extension_toolchain()
    root = Path(__file__).resolve().parent / "cuda_kernels"
    return load(
        name="flappy_cuda_kernels",
        sources=[
            str(root / "flappy_cuda.cpp"),
            str(root / "flappy_cuda_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class CudaFlappyRenderer:
    min_rotation = -90
    max_rotation = 20

    def __init__(self, assets: FlappyRenderAssets, *, device: torch.device) -> None:
        self.assets = assets
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CudaFlappyRenderer requires a CUDA device")
        self.ext = load_flappy_cuda_extension()
        self.has_background = assets.background_rgb is not None
        self.background_rgb = self._uint8_tensor(
            np.zeros((1, 1, 3), dtype=np.uint8)
            if assets.background_rgb is None
            else assets.background_rgb
        )
        self.fill_color = self._uint8_tensor(np.asarray(assets.fill_color, dtype=np.uint8))
        assert assets.base is not None
        assert assets.pipe_upper is not None
        assert assets.pipe_lower is not None
        assert assets.birds is not None
        self.base_rgb = self._uint8_tensor(assets.base.rgb)
        self.base_mask = self._bool_tensor(assets.base.mask)
        self.pipe_upper_rgb = self._uint8_tensor(assets.pipe_upper.rgb)
        self.pipe_upper_mask = self._bool_tensor(assets.pipe_upper.mask)
        self.pipe_lower_rgb = self._uint8_tensor(assets.pipe_lower.rgb)
        self.pipe_lower_mask = self._bool_tensor(assets.pipe_lower.mask)
        self.bird_rgb, self.bird_mask, self.bird_heights, self.bird_widths = self._bird_atlas(assets)

    def render_core_into(self, core: Any, out: torch.Tensor) -> torch.Tensor:
        self.ext.flappy_render(
            out,
            self.background_rgb,
            self.fill_color,
            self.has_background,
            self.base_rgb,
            self.base_mask,
            self.pipe_upper_rgb,
            self.pipe_upper_mask,
            self.pipe_lower_rgb,
            self.pipe_lower_mask,
            self.bird_rgb,
            self.bird_mask,
            self.bird_heights,
            self.bird_widths,
            core.player_x,
            core.player_y,
            core.player_idx,
            core.player_rot,
            core.ground_x,
            core.ground_y,
            core.upper_pipe_xy,
            core.lower_pipe_xy,
            int(core.batch_size),
            int(core.screen_width),
            int(core.screen_height),
            self.max_rotation - self.min_rotation + 1,
            self.min_rotation,
        )
        return out

    def _bird_atlas(self, assets: FlappyRenderAssets) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert assets.birds is not None
        rotations = list(range(self.min_rotation, self.max_rotation + 1))
        player_indices = sorted({int(key[0]) for key in assets.birds})
        max_h = max(sprite.height for sprite in assets.birds.values())
        max_w = max(sprite.width for sprite in assets.birds.values())
        count = (max(player_indices) + 1) * len(rotations)
        rgb = np.zeros((count, max_h, max_w, 3), dtype=np.uint8)
        mask = np.zeros((count, max_h, max_w), dtype=np.bool_)
        heights = np.zeros((count,), dtype=np.int64)
        widths = np.zeros((count,), dtype=np.int64)
        for player_idx in player_indices:
            for rotation in rotations:
                sprite = assets.birds[(player_idx, rotation)]
                atlas_idx = int(player_idx) * len(rotations) + (int(rotation) - self.min_rotation)
                rgb[atlas_idx, : sprite.height, : sprite.width] = sprite.rgb
                mask[atlas_idx, : sprite.height, : sprite.width] = sprite.mask
                heights[atlas_idx] = sprite.height
                widths[atlas_idx] = sprite.width
        return (
            self._uint8_tensor(rgb),
            self._bool_tensor(mask),
            self._int64_tensor(heights),
            self._int64_tensor(widths),
        )

    def _uint8_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.uint8, device=self.device).contiguous()

    def _bool_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.bool, device=self.device).contiguous()

    def _int64_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.int64, device=self.device).contiguous()
