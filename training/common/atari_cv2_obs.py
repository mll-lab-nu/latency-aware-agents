"""Shared Atari observation preprocessing that matches the ALE/SB3 (cv2) pipeline:
RGB -> grayscale (cv2 RGB2GRAY fixed-point weights) -> resize to 84x84 via cv2 INTER_AREA.

Used to make GPU (CULE) training/eval observations bit-identical to the CPU ALE eval
pipeline (latency_bench DemonAttackEnvAdapter via stable_baselines3 AtariWrapper), so that
CULE-trained policies transfer to ALE eval.

INTER_AREA downscaling is separable and, for each output pixel, is a weighted average of the
input pixels it overlaps (weight = overlap length / scale). We precompute the row/col weight
matrices and apply them as matmuls (GPU-friendly, deterministic). cv2 RGB2GRAY uses
Y = (R*4899 + G*9617 + B*1868 + 8192) >> 14.
"""
from __future__ import annotations

from functools import lru_cache

import torch

# cv2 COLOR_RGB2GRAY fixed-point coefficients (yuv_shift=15, sum=32768) — bit-exact vs cv2
_R2Y, _G2Y, _B2Y, _YUV_SHIFT = 9798, 19235, 3735, 15


@lru_cache(maxsize=8)
def _area_weight_matrix(n_in: int, n_out: int, device_str: str) -> torch.Tensor:
    """cv2 INTER_AREA (downscale) separable weight matrix: (n_out, n_in), rows sum to 1."""
    scale = n_in / n_out
    w = torch.zeros((n_out, n_in), dtype=torch.float64)
    for i in range(n_out):
        a = i * scale
        b = (i + 1) * scale
        j0 = int(a)                       # floor(a)
        j1 = min(int(-(-b // 1)), n_in)   # ceil(b), clamped
        for j in range(j0, j1):
            ov = min(b, j + 1.0) - max(a, float(j))
            if ov > 0:
                w[i, j] = ov / scale
    return w.to(device_str)


def rgb_to_gray_cv2(rgb: torch.Tensor) -> torch.Tensor:
    """(...,H,W,3) uint8 RGB -> (...,H,W) uint8 grayscale, matching cv2.cvtColor RGB2GRAY."""
    r = rgb[..., 0].to(torch.int32)
    g = rgb[..., 1].to(torch.int32)
    b = rgb[..., 2].to(torch.int32)
    y = (r * _R2Y + g * _G2Y + b * _B2Y + (1 << (_YUV_SHIFT - 1))) >> _YUV_SHIFT
    return y.to(torch.uint8)


def resize_area_84(gray: torch.Tensor, out_hw: int = 84) -> torch.Tensor:
    """(...,H,W) uint8 -> (...,84,84) uint8 via cv2 INTER_AREA-equivalent area resampling."""
    *lead, h, w = gray.shape
    dev = gray.device.type if gray.device.type != "cuda" else "cuda"
    # float32 to match cv2's INTER_AREA accumulation precision (matches cv2 to <=1 LSB)
    wr = _area_weight_matrix(h, out_hw, dev).to(torch.float32)   # (84, H)
    wc = _area_weight_matrix(w, out_hw, dev).to(torch.float32)   # (84, W)
    g = gray.reshape(-1, h, w).to(torch.float32)
    # out[i,j] = sum_{r,c} wr[i,r] * g[r,c] * wc[j,c]
    out = torch.einsum("oh,nhw,pw->nop", wr, g, wc)   # (-1, 84, 84)
    out = torch.round(out).clamp_(0, 255).to(torch.uint8)
    return out.reshape(*lead, out_hw, out_hw)


def native_rgb_to_obs84(rgb: torch.Tensor, out_hw: int = 84) -> torch.Tensor:
    """(...,210,160,3) uint8 native RGB -> (...,84,84) uint8, matching ALE/SB3 cv2 obs."""
    return resize_area_84(rgb_to_gray_cv2(rgb), out_hw=out_hw)
