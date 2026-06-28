"""Parity: training/common/atari_cv2_obs must match cv2 (the ALE/SB3 eval pipeline)."""
import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")
from training.common.atari_cv2_obs import native_rgb_to_obs84, rgb_to_gray_cv2, resize_area_84


def _cv2_ref(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_native_rgb_to_obs84_matches_cv2(seed):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    ref = _cv2_ref(rgb)
    got = native_rgb_to_obs84(torch.from_numpy(rgb)).cpu().numpy()
    diff = np.abs(ref.astype(int) - got.astype(int))
    # grayscale is bit-exact; INTER_AREA matches cv2 to <=1 LSB on a few pixels (float
    # accumulation-order noise, unstructured). This is ~50x smaller than the structured
    # NTSC-gray+bilinear mismatch that broke transfer, and negligible for the policy.
    assert diff.max() <= 1, f"max|diff|={diff.max()} mean={diff.mean():.4f} nonzero={int((diff>0).sum())}"
    assert (diff > 0).mean() < 0.01, f"too many differing pixels: {(diff>0).mean():.4f}"


def test_grayscale_matches_cv2():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    ref = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    got = rgb_to_gray_cv2(torch.from_numpy(rgb)).cpu().numpy()
    assert np.array_equal(ref, got), f"gray max|diff|={np.abs(ref.astype(int)-got.astype(int)).max()}"


def test_batched_shape():
    rgb = torch.zeros((5, 210, 160, 3), dtype=torch.uint8)
    out = native_rgb_to_obs84(rgb)
    assert out.shape == (5, 84, 84) and out.dtype == torch.uint8
