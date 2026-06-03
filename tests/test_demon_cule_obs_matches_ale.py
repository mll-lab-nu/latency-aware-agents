"""CULE GPU obs must match ALE obs after the native-RGB + cv2 alignment fix.
RAM-syncs CULE core and ALE (emulator is bit-faithful), steps identical actions, and
compares CULE's obs to ALE's. Requires CUDA + cule + ale_py."""
import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ale_py")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from ale_py import ALEInterface
from training.demon_attack_sf.gpu_core import CudaDemonAttackCore, demon_attack_rom_path
from training.common.atari_cv2_obs import native_rgb_to_obs84


def test_cule_obs_matches_ale():
    SEED, N = 12345, 150
    rng = np.random.default_rng(SEED)
    acts = [(i % 6) if i < 18 else int(rng.integers(0, 6)) for i in range(N)]
    core = CudaDemonAttackCore(batch_size=1, device="cuda", noop_max=1, terminal_on_life_loss=False, obs_stride=4)
    core.reset_batch(seeds=[SEED])
    ale = ALEInterface(); ale.setInt("random_seed", SEED); ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("color_averaging", False); ale.loadROM(demon_attack_rom_path())
    aset = [int(a) for a in ale.getMinimalActionSet()]
    ale.reset_game()
    for idx in (1, 2):
        ale.act(aset[idx])
    for a in range(128):
        ale.setRAM(a, int(core.ram[0, a].item()))
    d_op = 0
    for t in range(N):
        ale.act(aset[acts[t]]); rgb = ale.getScreenRGB()
        cule = core.step_batch([acts[t]]).observations[0, 0].cpu().numpy().astype(int)
        ale_op = native_rgb_to_obs84(torch.from_numpy(rgb)).numpy().astype(int)
        d_op = max(d_op, int(np.abs(cule - ale_op).max()))
    # Same preprocessing op + bit-faithful emulator -> identical observations.
    assert d_op == 0, f"CULE obs vs ALE(native+op) max diff {d_op} (expected 0)"
