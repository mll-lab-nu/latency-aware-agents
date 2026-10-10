"""CULE GPU obs must match ALE obs after the native-RGB + cv2 alignment fix.
Resets CULE and ALE with the same seed and reset actions, then compares their
observations with OpenCV preprocessing. Requires CUDA + cule + ale_py."""
import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ale_py")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from ale_py import ALEInterface
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.common.gpu_latency import CudaLatencyScheduler
from training.demon_attack_sf.gpu_core import CudaDemonAttackCore, demon_attack_rom_path


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
    ale.act(aset[0])
    for idx in (1, 2):
        ale.act(aset[idx])
    d_op = 0
    for t in range(N):
        ale.act(aset[acts[t]]); rgb = ale.getScreenRGB()
        cule = core.step_batch([acts[t]]).observations[0, 0].cpu().numpy().astype(int)
        ale_op = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA).astype(int)
        d_op = max(d_op, int(np.abs(cule - ale_op).max()))
    # Exact OpenCV preprocessing + matching initialization -> identical observations.
    assert d_op == 0, f"CULE obs vs ALE(cv2) max diff {d_op} (expected 0)"


def test_cule_native_rgb_frame_matches_ale_screen():
    SEED = 12345
    core = CudaDemonAttackCore(batch_size=1, device="cuda", noop_max=1, terminal_on_life_loss=False, obs_stride=4)
    core.reset_batch(seeds=[SEED])
    ale = ALEInterface(); ale.setInt("random_seed", SEED); ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("color_averaging", False); ale.loadROM(demon_attack_rom_path())
    aset = [int(a) for a in ale.getMinimalActionSet()]
    ale.reset_game()
    ale.act(aset[0])
    for idx in (1, 2):
        ale.act(aset[idx])

    ale.act(aset[0])
    core.step_batch([0])

    got = core.native_rgb_frames()[0].cpu().numpy()
    expected = ale.getScreenRGB()
    assert got.shape == expected.shape == (210, 160, 3)
    assert got.dtype == expected.dtype == np.uint8
    assert np.array_equal(got, expected)



def test_cule_preprocessing_matches_opencv_for_palette_frames():
    """Long-lived GPU preprocessing contract for native Atari palette frames."""
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(5, 300, 160), dtype=np.uint8)
    frames[0] = 0
    frames[1] = 14
    frames[2, :, ::2] = 0
    frames[2, :, 1::2] = 14
    core = CudaDemonAttackCore(batch_size=len(frames), device="cuda", noop_max=1)
    core.reset_batch(seeds=list(range(len(frames))))
    core.frame_buffer.copy_(torch.from_numpy(frames.reshape(len(frames), -1)))
    rgb = core.native_rgb_frames().cpu().numpy()
    core.env.generate_obs84(core._obs84_single.data_ptr())
    expected = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        for frame in rgb
    ])
    np.testing.assert_array_equal(core.observations()[:, 0].cpu().numpy(), expected)

def _latency_scheduler(batch_size: int, *, obs_fps: float = 15.0) -> CudaLatencyScheduler:
    return CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=obs_fps,
        hold_policy=ActionHoldPolicy.HOLD,
        ordering_policy=OrderingPolicy.LATEST_READY,
        seeds=list(range(1000, 1000 + batch_size)),
        env_reset_seed_bases=list(range(2000, 2000 + batch_size)),
        add_latency_info=False,
        device="cuda",
    )


@pytest.mark.parametrize(
    ("frame_stack", "expected_frames"),
    [
        (2, 2),
        (4, 4),
        (6, 4),
    ],
)
def test_decision_obs_only_contains_new_frames_that_enter_policy_stack(frame_stack, expected_frames):
    seed = 12345
    core = CudaDemonAttackCore(
        batch_size=1,
        device="cuda",
        noop_max=1,
        terminal_on_life_loss=False,
        obs_stride=4,
        frame_stack=frame_stack,
    )
    latency = _latency_scheduler(batch_size=1)
    latency.reset(seeds=[seed])
    core.reset_batch(seeds=[seed])
    result = core.step_decision(
        latency,
        [2],
        active_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        episode_step_idx=torch.zeros(1, dtype=torch.int64, device="cuda"),
        max_episode_steps=25000,
    )
    assert result.observations.shape == (1, expected_frames, 84, 84)


@pytest.mark.parametrize("frame_stack", [2, 4, 6])
def test_decision_obs_matches_step_batch_reference_for_rendered_frames(frame_stack):
    seed = 12345
    obs_stride = 4
    decision_core = CudaDemonAttackCore(
        batch_size=1,
        device="cuda",
        noop_max=1,
        terminal_on_life_loss=False,
        obs_stride=obs_stride,
        frame_stack=frame_stack,
    )
    reference_core = CudaDemonAttackCore(
        batch_size=1,
        device="cuda",
        noop_max=1,
        terminal_on_life_loss=False,
        obs_stride=obs_stride,
        frame_stack=frame_stack,
    )
    latency = _latency_scheduler(batch_size=1)
    latency.reset(seeds=[seed])
    decision_core.reset_batch(seeds=[seed])
    reference_core.reset_batch(seeds=[seed])

    action = 2
    result = decision_core.step_decision(
        latency,
        [action],
        active_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        episode_step_idx=torch.zeros(1, dtype=torch.int64, device="cuda"),
        max_episode_steps=25000,
    )
    reference_frames = torch.cat(
        [reference_core.step_batch([action]).observations.clone() for _ in range(obs_stride)],
        dim=1,
    )
    expected_frames = reference_frames[:, -min(frame_stack, obs_stride):, :, :]
    assert torch.equal(result.observations.cpu(), expected_frames.cpu())
