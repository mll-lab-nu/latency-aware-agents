from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


def test_demon_attack_rom_resource_matches_supported_md5():
    from training.demon_attack_sf.gpu_core import load_demon_attack_rom_bytes

    rom = load_demon_attack_rom_bytes()

    assert len(rom) == 4096
    assert hashlib.md5(rom).hexdigest() == "f0e0addc07971561ab80d9abe1b8d333"


def test_demon_attack_cuda_decodes_ram_reward_lives_and_terminal():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.demon_attack_cuda_extension import load_demon_attack_cuda_extension

    ext = load_demon_attack_cuda_extension()
    ram = torch.zeros((3, 128), dtype=torch.uint8, device="cuda")
    previous_score = torch.tensor([0, 1200, 0], dtype=torch.int32, device="cuda")
    reward = torch.empty((3,), dtype=torch.int32, device="cuda")
    score = torch.empty((3,), dtype=torch.int32, device="cuda")
    lives = torch.empty((3,), dtype=torch.int32, device="cuda")
    terminal = torch.empty((3,), dtype=torch.bool, device="cuda")

    ram[0, 0x05] = 0x12
    ram[0, 0x03] = 0x34
    ram[0, 0x01] = 0x56
    ram[0, 0x72] = 3
    ram[0, 0x71] = 0

    ram[1, 0x05] = 0x00
    ram[1, 0x03] = 0x13
    ram[1, 0x01] = 0x00
    ram[1, 0x72] = 0
    ram[1, 0x71] = 0xBD

    ram[2, 0x05] = 0xEA
    ram[2, 0x03] = 0xCD
    ram[2, 0x01] = 0xAB
    ram[2, 0x72] = 2
    ram[2, 0x71] = 0

    ext.demon_attack_decode_ram(ram, previous_score, reward, score, lives, terminal)
    torch.cuda.synchronize()

    assert score.cpu().tolist() == [123456, 1300, 0]
    assert reward.cpu().tolist() == [123456, 100, 0]
    assert lives.cpu().tolist() == [4, 0, 3]
    assert terminal.cpu().tolist() == [False, True, False]


def test_demon_attack_cuda_core_runs_real_atari_step():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore

    core = CudaDemonAttackCore(batch_size=2, device="cuda", noop_max=30, terminal_on_life_loss=True)

    obs, infos = core.reset_batch(seeds=[123, 456])
    assert tuple(obs.shape) == (2, 1, 84, 84)
    assert obs.dtype == torch.uint8
    assert core.minimal_action_values == [0, 16, 1, 2, 17, 18]
    assert [info["score"] for info in infos] == [0, 0]

    result = core.step_batch(torch.zeros(2, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()

    assert tuple(result.observations.shape) == (2, 1, 84, 84)
    assert tuple(result.rewards.shape) == (2,)
    assert tuple(result.terminated.shape) == (2,)
    assert tuple(result.scores.shape) == (2,)
    assert result.observations.dtype == torch.uint8


def test_demon_attack_cuda_core_resets_selected_indices():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore

    core = CudaDemonAttackCore(batch_size=3, device="cuda", noop_max=30, terminal_on_life_loss=True)
    core.reset_batch(seeds=[123, 456, 789])
    for _ in range(60):
        core.step_batch(torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda"))
    before_reset_ram = core.ram.clone()
    core._scores[:] = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda")

    obs, infos = core.reset_batch(seeds=[999], indices=[1])
    torch.cuda.synchronize()

    assert tuple(obs.shape) == (3, 1, 84, 84)
    assert core.score.cpu().tolist() == [11, 0, 33]
    assert [info["score"] for info in infos] == [11, 0, 33]
    assert torch.equal(core.ram[0], before_reset_ram[0])
    assert not torch.equal(core.ram[1], before_reset_ram[1])
    assert torch.equal(core.ram[2], before_reset_ram[2])


def test_demon_attack_batched_gpu_env_applies_fixed_latency_step():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    env = DemonAttackBatchedGpuEnv(
        cfg=SimpleNamespace(
            seed=0,
            env_fps=60,
            obs_fps=15,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            max_episode_steps=25000,
            latency_type="fixed",
            fixed_latency_ms=1000.0 / 60.0,
            add_latency_info=True,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
            episode_metrics_path="",
            latency_seed=123,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    obs, infos = env.reset()
    step_obs, rewards, terminated, truncated, step_infos = env.step(torch.zeros(2, dtype=torch.int64, device="cuda"))

    assert tuple(obs["obs"].shape) == (2, 4, 84, 84)
    assert tuple(step_obs["obs"].shape) == (2, 4, 84, 84)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert [info["latency_raw_frames"] for info in infos] == [1, 1]
    assert [info["latency_raw_frames"] for info in step_infos] == [1, 1]


def test_demon_attack_batched_gpu_env_resets_core_on_time_limit_truncation():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    env = DemonAttackBatchedGpuEnv(
        cfg=SimpleNamespace(
            seed=0,
            env_fps=60,
            obs_fps=15,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            max_episode_steps=1,
            latency_type="fixed",
            fixed_latency_ms=1000.0 / 60.0,
            add_latency_info=False,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
            episode_metrics_path="",
            latency_seed=123,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    env.reset()
    env.core._scores[:] = torch.tensor([5, 6], dtype=torch.int32, device="cuda")
    _, _, _, truncated, _ = env.step(torch.zeros(2, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()

    assert truncated.tolist() == [True, True]
    assert env._raw_episode_steps.tolist() == [0, 0]
    assert env.core.score.cpu().tolist() == [0, 0]


def test_demon_attack_gpu_latency_info_uses_fused_decision_path(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    fused_calls = 0
    original = CudaDemonAttackCore.step_decision

    def wrapped(self, *args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original(self, *args, **kwargs)

    def fail_step_batch(self, *args, **kwargs):
        raise AssertionError("Demon GPU env should not use per-raw-frame step_batch")

    monkeypatch.setattr(CudaDemonAttackCore, "step_decision", wrapped)
    monkeypatch.setattr(CudaDemonAttackCore, "step_batch", fail_step_batch)

    env = DemonAttackBatchedGpuEnv(
        cfg=SimpleNamespace(
            seed=0,
            env_fps=60,
            obs_fps=15,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            max_episode_steps=25000,
            latency_type="fixed",
            fixed_latency_ms=1000.0 / 60.0,
            add_latency_info=True,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
            episode_metrics_path="",
            latency_seed=123,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    try:
        env.reset()
        _obs, rewards, terminated, truncated, infos = env.step(torch.zeros(2, dtype=torch.int64, device="cuda"))

        assert fused_calls == 1
        assert rewards.device.type == "cuda"
        assert terminated.device.type == "cuda"
        assert truncated.device.type == "cuda"
        for info in infos:
            for key in (
                "queued_action",
                "ready_raw_frame",
                "raw_frames_advanced",
                "pending_action_count",
                "latency_raw_frames",
                "latency_ms",
                "decision_step_idx",
                "env_step_idx",
                "raw_frame",
                "applied_action",
                "obs_stride_raw_frames",
            ):
                assert key in info
    finally:
        env.close()


def test_demon_attack_gpu_max_episode_steps_uses_fused_decision_path(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    fused_calls = 0
    original = CudaDemonAttackCore.step_decision

    def wrapped(self, *args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original(self, *args, **kwargs)

    def fail_step_batch(self, *args, **kwargs):
        raise AssertionError("Demon GPU env should not use per-raw-frame step_batch")

    monkeypatch.setattr(CudaDemonAttackCore, "step_decision", wrapped)
    monkeypatch.setattr(CudaDemonAttackCore, "step_batch", fail_step_batch)

    env = DemonAttackBatchedGpuEnv(
        cfg=SimpleNamespace(
            seed=0,
            env_fps=60,
            obs_fps=15,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            max_episode_steps=1,
            latency_type="fixed",
            fixed_latency_ms=1000.0 / 60.0,
            add_latency_info=False,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
            episode_metrics_path="",
            latency_seed=123,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    try:
        env.reset()
        _obs, _rewards, _terminated, truncated, infos = env.step(torch.zeros(2, dtype=torch.int64, device="cuda"))

        assert fused_calls == 1
        assert torch.equal(truncated, torch.ones(2, dtype=torch.bool, device="cuda"))
        assert env._raw_episode_steps.tolist() == [0, 0]
        assert all("reset_info" in info for info in infos)
    finally:
        env.close()
