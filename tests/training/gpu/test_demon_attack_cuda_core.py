from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import REPO_ROOT


def test_demon_attack_rom_resource_matches_supported_md5():
    from training.demon_attack_sf.gpu_core import load_demon_attack_rom_bytes

    rom = load_demon_attack_rom_bytes()

    assert len(rom) == 4096
    assert hashlib.md5(rom).hexdigest() == "f0e0addc07971561ab80d9abe1b8d333"


def test_demon_attack_cule_decodes_ram_reward_lives_and_terminal():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from pathlib import Path

    from torch.utils.cpp_extension import load_inline

    repo_root = REPO_ROOT
    cule_root = repo_root / "third_party" / "cule"
    ext = load_inline(
        name="demon_attack_cule_decode_test",
        cpp_sources=r"""
            #include <pybind11/pybind11.h>
            #include <array>
            #include <cstdint>

            #include <cule/atari/games.hpp>
            #include <cule/atari/state.hpp>

            namespace py = pybind11;

            py::tuple decode_demon_attack_cases()
            {
                using cule::atari::games::demonattack::lives;
                using cule::atari::games::demonattack::reward;
                using cule::atari::games::demonattack::score;
                using cule::atari::games::demonattack::setTerminal;
                using cule::atari::ram::write;

                std::array<std::array<uint32_t, 32>, 3> ram{};
                std::array<cule::atari::state, 3> states{};

                states[0].score = 0;
                states[1].score = 1200;
                states[2].score = 0;
                for(size_t i = 0; i < states.size(); ++i)
                {
                    states[i].ram = ram[i].data();
                }

                write(states[0].ram, 0x05, 0x56);
                write(states[0].ram, 0x03, 0x34);
                write(states[0].ram, 0x01, 0x12);
                write(states[0].ram, 0x72, 3);
                write(states[0].ram, 0x71, 0);

                write(states[1].ram, 0x05, 0x00);
                write(states[1].ram, 0x03, 0x13);
                write(states[1].ram, 0x01, 0x00);
                write(states[1].ram, 0x72, 0);
                write(states[1].ram, 0x71, 0xBD);

                write(states[2].ram, 0x05, 0xEA);
                write(states[2].ram, 0x03, 0xCD);
                write(states[2].ram, 0x01, 0xAB);
                write(states[2].ram, 0x72, 2);
                write(states[2].ram, 0x71, 0);

                py::list decoded_scores;
                py::list decoded_rewards;
                py::list decoded_lives;
                py::list decoded_terminal;
                for(size_t i = 0; i < states.size(); ++i)
                {
                    setTerminal(states[i]);
                    decoded_scores.append(score(states[i]));
                    decoded_rewards.append(reward(states[i]));
                    decoded_lives.append(lives(states[i]));
                    decoded_terminal.append(bool(states[i].tiaFlags[cule::atari::FLAG_ALE_TERMINAL]));
                }
                return py::make_tuple(decoded_scores, decoded_rewards, decoded_lives, decoded_terminal);
            }

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
            {
                m.def("decode_demon_attack_cases", &decode_demon_attack_cases);
            }
        """,
        extra_include_paths=[str(cule_root)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )

    score, reward, lives, terminal = ext.decode_demon_attack_cases()

    assert score == [123456, 1300, 0]
    assert reward == [123456, 100, 0]
    assert lives == [4, 0, 3]
    assert terminal == [False, True, False]


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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=False,
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


def test_demon_attack_batched_gpu_env_exports_native_raw_rgb_stack():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=True,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    try:
        _, infos = env.reset(seed=123)
        reset_stack = infos[0][ENV_RAW_RGB_FRAME_STACK_INFO_KEY].copy()
        _, _, _, _, step_infos = env.step(torch.zeros(2, dtype=torch.int64, device="cuda"))
        step_stack = step_infos[0][ENV_RAW_RGB_FRAME_STACK_INFO_KEY]

        assert reset_stack.shape == (4, 210, 160, 3)
        assert reset_stack.dtype == np.uint8
        assert np.array_equal(step_stack[:-1], reset_stack[1:])
    finally:
        env.close()


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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=False,
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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=False,
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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=False,
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


def test_demon_attack_cuda_core_gpu_decision_matches_cpu_projection():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.common.gpu_latency import CudaLatencyScheduler
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore

    batch_size = 3
    core_cpu = CudaDemonAttackCore(
        batch_size=batch_size,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=True,
        obs_stride=4,
        frame_stack=4,
    )
    core_gpu = CudaDemonAttackCore(
        batch_size=batch_size,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=True,
        obs_stride=4,
        frame_stack=4,
    )
    seeds = [11, 22, 33]
    core_cpu.reset_batch(seeds=seeds)
    core_gpu.reset_batch(seeds=seeds)
    latency_cpu = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=5,
        env_fps=60,
        obs_fps=15,
        hold_policy="hold",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        sample_distribution_per_episode=False,
        seeds=[101, 102, 103],
        env_reset_seed_bases=[1001, 1002, 1003],
        add_latency_info=False,
        max_pending_actions=None,
        simulated_worker_capacity=1,
        device="cuda",
    )
    latency_gpu = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=5,
        env_fps=60,
        obs_fps=15,
        hold_policy="hold",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        sample_distribution_per_episode=False,
        seeds=[101, 102, 103],
        env_reset_seed_bases=[1001, 1002, 1003],
        add_latency_info=False,
        max_pending_actions=None,
        simulated_worker_capacity=1,
        device="cuda",
    )
    latency_cpu.reset()
    latency_gpu.reset()
    actions = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    active = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    episode_cpu = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    episode_gpu = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

    cpu_result = core_cpu.step_decision(
        latency_cpu,
        actions,
        active_mask=active,
        episode_step_idx=episode_cpu,
        max_episode_steps=25000,
    )
    gpu_result = core_gpu.step_decision_gpu(
        latency_gpu,
        actions,
        active_mask=active,
        episode_step_idx=episode_gpu,
        max_episode_steps=25000,
    )
    torch.cuda.synchronize()

    assert torch.equal(gpu_result.observations, cpu_result.observations)
    assert torch.equal(gpu_result.rewards_gpu.cpu(), torch.from_numpy(cpu_result.rewards))
    assert torch.equal(gpu_result.rewards_clipped_gpu.cpu(), torch.from_numpy(cpu_result.rewards_clipped))
    assert torch.equal(gpu_result.terminated_gpu.cpu(), torch.from_numpy(cpu_result.terminated))
    assert torch.equal(gpu_result.truncated_gpu.cpu(), torch.from_numpy(cpu_result.truncated))
    assert torch.equal(gpu_result.raw_frames_advanced_gpu.cpu(), torch.from_numpy(cpu_result.raw_frames_advanced))
    assert torch.equal(gpu_result.latency_raw_frames_gpu.cpu(), torch.from_numpy(cpu_result.latency_raw_frames))
    assert cpu_result.command_admitted.tolist() == [True, True, True]
    assert gpu_result.command_admitted.tolist() == [True, True, True]

    latency_cpu.fixed_latency_raw_frames = 0
    latency_gpu.fixed_latency_raw_frames = 0
    second_actions = torch.tensor([3, 1, 2], dtype=torch.int64, device="cuda")
    cpu_result = core_cpu.step_decision(
        latency_cpu,
        second_actions,
        active_mask=active,
        episode_step_idx=episode_cpu,
        max_episode_steps=25000,
    )
    gpu_result = core_gpu.step_decision_gpu(
        latency_gpu,
        second_actions,
        active_mask=active,
        episode_step_idx=episode_gpu,
        max_episode_steps=25000,
    )
    torch.cuda.synchronize()

    assert torch.equal(gpu_result.observations, cpu_result.observations)
    assert cpu_result.command_admitted.tolist() == [False, False, False]
    assert gpu_result.command_admitted.tolist() == [False, False, False]
    assert gpu_result.observation_dropped.tolist() == [True, True, True]
    assert gpu_result.raw_frames_advanced_gpu.tolist() == [4, 4, 4]
    assert gpu_result.env_step_indices_gpu.tolist() == [8, 8, 8]
    assert torch.equal(gpu_result.applied_actions_gpu, actions)
    assert torch.equal(gpu_result.applied_actions_gpu.cpu(), torch.from_numpy(cpu_result.applied_actions))
    assert torch.equal(gpu_result.pending_counts_gpu.cpu(), torch.from_numpy(cpu_result.pending_counts))
    assert gpu_result.pending_counts_gpu.tolist() == [0, 0, 0]


def test_demon_attack_batched_gpu_env_fast_step_matches_regular_step_without_full_infos():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    cfg = SimpleNamespace(
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
        simulated_worker_capacity=None,
        episode_metrics_path="",
        latency_seed=123,
        export_env_raw_rgb_frames=False,
    )
    regular = DemonAttackBatchedGpuEnv(
        cfg=cfg,
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )
    fast = DemonAttackBatchedGpuEnv(
        cfg=cfg,
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    try:
        regular.reset()
        fast.reset()
        actions = torch.zeros(2, dtype=torch.int64, device="cuda")
        obs_regular, rewards_regular, terminated_regular, truncated_regular, _infos_regular = regular.step(actions)
        obs_fast, rewards_fast, terminated_fast, truncated_fast, infos_fast = fast.sample_factory_gpu_step(actions)
        torch.cuda.synchronize()

        assert torch.equal(obs_fast["obs"], obs_regular["obs"])
        assert torch.equal(rewards_fast, rewards_regular)
        assert torch.equal(terminated_fast, terminated_regular)
        assert torch.equal(truncated_fast, truncated_regular)
        assert infos_fast == [{}, {}]
        assert torch.equal(fast.sample_factory_step_cpu_tensors["rewards"], regular.sample_factory_step_cpu_tensors["rewards"])
        assert torch.equal(fast.sample_factory_step_cpu_tensors["dones"], regular.sample_factory_step_cpu_tensors["dones"])
    finally:
        regular.close()
        fast.close()


def test_demon_attack_batched_gpu_env_fast_step_preserves_done_reset_info():
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
            simulated_worker_capacity=None,
            episode_metrics_path="",
            latency_seed=123,
            export_env_raw_rgb_frames=False,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=2,
    )

    try:
        env.reset()
        _obs, _rewards, _terminated, truncated, infos = env.sample_factory_gpu_step(
            torch.zeros(2, dtype=torch.int64, device="cuda")
        )
        torch.cuda.synchronize()

        assert torch.equal(truncated, torch.ones(2, dtype=torch.bool, device="cuda"))
        assert env._raw_episode_steps.tolist() == [0, 0]
        assert all("reset_info" in info for info in infos)
        assert all("episode_extra_stats" in info for info in infos)
    finally:
        env.close()


def _clip_parity_cfg(**overrides):
    cfg = dict(
        seed=0,
        env_fps=60,
        obs_fps=15,
        noop_max=30,
        frame_stack=4,
        terminal_on_life_loss=False,
        clip_reward=True,
        max_episode_steps=0,
        latency_type="fixed",
        fixed_latency_ms=0.0,
        add_latency_info=False,
        hold_policy="hold",
        ordering_policy="latest_ready",
        max_pending_actions=None,
        simulated_worker_capacity=None,
        episode_metrics_path="",
        latency_seed=123,
        export_env_raw_rgb_frames=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_demon_attack_gpu_clip_reward_is_sum_of_per_frame_signs():
    """clip_reward must clip per raw frame then sum (matching the CPU path),
    not sign the accumulated decision reward (which would saturate to {-1,0,1}
    and starve the learner relative to the CPU baseline)."""
    torch = pytest.importorskip("torch")
    import numpy as np

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    batch_size = 8
    # A wide decision window (obs_fps=2 -> 30 raw frames per decision) makes it likely
    # that more than one raw frame scores within a single window, which is exactly the
    # case where sum-of-signs (correct) and sign-of-sum (the bug) diverge.
    env = DemonAttackBatchedGpuEnv(
        cfg=_clip_parity_cfg(obs_fps=2),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=batch_size,
    )
    obs_stride = env.clock.obs_stride_raw_frames
    assert obs_stride == 30

    try:
        env.reset()
        # FIRE every decision to provoke scoring; let the env auto-reset on game over
        # so we accumulate many decision windows.
        fire = torch.ones(batch_size, dtype=torch.int64, device="cuda")
        observed_multi_score = False
        for _ in range(300):
            _obs, rewards, _terminated, _truncated, infos = env.step(fire)
            learner = rewards.detach().cpu().numpy()
            raw = np.asarray([info["raw_reward"] for info in infos], dtype=np.float64)
            # Learner reward is a sum of per-frame signs: integer-valued and bounded
            # by the number of raw frames advanced in the decision window.
            np.testing.assert_array_equal(learner, np.round(learner))
            assert np.all(np.abs(learner) <= obs_stride)
            # Demon Attack rewards are non-negative; clipped reward is positive iff scored.
            assert np.all((learner > 0) == (raw > 0))
            assert np.all(learner >= 0)
            if np.any(learner >= 2):
                observed_multi_score = True

        # The discriminating property vs the old sign(sum) behavior: the clipped
        # learner reward can exceed 1 when more than one raw frame scores in a window.
        assert observed_multi_score, "expected at least one decision window with 2+ scoring frames"
    finally:
        env.close()
