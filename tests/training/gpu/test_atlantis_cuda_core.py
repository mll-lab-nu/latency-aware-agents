from __future__ import annotations

import numpy as np
import pytest


_ALL_NOOP_CACHE_SEEDS = (
    4,
    29,
    38,
    28,
    1,
    30,
    13,
    27,
    22,
    32,
    11,
    25,
    106,
    7,
    12,
    23,
    16,
    57,
    2,
    67,
    6,
    0,
    130,
    21,
    3,
    73,
    10,
    14,
    5,
    19,
)


def _make_cpu_atlantis_env():
    from latency_bench.envs.tasks.atlantis import make_latency_sensitive_atlantis_env

    return make_latency_sensitive_atlantis_env(
        base_env_id="ALE/Atlantis-v5",
        render_mode="rgb_array",
        screen_size=84,
        noop_max=30,
        base_make_kwargs={
            "obs_type": "rgb",
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "mode": 0,
            "difficulty": 0,
            "max_num_frames_per_episode": 108_000,
        },
    )


def test_atlantis_cuda_core_runs_full_gpu_batch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from training.atlantis_sf.gpu_core import CudaAtlantisCore

    core = CudaAtlantisCore(
        batch_size=2,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=False,
        obs_stride=4,
        frame_stack=4,
    )
    obs, infos = core.reset_batch(seeds=[0, 1])

    assert obs.is_cuda
    assert obs.dtype == torch.uint8
    assert tuple(obs.shape) == (2, 1, 84, 84)
    assert core.minimal_action_values == [0, 16, 17, 18]
    assert core.reset_raw_frame_counts.cpu().tolist() == [22, 5]
    assert [info["ale.lives"] for info in infos] == [6, 6]


def test_atlantis_cule_gpu_matches_ale_all_noop_reset_states():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ale_py")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
    from training.atlantis_sf.gpu_core import CudaAtlantisCore
    from training.common.gpu_latency import CudaLatencyScheduler

    batch_size = len(_ALL_NOOP_CACHE_SEEDS)
    cpu_envs = [_make_cpu_atlantis_env() for _ in range(batch_size)]
    gpu = CudaAtlantisCore(
        batch_size=batch_size,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=False,
        obs_stride=4,
        frame_stack=4,
    )
    latency = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=15,
        hold_policy=ActionHoldPolicy.HOLD,
        ordering_policy=OrderingPolicy.LATEST_READY,
        seeds=_ALL_NOOP_CACHE_SEEDS,
        env_reset_seed_bases=[seed + 104729 for seed in _ALL_NOOP_CACHE_SEEDS],
        add_latency_info=False,
        device="cuda",
    )

    try:
        cpu_infos = [env.reset(seed=seed)[1] for env, seed in zip(cpu_envs, _ALL_NOOP_CACHE_SEEDS)]
        latency.reset(seeds=_ALL_NOOP_CACHE_SEEDS)
        gpu.reset_batch(seeds=_ALL_NOOP_CACHE_SEEDS)

        assert gpu.cache_index.cpu().tolist() == list(range(batch_size))
        assert gpu.reset_raw_frame_counts.cpu().tolist() == list(range(1, batch_size + 1))
        assert [info["episode_frame_number"] for info in cpu_infos] == list(range(1, batch_size + 1))
        assert [info["lives"] for info in cpu_infos] == [6] * batch_size

        raw_steps = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
        active = np.ones(batch_size, dtype=np.bool_)
        cpu_scores = np.zeros(batch_size, dtype=np.int64)

        for decision in range(256):
            active_before = active.copy()
            actions = np.asarray(
                [(decision * (env_idx + 2) + env_idx + 1) % 4 for env_idx in range(batch_size)],
                dtype=np.int64,
            )
            cpu_rewards = np.zeros(batch_size, dtype=np.float32)
            cpu_terminated = np.zeros(batch_size, dtype=np.bool_)
            cpu_truncated = np.zeros(batch_size, dtype=np.bool_)
            cpu_raw_frames = np.zeros(batch_size, dtype=np.int64)

            for env_idx in np.flatnonzero(active_before):
                for _ in range(4):
                    _, reward, terminated, truncated, info = cpu_envs[env_idx].step(int(actions[env_idx]))
                    cpu_rewards[env_idx] += reward
                    cpu_raw_frames[env_idx] += 1
                    cpu_infos[env_idx] = info
                    if terminated or truncated:
                        cpu_terminated[env_idx] = terminated
                        cpu_truncated[env_idx] = truncated
                        break

            result = gpu.step_decision(
                latency,
                actions,
                active_mask=active_before,
                episode_step_idx=raw_steps,
                max_episode_steps=108_000,
            )
            cpu_scores += cpu_rewards.astype(np.int64)
            active_indices = np.flatnonzero(active_before)

            np.testing.assert_array_equal(result.raw_frames_advanced[active_indices], cpu_raw_frames[active_indices])
            np.testing.assert_array_equal(result.rewards[active_indices], cpu_rewards[active_indices])
            np.testing.assert_array_equal(result.terminated[active_indices], cpu_terminated[active_indices])
            np.testing.assert_array_equal(result.truncated[active_indices], cpu_truncated[active_indices])
            np.testing.assert_array_equal(result.scores[active_indices], cpu_scores[active_indices])
            np.testing.assert_array_equal(
                result.lives[active_indices],
                [cpu_infos[env_idx]["lives"] for env_idx in active_indices],
            )
            np.testing.assert_array_equal(
                gpu.reset_raw_frame_counts.cpu().numpy()[active_indices] + raw_steps.cpu().numpy()[active_indices],
                [cpu_infos[env_idx]["episode_frame_number"] for env_idx in active_indices],
            )

            active &= ~(cpu_terminated | cpu_truncated)
            if not active.any():
                break
    finally:
        for env in cpu_envs:
            env.close()


def test_atlantis_cule_gpu_matches_ale_fixed_trace():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ale_py")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
    from training.atlantis_sf.gpu_core import CudaAtlantisCore
    from training.common.gpu_latency import CudaLatencyScheduler

    seed = 0
    cpu = _make_cpu_atlantis_env()
    gpu = CudaAtlantisCore(
        batch_size=1,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=False,
        obs_stride=4,
        frame_stack=4,
    )
    latency = CudaLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=15,
        hold_policy=ActionHoldPolicy.HOLD,
        ordering_policy=OrderingPolicy.LATEST_READY,
        seeds=[seed],
        env_reset_seed_bases=[seed + 104729],
        add_latency_info=False,
        device="cuda",
    )

    try:
        cpu_obs, cpu_info = cpu.reset(seed=seed)
        latency.reset(seeds=[seed])
        gpu_obs, _ = gpu.reset_batch(seeds=[seed])
        assert tuple(gpu_obs.shape) == (1, 1, 84, 84)
        assert np.asarray(cpu_obs).shape == (1, 84, 84)
        assert int(gpu.reset_raw_frame_counts[0]) == cpu_info["episode_frame_number"]

        raw_steps = torch.zeros(1, dtype=torch.int64, device="cuda")
        active = torch.ones(1, dtype=torch.bool, device="cuda")
        cpu_score = 0.0
        for decision in range(2_000):
            action = (decision * 3 + 1) % 4
            cpu_frames = []
            cpu_reward = 0.0
            for _ in range(4):
                obs, reward, terminated, truncated, cpu_info = cpu.step(action)
                cpu_frames.append(np.asarray(obs))
                cpu_reward += float(reward)
                if terminated or truncated:
                    break

            result = gpu.step_decision(
                latency,
                [action],
                active_mask=active,
                episode_step_idx=raw_steps,
                max_episode_steps=108_000,
            )
            raw_frames_advanced = int(result.raw_frames_advanced[0])
            cpu_score += cpu_reward

            assert raw_frames_advanced == len(cpu_frames)
            assert float(result.rewards[0]) == cpu_reward, (
                decision,
                float(result.rewards[0]),
                cpu_reward,
                int(result.scores[0]),
                cpu_score,
            )
            assert bool(result.terminated[0]) == bool(terminated)
            assert bool(result.truncated[0]) == bool(truncated)
            assert int(result.scores[0]) == int(cpu_score)
            assert int(result.lives[0]) == int(cpu_info["lives"])
            assert int(gpu.reset_raw_frame_counts[0]) + int(raw_steps[0]) == cpu_info["episode_frame_number"]
            if terminated or truncated:
                break
        else:
            pytest.fail("Atlantis fixed trace did not reach an episode boundary")
    finally:
        cpu.close()
