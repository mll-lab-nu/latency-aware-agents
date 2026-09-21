from __future__ import annotations


import pytest


def test_asterix_cuda_core_runs_full_gpu_batch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from training.asterix_sf.gpu_core import CudaAsterixCore

    core = CudaAsterixCore(
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
    assert core.minimal_action_values == [0, 8, 1, 2, 4, 9, 10, 5, 6]
    assert core.reset_raw_frame_counts.cpu().tolist() == [22, 5]
    assert [info["ale.lives"] for info in infos] == [3, 3]


def test_asterix_cule_gpu_matches_ale_fixed_trace():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ale_py")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
    from latency_bench.envs.gymnasium_asterix import make_latency_sensitive_asterix_env
    from training.asterix_sf.gpu_core import CudaAsterixCore
    from training.common.gpu_latency import CudaLatencyScheduler

    seed = 0
    cpu = make_latency_sensitive_asterix_env(
        base_env_id="ALE/Asterix-v5",
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
    gpu = CudaAsterixCore(
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
        _, cpu_info = cpu.reset(seed=seed)
        latency.reset(seeds=[seed])
        gpu.reset_batch(seeds=[seed])
        assert int(gpu.reset_raw_frame_counts[0]) == cpu_info["episode_frame_number"]

        raw_steps = torch.zeros(1, dtype=torch.int64, device="cuda")
        active = torch.ones(1, dtype=torch.bool, device="cuda")
        cpu_score = 0.0
        terminal_seen = False
        for decision in range(400):
            action = (decision * 7 + 3) % 9
            cpu_raw_frames = 0
            cpu_reward = 0.0
            for _ in range(4):
                _obs, reward, terminated, truncated, cpu_info = cpu.step(action)
                cpu_raw_frames += 1
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

            assert raw_frames_advanced == cpu_raw_frames
            assert float(result.rewards[0]) == cpu_reward
            assert bool(result.terminated[0]) == bool(terminated)
            assert bool(result.truncated[0]) == bool(truncated)
            assert int(result.scores[0]) == int(cpu_score)
            assert int(result.lives[0]) == int(cpu_info["lives"])
            assert int(gpu.reset_raw_frame_counts[0]) + int(raw_steps[0]) == cpu_info["episode_frame_number"]

            if terminated or truncated:
                terminal_seen = True
                break

        assert terminal_seen
    finally:
        cpu.close()
