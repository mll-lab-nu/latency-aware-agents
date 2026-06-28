from __future__ import annotations

import numpy as np
import pytest
import torch


class _LatencyDummyEnv:
    def __init__(self):
        import gymnasium as gym

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        del kwargs
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.asarray([int(action)], dtype=np.float32), 0.0, False, False, {}


def test_batched_latency_fixed_one_frame_then_noop_matches_scalar_action_timing():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=1,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="issue_order_fifo",
        latency_distribution_spec=None,
        seeds=[7],
        env_reset_seed_bases=[100],
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])

    scheduler.begin_decision(np.asarray([1], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [0]
    scheduler.record_raw_step(np.asarray([True]))
    decision = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert int(decision.latency_raw_frames[0]) == 1
    assert int(decision.queued_actions[0]) == 1
    assert int(decision.applied_actions[0]) == 0

    scheduler.begin_decision(np.asarray([0], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [1]
    scheduler.record_raw_step(np.asarray([True]))
    decision = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert int(decision.queued_actions[0]) == 0
    assert int(decision.applied_actions[0]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("ordering_policy", ("issue_order_fifo", "latest_ready"))
@pytest.mark.parametrize("hold_policy", ("one_frame_then_noop", "hold"))
def test_cuda_batched_latency_fixed_path_matches_cpu_scheduler(ordering_policy, hold_policy):
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.gpu_latency import CudaLatencyScheduler

    batch_size = 4
    seeds = [7, 8, 9, 10]
    reset_seeds = [101, 102, 103, 104]
    cpu = BatchedFlappyLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=1,
        env_fps=30,
        obs_fps=30,
        hold_policy=hold_policy,
        ordering_policy=ordering_policy,
        latency_distribution_spec=None,
        seeds=seeds,
        env_reset_seed_bases=[1001, 1002, 1003, 1004],
        add_latency_info=False,
    )
    cuda = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=1,
        env_fps=30,
        obs_fps=30,
        hold_policy=hold_policy,
        ordering_policy=ordering_policy,
        seeds=seeds,
        env_reset_seed_bases=[1001, 1002, 1003, 1004],
        add_latency_info=False,
        device=torch.device("cuda"),
    )

    assert cuda.reset(seeds=reset_seeds) == cpu.reset(seeds=reset_seeds)

    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    active_rows = np.asarray(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
        ],
        dtype=np.bool_,
    )
    for actions, active in zip(action_rows, active_rows):
        cpu.begin_decision(actions, active_mask=active)
        cuda.begin_decision(
            torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
            active_mask=torch.as_tensor(active, dtype=torch.bool, device="cuda"),
        )

        cpu_raw = cpu.raw_actions(active)
        cuda_raw = cuda.raw_actions(torch.as_tensor(active, dtype=torch.bool, device="cuda"))
        np.testing.assert_array_equal(cuda_raw.detach().cpu().numpy(), cpu_raw)

        cpu.record_raw_step(active)
        cuda.record_raw_step(torch.as_tensor(active, dtype=torch.bool, device="cuda"))

        raw_frames = np.ones(batch_size, dtype=np.int64)
        cpu_result = cpu.finish_decision(active, raw_frames)
        cuda_result = cuda.finish_decision(
            torch.as_tensor(active, dtype=torch.bool, device="cuda"),
            torch.as_tensor(raw_frames, dtype=torch.int64, device="cuda"),
        )
        for field in (
            "queued_actions",
            "applied_actions",
            "latency_raw_frames",
            "ready_raw_frames",
            "raw_frames_advanced",
            "pending_counts",
        ):
            np.testing.assert_array_equal(getattr(cuda_result, field).detach().cpu().numpy(), getattr(cpu_result, field))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "latency_distribution_spec,expected_min,expected_max",
    (
        ({"type": "uniform", "min": 1.2, "max": 3.2}, 2, 4),
        ({"type": "normal", "mean": -10.0, "std": 0.0, "min": 1.2, "max": 3.2}, 2, 4),
    ),
)
def test_cuda_batched_latency_distribution_samples_on_gpu(
    latency_distribution_spec,
    expected_min,
    expected_max,
):
    from latency_bench.core.latency_distribution import parse_latency_distribution
    from training.common.gpu_latency import CudaLatencyScheduler

    scheduler = CudaLatencyScheduler(
        batch_size=4,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=parse_latency_distribution(latency_distribution_spec),
        seeds=[7, 8, 9, 10],
        env_reset_seed_bases=[1001, 1002, 1003, 1004],
        add_latency_info=False,
        device=torch.device("cuda"),
    )
    scheduler.reset(seeds=[101, 102, 103, 104])

    for _ in range(8):
        active = torch.ones(4, dtype=torch.bool, device="cuda")
        scheduler.begin_decision(torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda"), active_mask=active)
        result = scheduler.finish_decision(active, torch.ones(4, dtype=torch.int64, device="cuda"))
        sampled = result.latency_raw_frames.detach().cpu().numpy()

        assert np.all(sampled >= expected_min)
        assert np.all(sampled <= expected_max)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_latency_core_decision_fused_path_matches_cpu_for_stride_gt_one():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.gpu_latency import CudaLatencyScheduler
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    batch_size = 4
    raw_steps = 2
    seeds = [101, 202, 303, 404]
    latency_seeds = [11, 12, 13, 14]
    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    active_rows = np.asarray(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
            [True, True, True, False],
        ],
        dtype=np.bool_,
    )

    cpu_core = BatchedFlappyBirdCore(batch_size=batch_size)
    cuda_core = CudaFlappyBirdCore(batch_size=batch_size, device=torch.device("cuda"))
    cpu_latency = BatchedFlappyLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=1,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        seeds=latency_seeds,
        env_reset_seed_bases=[1001, 1002, 1003, 1004],
        add_latency_info=False,
    )
    cuda_latency = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=1,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        seeds=latency_seeds,
        env_reset_seed_bases=[1001, 1002, 1003, 1004],
        add_latency_info=False,
        device=torch.device("cuda"),
    )

    cpu_core.reset_batch(seeds=seeds)
    cuda_core.reset_batch(seeds=seeds)
    cpu_latency.reset(seeds=[201, 202, 203, 204])
    cuda_latency.reset(seeds=[201, 202, 203, 204])

    for actions, active in zip(action_rows, active_rows):
        cpu_latency.begin_decision(actions, active_mask=active)
        rewards = np.zeros(batch_size, dtype=np.float64)
        terminated = np.zeros(batch_size, dtype=np.bool_)
        truncated = np.zeros(batch_size, dtype=np.bool_)
        raw_frames_advanced = np.zeros(batch_size, dtype=np.int64)
        still_active = active.copy()
        latest_scores = cpu_core.score.copy()
        for _ in range(raw_steps):
            raw_actions = cpu_latency.raw_actions(still_active)
            step_result = cpu_core.step_batch(raw_actions, active_mask=still_active)
            advanced = still_active.copy()
            raw_frames_advanced[advanced] += 1
            rewards[advanced] += step_result.rewards[advanced]
            terminated |= step_result.terminated
            truncated |= step_result.truncated
            latest_scores = step_result.scores
            cpu_latency.record_raw_step(advanced)
            still_active &= ~np.logical_or(terminated, truncated)
            if not np.any(still_active):
                break
        cpu_decision = cpu_latency.finish_decision(active, raw_frames_advanced)

        cuda_decision = cuda_latency.step_core_decision_numpy(
            cuda_core,
            torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
            active_mask=torch.as_tensor(active, dtype=torch.bool, device="cuda"),
            episode_step_idx=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
            max_episode_steps=0,
        )
        torch.cuda.synchronize()

        np.testing.assert_array_equal(cuda_decision.rewards, rewards)
        np.testing.assert_array_equal(cuda_decision.terminated, terminated)
        np.testing.assert_array_equal(cuda_decision.truncated, truncated)
        np.testing.assert_array_equal(cuda_decision.scores, latest_scores)
        np.testing.assert_array_equal(cuda_decision.raw_frames_advanced, raw_frames_advanced)
        np.testing.assert_array_equal(cuda_decision.latency_raw_frames, cpu_decision.latency_raw_frames)
        np.testing.assert_allclose(cuda_core.observations(), cpu_core.observations(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_latency_core_decision_fused_path_supports_distribution_latency():
    from latency_bench.core.latency_distribution import parse_latency_distribution

    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.gpu_latency import CudaLatencyScheduler
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    batch_size = 2
    raw_steps = 2
    spec = parse_latency_distribution({"type": "normal", "mean": 1.2, "std": 0.0, "min": 1.2, "max": 1.2})
    actions = np.asarray([1, 0], dtype=np.int64)
    active = np.asarray([True, True], dtype=np.bool_)

    cpu_core = BatchedFlappyBirdCore(batch_size=batch_size)
    cuda_core = CudaFlappyBirdCore(batch_size=batch_size, device=torch.device("cuda"))
    cpu_latency = BatchedFlappyLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=spec,
        seeds=[11, 12],
        env_reset_seed_bases=[1001, 1002],
        add_latency_info=False,
    )
    cuda_latency = CudaLatencyScheduler(
        batch_size=batch_size,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=spec,
        seeds=[11, 12],
        env_reset_seed_bases=[1001, 1002],
        add_latency_info=False,
        device=torch.device("cuda"),
    )

    cpu_core.reset_batch(seeds=[101, 202])
    cuda_core.reset_batch(seeds=[101, 202])
    cpu_latency.reset(seeds=[201, 202])
    cuda_latency.reset(seeds=[201, 202])

    cpu_latency.begin_decision(actions, active_mask=active)
    rewards = np.zeros(batch_size, dtype=np.float64)
    terminated = np.zeros(batch_size, dtype=np.bool_)
    truncated = np.zeros(batch_size, dtype=np.bool_)
    raw_frames_advanced = np.zeros(batch_size, dtype=np.int64)
    still_active = active.copy()
    latest_scores = cpu_core.score.copy()
    for _ in range(raw_steps):
        raw_actions = cpu_latency.raw_actions(still_active)
        step_result = cpu_core.step_batch(raw_actions, active_mask=still_active)
        advanced = still_active.copy()
        raw_frames_advanced[advanced] += 1
        rewards[advanced] += step_result.rewards[advanced]
        terminated |= step_result.terminated
        truncated |= step_result.truncated
        latest_scores = step_result.scores
        cpu_latency.record_raw_step(advanced)
        still_active &= ~np.logical_or(terminated, truncated)
    cpu_decision = cpu_latency.finish_decision(active, raw_frames_advanced)

    cuda_decision = cuda_latency.step_core_decision_numpy(
        cuda_core,
        torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
        active_mask=torch.as_tensor(active, dtype=torch.bool, device="cuda"),
        episode_step_idx=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
        max_episode_steps=0,
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(cuda_decision.rewards, rewards)
    np.testing.assert_array_equal(cuda_decision.terminated, terminated)
    np.testing.assert_array_equal(cuda_decision.truncated, truncated)
    np.testing.assert_array_equal(cuda_decision.scores, latest_scores)
    np.testing.assert_array_equal(cuda_decision.raw_frames_advanced, raw_frames_advanced)
    np.testing.assert_array_equal(cuda_decision.latency_raw_frames, cpu_decision.latency_raw_frames)
    np.testing.assert_allclose(cuda_core.observations(), cpu_core.observations(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_latency_core_decision_resets_offscreen_pipe_during_stride():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
    from flappy_bird_gymnasium.envs.constants import PIPE_VEL_X, PIPE_WIDTH
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.gpu_latency import CudaLatencyScheduler
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    cpu_core = BatchedFlappyBirdCore(batch_size=1)
    cuda_core = CudaFlappyBirdCore(batch_size=1, device=torch.device("cuda"))
    cpu_latency = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        seeds=[11],
        env_reset_seed_bases=[1001],
        add_latency_info=False,
    )
    cuda_latency = CudaLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=60,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        seeds=[11],
        env_reset_seed_bases=[1001],
        add_latency_info=False,
        device=torch.device("cuda"),
    )

    cpu_core.reset_batch(seeds=[101])
    cuda_core.reset_batch(seeds=[101])
    cpu_latency.reset(seeds=[201])
    cuda_latency.reset(seeds=[201])

    offscreen_next_x = -PIPE_WIDTH - PIPE_VEL_X - 1
    cpu_core.upper_pipe_xy[0, 0, 0] = offscreen_next_x
    cpu_core.lower_pipe_xy[0, 0, 0] = offscreen_next_x
    cuda_core.upper_pipe_xy[0, 0, 0] = offscreen_next_x
    cuda_core.lower_pipe_xy[0, 0, 0] = offscreen_next_x

    actions = np.asarray([0], dtype=np.int64)
    active = np.asarray([True], dtype=np.bool_)
    cpu_latency.begin_decision(actions, active_mask=active)
    rewards = np.zeros(1, dtype=np.float64)
    terminated = np.zeros(1, dtype=np.bool_)
    truncated = np.zeros(1, dtype=np.bool_)
    raw_frames_advanced = np.zeros(1, dtype=np.int64)
    still_active = active.copy()
    latest_scores = cpu_core.score.copy()
    for _ in range(2):
        raw_actions = cpu_latency.raw_actions(still_active)
        step_result = cpu_core.step_batch(raw_actions, active_mask=still_active)
        advanced = still_active.copy()
        raw_frames_advanced[advanced] += 1
        rewards[advanced] += step_result.rewards[advanced]
        terminated |= step_result.terminated
        truncated |= step_result.truncated
        latest_scores = step_result.scores
        cpu_latency.record_raw_step(advanced)
        still_active &= ~np.logical_or(terminated, truncated)
    cpu_latency.finish_decision(active, raw_frames_advanced)

    cuda_decision = cuda_latency.step_core_decision_numpy(
        cuda_core,
        torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
        active_mask=torch.as_tensor(active, dtype=torch.bool, device="cuda"),
        episode_step_idx=torch.zeros(1, dtype=torch.int64, device="cuda"),
        max_episode_steps=0,
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(cuda_decision.rewards, rewards)
    np.testing.assert_array_equal(cuda_decision.terminated, terminated)
    np.testing.assert_array_equal(cuda_decision.truncated, truncated)
    np.testing.assert_array_equal(cuda_decision.scores, latest_scores)
    np.testing.assert_array_equal(cuda_decision.raw_frames_advanced, raw_frames_advanced)
    np.testing.assert_allclose(cuda_core.upper_pipe_xy.detach().cpu().numpy(), cpu_core.upper_pipe_xy, rtol=0, atol=0)
    np.testing.assert_allclose(cuda_core.lower_pipe_xy.detach().cpu().numpy(), cpu_core.lower_pipe_xy, rtol=0, atol=0)
    np.testing.assert_allclose(cuda_core.observations(), cpu_core.observations(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_latency_core_decision_resets_same_pipe_twice_during_large_stride():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
    from flappy_bird_gymnasium.envs.constants import PIPE_VEL_X, PIPE_WIDTH
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.gpu_latency import CudaLatencyScheduler
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    raw_steps = 130
    cpu_core = BatchedFlappyBirdCore(batch_size=1, pipe_gap=10000)
    cuda_core = CudaFlappyBirdCore(batch_size=1, pipe_gap=10000, device=torch.device("cuda"))
    cpu_latency = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=3900,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        seeds=[11],
        env_reset_seed_bases=[1001],
        add_latency_info=False,
    )
    cuda_latency = CudaLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=3900,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        seeds=[11],
        env_reset_seed_bases=[1001],
        add_latency_info=False,
        device=torch.device("cuda"),
    )

    cpu_core.reset_batch(seeds=[101])
    cuda_core.reset_batch(seeds=[101])
    cpu_latency.reset(seeds=[201])
    cuda_latency.reset(seeds=[201])

    cpu_core.ground_y[0] = 10000.0
    cuda_core.ground_y[0] = 10000.0
    cpu_core.player_y[0] = 200.0
    cuda_core.player_y[0] = 200.0
    cpu_core.player_vel_y[0] = 0.0
    cuda_core.player_vel_y[0] = 0.0

    offscreen_next_x = -PIPE_WIDTH - PIPE_VEL_X - 1
    cpu_core.upper_pipe_xy[0, 0, 0] = offscreen_next_x
    cpu_core.lower_pipe_xy[0, 0, 0] = offscreen_next_x
    cuda_core.upper_pipe_xy[0, 0, 0] = offscreen_next_x
    cuda_core.lower_pipe_xy[0, 0, 0] = offscreen_next_x
    cpu_core.upper_pipe_xy[0, 1:, 0] = 2000.0
    cpu_core.lower_pipe_xy[0, 1:, 0] = 2000.0
    cuda_core.upper_pipe_xy[0, 1:, 0] = 2000.0
    cuda_core.lower_pipe_xy[0, 1:, 0] = 2000.0

    actions = np.asarray([0], dtype=np.int64)
    active = np.asarray([True], dtype=np.bool_)
    cpu_latency.begin_decision(actions, active_mask=active)
    rewards = np.zeros(1, dtype=np.float64)
    terminated = np.zeros(1, dtype=np.bool_)
    truncated = np.zeros(1, dtype=np.bool_)
    raw_frames_advanced = np.zeros(1, dtype=np.int64)
    still_active = active.copy()
    latest_scores = cpu_core.score.copy()
    for _ in range(raw_steps):
        raw_actions = cpu_latency.raw_actions(still_active)
        step_result = cpu_core.step_batch(raw_actions, active_mask=still_active)
        advanced = still_active.copy()
        raw_frames_advanced[advanced] += 1
        rewards[advanced] += step_result.rewards[advanced]
        terminated |= step_result.terminated
        truncated |= step_result.truncated
        latest_scores = step_result.scores
        cpu_latency.record_raw_step(advanced)
        still_active &= ~np.logical_or(terminated, truncated)
    cpu_latency.finish_decision(active, raw_frames_advanced)
    assert int(raw_frames_advanced[0]) == raw_steps

    cuda_decision = cuda_latency.step_core_decision_numpy(
        cuda_core,
        torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
        active_mask=torch.as_tensor(active, dtype=torch.bool, device="cuda"),
        episode_step_idx=torch.zeros(1, dtype=torch.int64, device="cuda"),
        max_episode_steps=0,
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(cuda_decision.rewards, rewards)
    np.testing.assert_array_equal(cuda_decision.terminated, terminated)
    np.testing.assert_array_equal(cuda_decision.truncated, truncated)
    np.testing.assert_array_equal(cuda_decision.scores, latest_scores)
    np.testing.assert_array_equal(cuda_decision.raw_frames_advanced, raw_frames_advanced)
    np.testing.assert_allclose(cuda_core.upper_pipe_xy.detach().cpu().numpy(), cpu_core.upper_pipe_xy, rtol=0, atol=0)
    np.testing.assert_allclose(cuda_core.lower_pipe_xy.detach().cpu().numpy(), cpu_core.lower_pipe_xy, rtol=0, atol=0)
    np.testing.assert_allclose(cuda_core.observations(), cpu_core.observations(), rtol=0, atol=0)


def test_batched_latency_reset_clears_pending_actions():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=3,
        env_fps=30,
        obs_fps=30,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        latency_distribution_spec=None,
        seeds=[7],
        env_reset_seed_bases=[100],
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])
    scheduler.begin_decision(np.asarray([1], dtype=np.int64), active_mask=np.asarray([True]))
    scheduler.reset(indices=[0], seeds=[456])

    scheduler.begin_decision(np.asarray([0], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [0]


@pytest.mark.parametrize(
    "latency_distribution_spec",
    (
        {"type": "uniform", "min": 0, "max": 3},
        {"type": "normal", "mean": 1, "std": 1, "min": 0, "max": 3},
    ),
)
def test_batched_latency_sampled_distribution_matches_scalar_wrapper(latency_distribution_spec):
    from latency_bench.core.latency_distribution import parse_latency_distribution
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.common.latency_wrapper import ActionLatencyWrapper

    spec = parse_latency_distribution(latency_distribution_spec)
    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=spec,
        seeds=[17],
        env_reset_seed_bases=[101],
        add_latency_info=True,
    )
    scalar = ActionLatencyWrapper(
        _LatencyDummyEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=spec,
        seed=17,
        env_reset_seed_base=101,
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])
    scalar.reset(seed=123)

    for action in [0, 1, 1, 0, 1, 0, 0, 1]:
        scheduler.begin_decision(np.asarray([action], dtype=np.int64), active_mask=np.asarray([True]))
        advanced = np.zeros(1, dtype=np.int64)
        for _ in range(scheduler.clock.obs_stride_raw_frames):
            scheduler.raw_actions(np.asarray([True]))
            scheduler.record_raw_step(np.asarray([True]))
            advanced[0] += 1
        decision = scheduler.finish_decision(np.asarray([True]), advanced)
        info = scheduler.decision_info_for_agent(0, decision)

        _, _, _, _, scalar_info = scalar.step(action)
        assert info["latency_raw_frames"] == scalar_info["latency_raw_frames"]
        assert info["queued_action"] == scalar_info["queued_action"]
        assert info["ready_raw_frame"] == scalar_info["ready_raw_frame"]
        assert info["applied_action"] == scalar_info["applied_action"]
        assert info["decision_step_idx"] == scalar_info["decision_step_idx"]
