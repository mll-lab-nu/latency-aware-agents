from __future__ import annotations

import numpy as np
import pytest
import torch

from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _assert_render_state_equal(actual, expected) -> None:
    np.testing.assert_array_equal(_array(actual.player_xy), _array(expected.player_xy))
    np.testing.assert_array_equal(_array(actual.player_indices), _array(expected.player_indices))
    np.testing.assert_array_equal(_array(actual.visible_rotations), _array(expected.visible_rotations))
    np.testing.assert_array_equal(_array(actual.upper_pipe_xy), _array(expected.upper_pipe_xy))
    np.testing.assert_array_equal(_array(actual.lower_pipe_xy), _array(expected.lower_pipe_xy))
    np.testing.assert_array_equal(_array(actual.ground_xy), _array(expected.ground_xy))
    assert actual.screen_width == expected.screen_width
    assert actual.screen_height == expected.screen_height
    assert actual.bird_color == expected.bird_color
    assert actual.pipe_color == expected.pipe_color
    assert actual.background == expected.background


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_flappy_core_matches_cpu_core_for_seeded_action_stream():
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    seeds = [101, 202, 303, 404]
    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
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
            [True, True, True, True],
            [True, False, True, False],
            [True, True, True, True],
        ],
        dtype=np.bool_,
    )

    cpu = BatchedFlappyBirdCore(batch_size=len(seeds))
    cuda = CudaFlappyBirdCore(batch_size=len(seeds), device=torch.device("cuda"))

    cpu_obs, cpu_infos = cpu.reset_batch(seeds=seeds)
    cuda_obs, cuda_infos = cuda.reset_batch(seeds=seeds)

    np.testing.assert_allclose(cuda_obs, cpu_obs, rtol=0, atol=0)
    assert cuda_infos == cpu_infos
    _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())

    for row_idx, (actions, active) in enumerate(zip(action_rows, active_rows)):
        cpu_result = cpu.step_batch(actions, active_mask=active)
        if row_idx % 2 == 0:
            cuda_result = cuda.step_batch_numpy(actions, active_mask=active)
            torch.cuda.synchronize()

            np.testing.assert_allclose(cuda.observations(), cpu_result.observations, rtol=0, atol=0)
            np.testing.assert_array_equal(cuda_result.rewards, cpu_result.rewards)
            np.testing.assert_array_equal(cuda_result.terminated, cpu_result.terminated)
            np.testing.assert_array_equal(cuda_result.truncated, cpu_result.truncated)
            np.testing.assert_array_equal(cuda_result.scores, cpu_result.scores)
            _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())
            continue

        cuda_result = cuda.step_batch_tensors(
            torch.as_tensor(actions, dtype=torch.int64, device="cuda"),
            active_mask=active,
        )
        torch.cuda.synchronize()

        np.testing.assert_allclose(cuda.observations(), cpu_result.observations, rtol=0, atol=0)
        np.testing.assert_array_equal(cuda_result.rewards.detach().cpu().numpy(), cpu_result.rewards)
        np.testing.assert_array_equal(cuda_result.terminated.detach().cpu().numpy(), cpu_result.terminated)
        np.testing.assert_array_equal(cuda_result.truncated.detach().cpu().numpy(), cpu_result.truncated)
        np.testing.assert_array_equal(cuda_result.scores.detach().cpu().numpy(), cpu_result.scores)
        _assert_render_state_equal(cuda.render_state_batch(), cpu.render_state_batch())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_flappy_render_matches_torch_renderer_for_seeded_core_state():
    from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer
    from training.flappy_sf.flappy_cuda_extension import CudaFlappyRenderer
    from training.flappy_sf.gpu_core import CudaFlappyBirdCore

    seeds = [7, 8, 9]
    actions = np.asarray(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    cpu = BatchedFlappyBirdCore(batch_size=len(seeds))
    cuda = CudaFlappyBirdCore(batch_size=len(seeds), device=torch.device("cuda"))
    cpu.reset_batch(seeds=seeds)
    cuda.reset_batch(seeds=seeds)
    for step_actions in actions:
        cpu.step_batch(step_actions)
        cuda.step_batch_tensors(torch.as_tensor(step_actions, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()

    assets = FlappyRenderAssets.load()
    expected = TorchFlappyRenderer(assets, device="cpu").render(cpu.render_state_batch()).numpy()
    renderer = CudaFlappyRenderer(assets, device=torch.device("cuda"))
    out = torch.empty((len(seeds), cuda.screen_height, cuda.screen_width, 3), dtype=torch.uint8, device="cuda")
    actual = renderer.render_core_into(cuda, out).cpu().numpy()

    np.testing.assert_array_equal(actual, expected)
