from __future__ import annotations

import numpy as np
import pytest
import torch

from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path


ensure_flappy_gymnasium_on_path()

import flappy_bird_gymnasium  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from flappy_bird_gymnasium.envs.render_state import (  # noqa: E402
    BatchedFlappyRenderState,
    FlappyRenderState,
)
from flappy_bird_gymnasium.gpu_render import (  # noqa: E402
    FlappyRenderAssets,
    TorchFlappyRenderer,
)
from training.flappy_ppo.env_batch import SyncFlappyBatch  # noqa: E402


def _make_rgb_env(seed: int):
    env = gym.make(
        "FlappyBird-v0",
        render_mode="rgb_array",
        use_lidar=False,
        normalize_obs=True,
    )
    env.reset(seed=seed)
    return env


def _edge_states() -> list[FlappyRenderState]:
    return [
        FlappyRenderState(
            player_x=57.8,
            player_y=245.2,
            player_index=0,
            visible_rotation=20,
            upper_pipes=((-30.9, -210.0), (120.5, -260.0), (300.0, -180.0)),
            lower_pipes=((-30.9, 190.0), (120.5, 250.0), (300.0, 220.0)),
            ground_x=-12.4,
            ground_y=404.4,
            screen_width=288,
            screen_height=512,
        ),
        FlappyRenderState(
            player_x=-4.8,
            player_y=5.9,
            player_index=1,
            visible_rotation=-20,
            upper_pipes=((0.0, -300.0), (145.0, -230.0), (287.7, -280.0)),
            lower_pipes=((0.0, 100.0), (145.0, 210.0), (287.7, 180.0)),
            ground_x=-40.0,
            ground_y=404.4,
            screen_width=288,
            screen_height=512,
        ),
        FlappyRenderState(
            player_x=270.2,
            player_y=390.8,
            player_index=2,
            visible_rotation=-90,
            upper_pipes=((-52.0, -240.0), (60.0, -210.0), (220.0, -320.0)),
            lower_pipes=((-52.0, 160.0), (60.0, 240.0), (220.0, 90.0)),
            ground_x=0.0,
            ground_y=404.4,
            screen_width=288,
            screen_height=512,
        ),
    ]


def test_torch_full_render_matches_env_rgb_array_for_seeded_steps():
    env = _make_rgb_env(seed=123)
    try:
        assets = FlappyRenderAssets.for_state(env.unwrapped.get_render_state())
        renderer = TorchFlappyRenderer(assets, device="cpu")
        actions = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

        for action in [None, *actions]:
            if action is not None:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()
            state = env.unwrapped.get_render_state()
            expected = env.render()
            actual = renderer.render(state)[0].cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
    finally:
        env.close()


def test_torch_full_batch_render_matches_env_rgb_array_across_steps_and_reset():
    envs = [_make_rgb_env(seed=idx) for idx in range(4)]
    try:
        assets = FlappyRenderAssets.for_state(envs[0].unwrapped.get_render_state())
        renderer = TorchFlappyRenderer(assets, device="cpu")
        actions = np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.int64,
        )

        for step_idx, step_actions in enumerate(actions):
            for env_idx, (env, action) in enumerate(zip(envs, step_actions)):
                _, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated or (step_idx == 2 and env_idx == 1):
                    env.reset()

            states = [env.unwrapped.get_render_state() for env in envs]
            expected = np.stack([env.render() for env in envs], axis=0)
            batch = BatchedFlappyRenderState.from_states(states)
            actual = renderer.render(batch).cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
    finally:
        for env in envs:
            env.close()


def test_torch_full_batch_render_matches_scalar_reference_for_edge_states():
    states = _edge_states()
    assets = FlappyRenderAssets.for_state(states[0])
    renderer = TorchFlappyRenderer(assets, device="cpu")
    batch = BatchedFlappyRenderState.from_states(states)

    actual = renderer.render(batch)
    scalar = renderer._render_scalar(batch)
    expected = np.stack([assets.reference_frame(state) for state in states], axis=0)

    assert torch.equal(actual, scalar)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_full_render_cpu_cuda_parity():
    states = _edge_states()
    assets = FlappyRenderAssets.for_state(states[0])
    batch = BatchedFlappyRenderState.from_states(states)
    cpu = TorchFlappyRenderer(assets, device="cpu").render(batch)
    cuda = TorchFlappyRenderer(assets, device="cuda").render(batch).cpu()

    assert torch.equal(cpu, cuda)


def test_render_into_rewrites_full_frame_without_stale_pixels():
    first, second = _edge_states()[:2]
    assets = FlappyRenderAssets.for_state(first)
    renderer = TorchFlappyRenderer(assets, device="cpu")
    batch = BatchedFlappyRenderState.from_states([first])
    out = torch.full(
        (1, first.screen_height, first.screen_width, 3),
        255,
        dtype=torch.uint8,
    )

    returned = renderer.render_into(batch, out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out, renderer.render(batch))

    next_batch = BatchedFlappyRenderState.from_states([second])
    renderer.render_into(next_batch, out)

    assert torch.equal(out, renderer.render(next_batch))


def test_render_into_reuses_cached_blit_index_buffers():
    states = _edge_states()
    assets = FlappyRenderAssets.for_state(states[0])
    renderer = TorchFlappyRenderer(assets, device="cpu")
    batch = BatchedFlappyRenderState.from_states(states)
    out = torch.empty(
        (batch.batch_size, batch.screen_height, batch.screen_width, 3),
        dtype=torch.uint8,
    )

    renderer.render_into(batch, out)
    first_ptrs = {
        key: tuple(tensor.data_ptr() for tensor in buffers)
        for key, buffers in renderer._blit_index_cache.items()
    }

    assert first_ptrs

    renderer.render_into(batch, out)
    second_ptrs = {
        key: tuple(tensor.data_ptr() for tensor in buffers)
        for key, buffers in renderer._blit_index_cache.items()
    }

    assert second_ptrs == first_ptrs
    assert torch.equal(out, renderer._render_scalar(batch))


def test_sync_flappy_batch_raw_rgb_buffer_matches_fresh_render_after_reset_step():
    envs = SyncFlappyBatch(
        n_envs=3,
        seed=7,
        fixed_frame_skip=0,
        device="cpu",
    )
    try:
        reset_raw = envs.reset().clone()
        reset_batch = BatchedFlappyRenderState.from_states(envs.render_states())
        assert torch.equal(reset_raw, envs.renderer.render(reset_batch))

        envs.envs[0].unwrapped._score_limit = 0
        _, dones, _, step_raw = envs.step(np.asarray([0, 1, 0], dtype=np.int64))
        step_raw = step_raw.clone()
        step_batch = BatchedFlappyRenderState.from_states(envs.render_states())

        assert bool(dones[0])
        assert torch.equal(step_raw, envs.renderer.render(step_batch))
    finally:
        envs.close()
