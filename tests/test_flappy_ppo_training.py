from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from modeling.flappy_ppo import (
    FlappyPpoObservationConfig,
    FlappyPpoTorchObservationPipeline,
)
from training.flappy_ppo.env_batch import SyncFlappyBatch
from training.flappy_ppo.observation_pipeline import ObservationPipeline


def _args() -> Namespace:
    return Namespace(
        image_size=4,
        grayscale=False,
        stack_frames=3,
    )


def _to_hwc(obs: torch.Tensor) -> np.ndarray:
    return obs.permute(0, 2, 3, 1).cpu().numpy()


def test_training_observation_pipeline_stack_done_reset():
    pipeline = ObservationPipeline(args=_args(), n_envs=2, device=torch.device("cpu"))
    raw_rgb = torch.stack(
        [
            torch.full((4, 4, 3), 10, dtype=torch.uint8),
            torch.full((4, 4, 3), 30, dtype=torch.uint8),
        ],
        dim=0,
    )

    reset_obs = _to_hwc(pipeline.reset(raw_rgb, np.asarray([0, 4], dtype=np.int64)))

    assert reset_obs.shape == (2, 4, 4, 9)
    assert np.all(reset_obs[0, :, :, :9] == 10)
    assert np.all(reset_obs[1, :, :, :9] == 30)

    next_raw_rgb = torch.stack(
        [
            torch.full((4, 4, 3), 20, dtype=torch.uint8),
            torch.full((4, 4, 3), 40, dtype=torch.uint8),
        ],
        dim=0,
    )

    step_obs = _to_hwc(
        pipeline.step(
            next_raw_rgb,
            dones=np.asarray([False, True], dtype=np.bool_),
            frame_skips=np.asarray([2, 4], dtype=np.int64),
        )
    )

    assert np.all(step_obs[0, :, :, :6] == 10)
    assert np.all(step_obs[0, :, :, 6:9] == 20)
    assert np.all(step_obs[1, :, :, :9] == 40)


def test_training_pipeline_matches_shared_torch_pipeline_for_same_raw_sequence():
    args = _args()
    config = FlappyPpoObservationConfig.from_dict(vars(args))
    training_pipeline = ObservationPipeline(args=args, n_envs=2, device="cpu")
    shared_pipeline = FlappyPpoTorchObservationPipeline(config, n_envs=2, device="cpu")
    first = torch.randint(0, 256, (2, 6, 5, 3), dtype=torch.uint8)
    second = torch.randint(0, 256, (2, 6, 5, 3), dtype=torch.uint8)
    frame_skips = np.asarray([0, 4], dtype=np.int64)
    dones = np.asarray([False, True], dtype=np.bool_)

    assert torch.equal(
        training_pipeline.reset(first, frame_skips),
        shared_pipeline.reset(first, frame_skips),
    )
    assert torch.equal(
        training_pipeline.step(second, dones, frame_skips),
        shared_pipeline.step(second, dones, frame_skips),
    )


def test_sync_flappy_batch_reset_returns_full_size_raw_rgb_and_skips_env_render(monkeypatch):
    envs = SyncFlappyBatch(
        n_envs=1,
        seed=0,
        frame_skip_values=(0,),
        device="cpu",
    )
    try:
        for env in envs.envs:
            monkeypatch.setattr(
                env,
                "render",
                lambda: pytest.fail("training hot path must not call env.render()"),
            )

        raw_rgb = envs.reset()

        assert raw_rgb.dtype == torch.uint8
        assert raw_rgb.device.type == "cpu"
        assert raw_rgb.ndim == 4
        assert raw_rgb.shape[0] == 1
        assert raw_rgb.shape[-1] == 3
        assert raw_rgb.shape[1] > 84
        assert raw_rgb.shape[2] > 84
    finally:
        envs.close()
