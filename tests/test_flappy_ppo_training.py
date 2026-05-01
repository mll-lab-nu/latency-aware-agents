from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from modeling.flappy_ppo import latency_channel_value
from training.flappy_ppo.observation_pipeline import ObservationPipeline


class FakeFlappyBatch:
    def __init__(self, frames: list[np.ndarray], frame_skips: list[int]):
        self.frames = frames
        self._frame_skips = np.asarray(frame_skips, dtype=np.int64)

    @property
    def n_envs(self) -> int:
        return len(self.frames)

    def render_frames(self) -> list[np.ndarray]:
        return self.frames

    def frame_skips(self) -> np.ndarray:
        return self._frame_skips


def _args() -> Namespace:
    return Namespace(
        image_size=4,
        grayscale=False,
        stack_frames=3,
        obs_mode="latency_channel",
        frame_skip_values="0,4",
        renderer="cpu_oracle",
        resize_backend="pil",
        observation_render_mode="full_then_resize",
    )


def _to_hwc(obs: torch.Tensor) -> np.ndarray:
    return obs.permute(0, 2, 3, 1).cpu().numpy()


def _raw_rgb(envs: FakeFlappyBatch) -> torch.Tensor:
    return torch.as_tensor(np.stack(envs.render_frames()), dtype=torch.uint8)


def test_training_observation_pipeline_stack_done_reset_and_latency_channel():
    envs = FakeFlappyBatch(
        frames=[
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ],
        frame_skips=[0, 4],
    )
    pipeline = ObservationPipeline(args=_args(), envs=envs, device=torch.device("cpu"))

    reset_obs = _to_hwc(pipeline.reset(_raw_rgb(envs), envs.frame_skips()))

    assert reset_obs.shape == (2, 4, 4, 10)
    assert np.all(reset_obs[0, :, :, :9] == 10)
    assert np.all(reset_obs[1, :, :, :9] == 30)
    assert np.all(reset_obs[0, :, :, -1] == latency_channel_value(0, 4))
    assert np.all(reset_obs[1, :, :, -1] == latency_channel_value(4, 4))

    envs.frames = [
        np.full((4, 4, 3), 20, dtype=np.uint8),
        np.full((4, 4, 3), 40, dtype=np.uint8),
    ]
    envs._frame_skips = np.asarray([2, 4], dtype=np.int64)

    step_obs = _to_hwc(
        pipeline.step(
            _raw_rgb(envs),
            dones=np.asarray([False, True], dtype=np.bool_),
            frame_skips=envs.frame_skips(),
        )
    )

    assert np.all(step_obs[0, :, :, :6] == 10)
    assert np.all(step_obs[0, :, :, 6:9] == 20)
    assert np.all(step_obs[0, :, :, -1] == latency_channel_value(2, 4))
    assert np.all(step_obs[1, :, :, :9] == 40)
    assert np.all(step_obs[1, :, :, -1] == latency_channel_value(4, 4))
