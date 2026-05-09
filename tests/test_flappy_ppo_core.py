from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoEvalObservationPipeline,
    FlappyPpoObservationConfig,
    FlappyPpoTorchObservationPipeline,
    build_agent_from_checkpoint,
    load_flappy_ppo_checkpoint,
    observation_config_from_checkpoint,
    reset_frame_stack,
    resize_frame_pil,
    resize_rgb_batch_torch,
    save_flappy_ppo_checkpoint,
    step_frame_stack,
)


def test_flappy_ppo_agent_forward_accepts_uint8_and_scales_internally():
    config = FlappyPpoObservationConfig()
    agent = FlappyPpoAgent(config.in_channels, config.image_size, n_actions=2)
    obs = torch.full(
        (2, config.in_channels, config.image_size, config.image_size),
        255,
        dtype=torch.uint8,
    )

    action, logprob, entropy, value = agent.get_action_and_value(obs)
    logits = agent.get_logits(obs)

    assert obs.dtype == torch.uint8
    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)
    assert logits.shape == (2, 2)


def test_flappy_ppo_observation_config_channel_counts():
    assert FlappyPpoObservationConfig(grayscale=False, stack_frames=4).in_channels == 12
    assert FlappyPpoObservationConfig(grayscale=True, stack_frames=4).in_channels == 4


def test_checkpoint_save_load_restores_agent_optimizer_config_and_global_step(tmp_path: Path):
    config = FlappyPpoObservationConfig()
    agent = FlappyPpoAgent(config.in_channels, config.image_size, n_actions=2)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    path = tmp_path / "checkpoint.pt"

    save_flappy_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=optimizer,
        observation_config=config,
        global_step=42,
        train_config={"seed": 7},
    )

    checkpoint = load_flappy_ppo_checkpoint(path, map_location="cpu")
    loaded_agent = build_agent_from_checkpoint(checkpoint, device="cpu")
    loaded_config = observation_config_from_checkpoint(checkpoint)

    assert checkpoint["format"] == "flappy_ppo"
    assert checkpoint["global_step"] == 42
    assert "optimizer" in checkpoint
    assert checkpoint["train_config"] == {"seed": 7}
    assert checkpoint["observation_config"]["observation_protocol"] == "raw_rgb_v1"
    assert checkpoint["observation_config"]["preprocess_backend"] == "torch"
    assert loaded_config.to_dict() == config.to_dict()
    for key, tensor in agent.state_dict().items():
        assert torch.equal(tensor, loaded_agent.state_dict()[key])


def test_legacy_checkpoint_observation_config_can_be_inferred_from_agent_channels():
    config = FlappyPpoObservationConfig()
    agent = FlappyPpoAgent(config.in_channels, config.image_size, n_actions=2)

    loaded_config = observation_config_from_checkpoint({"agent": agent.state_dict()})

    assert loaded_config.in_channels == config.in_channels
    assert loaded_config.stack_frames == 4
    assert loaded_config.observation_protocol == "raw_rgb_v1"
    assert loaded_config.preprocess_backend == "torch"


def test_frame_stack_matches_training_formula():
    frame_a = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 20, dtype=np.uint8)

    stack = reset_frame_stack(frame_a, stack_frames=4)
    stepped = step_frame_stack(stack, frame_b, base_channels=3)

    assert stack.shape == (2, 2, 12)
    assert np.all(stack[:, :, :3] == 10)
    assert np.all(stepped[:, :, :9] == 10)
    assert np.all(stepped[:, :, 9:12] == 20)


def test_resize_rgb_batch_torch_contract_and_grayscale_formula():
    raw_rgb = torch.tensor(
        [
            [
                [[10, 20, 30], [100, 110, 120]],
                [[200, 210, 220], [30, 40, 50]],
            ]
        ],
        dtype=torch.uint8,
    )

    rgb = resize_rgb_batch_torch(raw_rgb, image_size=2, grayscale=False)
    gray = resize_rgb_batch_torch(raw_rgb, image_size=2, grayscale=True)

    assert rgb.shape == (1, 2, 2, 3)
    assert rgb.dtype == torch.uint8
    assert torch.equal(rgb, raw_rgb)
    expected_gray = torch.floor(
        (
            raw_rgb[..., 0:1].float() * 299.0
            + raw_rgb[..., 1:2].float() * 587.0
            + raw_rgb[..., 2:3].float() * 114.0
            + 500.0
        )
        / 1000.0
    ).to(torch.uint8)
    assert gray.shape == (1, 2, 2, 1)
    assert torch.equal(gray, expected_gray)


def test_torch_observation_pipeline_reset_step_and_done_reset():
    config = FlappyPpoObservationConfig(
        image_size=4,
        grayscale=False,
        stack_frames=3,
    )
    pipeline = FlappyPpoTorchObservationPipeline(config, n_envs=2, device="cpu")
    frame_a = torch.stack(
        [
            torch.full((4, 4, 3), 11, dtype=torch.uint8),
            torch.full((4, 4, 3), 33, dtype=torch.uint8),
        ],
        dim=0,
    )
    frame_b = torch.stack(
        [
            torch.full((4, 4, 3), 22, dtype=torch.uint8),
            torch.full((4, 4, 3), 99, dtype=torch.uint8),
        ],
        dim=0,
    )

    pipeline.reset(frame_a, frame_skips=np.asarray([0, 4], dtype=np.int64))
    obs = pipeline.step(
        frame_b,
        dones=np.asarray([False, True], dtype=np.bool_),
        frame_skips=np.asarray([2, 4], dtype=np.int64),
    )
    hwc = obs.permute(0, 2, 3, 1).cpu().numpy()

    assert obs.shape == (2, config.in_channels, 4, 4)
    assert obs.dtype == torch.uint8
    assert np.all(hwc[0, :, :, :6] == 11)
    assert np.all(hwc[0, :, :, 6:9] == 22)
    assert np.all(hwc[1, :, :, :9] == 99)


def test_eval_pipeline_resets_stack_on_episode_start_and_returns_nchw_tensor():
    config = FlappyPpoObservationConfig(
        image_size=4,
        grayscale=False,
        stack_frames=3,
    )
    pipeline = FlappyPpoEvalObservationPipeline(config)
    frame_a = np.full((4, 4, 3), 11, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 22, dtype=np.uint8)
    frame_c = np.full((4, 4, 3), 99, dtype=np.uint8)

    pipeline.build_observation(frame_a, env_step=0, frame_skip=0)
    obs = pipeline.build_observation(frame_b, env_step=1, frame_skip=4)
    reset_obs = pipeline.build_observation(frame_c, env_step=0, frame_skip=2)
    tensor = pipeline.build_tensor(frame_c, env_step=1, frame_skip=2, device="cpu")

    assert np.all(obs[:, :, :6] == 11)
    assert np.all(obs[:, :, 6:9] == 22)
    assert np.all(reset_obs[:, :, :9] == 99)
    assert tensor.shape == (1, config.in_channels, config.image_size, config.image_size)
    assert tensor.dtype == torch.uint8


def test_resize_frame_pil_matches_existing_oracle():
    flappy_root = Path(__file__).resolve().parents[1] / "flappy-bird-gymnasium"
    if str(flappy_root) not in sys.path:
        sys.path.insert(0, str(flappy_root))
    from flappy_bird_gymnasium.gpu_render.oracle import resize_frame_pil as oracle_resize_frame_pil

    frame = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)

    assert np.array_equal(
        resize_frame_pil(frame, image_size=8, grayscale=False),
        oracle_resize_frame_pil(frame, image_size=8, grayscale=False),
    )
    assert np.array_equal(
        resize_frame_pil(frame, image_size=8, grayscale=True),
        oracle_resize_frame_pil(frame, image_size=8, grayscale=True),
    )
