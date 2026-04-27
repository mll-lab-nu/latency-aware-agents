from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoEvalObservationPipeline,
    FlappyPpoObservationConfig,
    build_agent_from_checkpoint,
    latency_channel_value,
    load_flappy_ppo_checkpoint,
    observation_config_from_checkpoint,
    reset_frame_stack,
    resize_frame_pil,
    save_flappy_ppo_checkpoint,
    step_frame_stack,
)


def test_flappy_ppo_agent_forward_accepts_uint8_and_scales_internally():
    config = FlappyPpoObservationConfig(obs_mode="image_only", frame_skip_values=(0,))
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
    assert FlappyPpoObservationConfig(
        grayscale=False,
        stack_frames=4,
        obs_mode="image_only",
        frame_skip_values=(0,),
    ).in_channels == 12
    assert FlappyPpoObservationConfig(
        grayscale=False,
        stack_frames=4,
        obs_mode="latency_channel",
        frame_skip_values=(0, 5),
    ).in_channels == 13
    assert FlappyPpoObservationConfig(
        grayscale=True,
        stack_frames=4,
        obs_mode="latency_channel",
        frame_skip_values=(0, 5),
    ).in_channels == 5


def test_checkpoint_save_load_restores_agent_optimizer_config_and_global_step(tmp_path: Path):
    config = FlappyPpoObservationConfig(obs_mode="latency_channel", frame_skip_values=(0, 5))
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
    assert loaded_config.to_dict() == config.to_dict()
    for key, tensor in agent.state_dict().items():
        assert torch.equal(tensor, loaded_agent.state_dict()[key])


def test_legacy_checkpoint_observation_config_can_be_inferred_from_agent_channels():
    config = FlappyPpoObservationConfig(obs_mode="latency_channel", frame_skip_values=(0, 5))
    agent = FlappyPpoAgent(config.in_channels, config.image_size, n_actions=2)

    loaded_config = observation_config_from_checkpoint({"agent": agent.state_dict()})

    assert loaded_config.in_channels == config.in_channels
    assert loaded_config.stack_frames == 4
    assert loaded_config.obs_mode == "latency_channel"


def test_frame_stack_and_latency_channel_match_training_formula():
    frame_a = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 20, dtype=np.uint8)

    stack = reset_frame_stack(frame_a, stack_frames=4)
    stepped = step_frame_stack(stack, frame_b, base_channels=3)

    assert stack.shape == (2, 2, 12)
    assert np.all(stack[:, :, :3] == 10)
    assert np.all(stepped[:, :, :9] == 10)
    assert np.all(stepped[:, :, 9:12] == 20)
    assert latency_channel_value(frame_skip=3, max_frame_skip=5) == np.uint8(np.rint((3 / 5) * 255))


def test_eval_pipeline_resets_stack_on_episode_start_and_returns_nchw_tensor():
    config = FlappyPpoObservationConfig(
        image_size=4,
        grayscale=False,
        stack_frames=3,
        obs_mode="latency_channel",
        frame_skip_values=(0, 4),
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
    assert np.all(obs[:, :, -1] == 255)
    assert np.all(reset_obs[:, :, :9] == 99)
    assert np.all(reset_obs[:, :, -1] == np.uint8(np.rint(0.5 * 255)))
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
