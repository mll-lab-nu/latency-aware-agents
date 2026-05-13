from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from modeling.demon_attack_ppo import (
    DemonAttackPpoAgent,
    DemonAttackPpoObservationConfig,
    DemonAttackPpoTorchObservationPipeline,
    build_agent_from_checkpoint,
    latency_channel_value,
    load_demon_attack_ppo_checkpoint,
    observation_config_from_checkpoint,
    save_demon_attack_ppo_checkpoint,
)


def test_demon_attack_ppo_agent_forward_accepts_uint8_and_dynamic_action_count():
    config = DemonAttackPpoObservationConfig(obs_mode="image_only", frame_skip_values=(0,))
    agent = DemonAttackPpoAgent(config.in_channels, config.image_size, n_actions=6)
    obs = torch.full(
        (2, config.in_channels, config.image_size, config.image_size),
        255,
        dtype=torch.uint8,
    )

    action, logprob, entropy, value = agent.get_action_and_value(obs)
    logits = agent.get_logits(obs)

    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)
    assert logits.shape == (2, 6)


def test_demon_attack_observation_config_channel_counts():
    assert DemonAttackPpoObservationConfig(
        base_channels=1,
        stack_frames=4,
        obs_mode="image_only",
        frame_skip_values=(0,),
    ).in_channels == 4
    assert DemonAttackPpoObservationConfig(
        base_channels=1,
        stack_frames=4,
        obs_mode="latency_channel",
        frame_skip_values=(0, 5),
    ).in_channels == 5
    assert DemonAttackPpoObservationConfig(
        base_channels=3,
        stack_frames=2,
        obs_mode="latency_channel",
        frame_skip_values=(0, 3),
    ).in_channels == 7


def test_torch_pipeline_accepts_nhw_and_resets_done_stacks_with_latency_channel():
    config = DemonAttackPpoObservationConfig(
        image_size=4,
        base_channels=1,
        stack_frames=3,
        obs_mode="latency_channel",
        frame_skip_values=(0, 4),
    )
    pipeline = DemonAttackPpoTorchObservationPipeline(config, n_envs=2, device="cpu")
    frame_a = torch.stack(
        [
            torch.full((4, 4), 11, dtype=torch.uint8),
            torch.full((4, 4), 33, dtype=torch.uint8),
        ],
        dim=0,
    )
    frame_b = torch.stack(
        [
            torch.full((4, 4), 22, dtype=torch.uint8),
            torch.full((4, 4), 99, dtype=torch.uint8),
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
    assert np.all(hwc[0, :, :, :2] == 11)
    assert np.all(hwc[0, :, :, 2:3] == 22)
    assert np.all(hwc[0, :, :, -1] == latency_channel_value(2, 4))
    assert np.all(hwc[1, :, :, :3] == 99)
    assert np.all(hwc[1, :, :, -1] == latency_channel_value(4, 4))


def test_torch_pipeline_accepts_nhwc_input():
    config = DemonAttackPpoObservationConfig(
        image_size=4,
        base_channels=1,
        stack_frames=1,
        obs_mode="image_only",
        frame_skip_values=(0,),
    )
    pipeline = DemonAttackPpoTorchObservationPipeline(config, n_envs=1, device="cpu")
    frame = torch.full((1, 4, 4, 1), 77, dtype=torch.uint8)

    obs = pipeline.reset(frame)

    assert obs.shape == (1, 1, 4, 4)
    assert torch.all(obs == 77)


def test_checkpoint_save_load_restores_config_and_infers_action_count(tmp_path: Path):
    config = DemonAttackPpoObservationConfig(obs_mode="latency_channel", frame_skip_values=(0, 5))
    agent = DemonAttackPpoAgent(config.in_channels, config.image_size, n_actions=9)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    path = tmp_path / "checkpoint.pt"

    save_demon_attack_ppo_checkpoint(
        path=path,
        agent=agent,
        optimizer=optimizer,
        observation_config=config,
        global_step=42,
        train_config={"seed": 7},
    )

    checkpoint = load_demon_attack_ppo_checkpoint(path, map_location="cpu")
    loaded_agent = build_agent_from_checkpoint(checkpoint, device="cpu")
    loaded_config = observation_config_from_checkpoint(checkpoint)

    assert checkpoint["format"] == "demon_attack_ppo"
    assert checkpoint["global_step"] == 42
    assert "optimizer" in checkpoint
    assert checkpoint["train_config"] == {"seed": 7}
    assert loaded_config.to_dict() == config.to_dict()
    assert loaded_agent.actor.out_features == 9
    for key, tensor in agent.state_dict().items():
        assert torch.equal(tensor, loaded_agent.state_dict()[key])
