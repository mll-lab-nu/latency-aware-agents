from __future__ import annotations

import pytest
import torch
from argparse import Namespace

from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoObservationConfig,
    load_flappy_ppo_checkpoint,
    save_flappy_ppo_unified_checkpoint,
)
from training.common.checkpoints import format_checkpoint_name
from training.flappy_ppo.ppo_loop import save_checkpoint


def test_unified_flappy_ppo_checkpoint_uses_sample_factory_style_layout(tmp_path):
    obs_config = FlappyPpoObservationConfig(
        image_size=84,
        grayscale=False,
        stack_frames=4,
    )
    agent = FlappyPpoAgent(in_channels=obs_config.in_channels, image_size=84, n_actions=2)

    path = save_flappy_ppo_unified_checkpoint(
        train_dir=tmp_path,
        experiment_name="flappy_l3_seed0",
        agent=agent,
        optimizer=None,
        observation_config=obs_config,
        train_step=5,
        env_steps=1280,
        train_config={"seed": 7},
        resolved_config={"backend": {"type": "flappy_ppo"}},
        frame_skip_training_setting="action_delay_frame_skip",
    )

    assert path == tmp_path / "flappy_l3_seed0" / "checkpoint_p0" / format_checkpoint_name(
        train_step=5,
        env_steps=1280,
    )
    checkpoint = load_flappy_ppo_checkpoint(path, map_location="cpu")
    assert checkpoint["backend_type"] == "flappy_ppo"
    assert checkpoint["format"] == "flappy_ppo"
    assert checkpoint["train_step"] == 5
    assert checkpoint["env_steps"] == 1280
    assert checkpoint["global_step"] == 1280
    assert checkpoint["metadata"]["obs_shape"] == [12, 84, 84]
    assert checkpoint["metadata"]["frame_stack"] == 4
    assert checkpoint["metadata"]["image_size"] == 84
    assert checkpoint["metadata"]["frame_skip_training_setting"] == "action_delay_frame_skip"
    assert checkpoint["observation_config"]["stack_frames"] == 4


def test_flappy_ppo_loader_rejects_sample_factory_checkpoint(tmp_path):
    path = tmp_path / "checkpoint_000000001_000000000128.pth"
    torch.save({"model": {}, "train_step": 1, "env_steps": 128}, path)

    with pytest.raises(ValueError, match="not a flappy_ppo checkpoint"):
        load_flappy_ppo_checkpoint(path, map_location="cpu")


def test_flappy_ppo_loader_rejects_wrong_backend_type(tmp_path):
    path = tmp_path / "checkpoint_000000001_000000000128.pth"
    torch.save({"backend_type": "sample_factory", "format": "flappy_ppo", "agent": {}}, path)

    with pytest.raises(ValueError, match="backend_type"):
        load_flappy_ppo_checkpoint(path, map_location="cpu")


def test_ppo_loop_save_checkpoint_uses_unified_layout_when_enabled(tmp_path):
    obs_config = FlappyPpoObservationConfig(
        image_size=84,
        grayscale=False,
        stack_frames=4,
    )
    agent = FlappyPpoAgent(in_channels=obs_config.in_channels, image_size=84, n_actions=2)
    args = Namespace(
        train_dir=str(tmp_path),
        experiment_name="flappy_l0_seed0",
        unified_checkpoint_layout=True,
        image_size=84,
        grayscale=False,
        stack_frames=4,
        frame_skip_values="0",
        frame_skip_training_setting="action_delay_frame_skip",
        resolved_config={"backend": {"type": "flappy_ppo"}},
    )

    path = save_checkpoint("ignored.pt", agent, None, args, global_step=4096, train_step=2)

    assert path == tmp_path / "flappy_l0_seed0" / "checkpoint_p0" / format_checkpoint_name(
        train_step=2,
        env_steps=4096,
    )
    checkpoint = load_flappy_ppo_checkpoint(path, map_location="cpu")
    assert checkpoint["metadata"]["frame_skip_training_setting"] == "action_delay_frame_skip"
