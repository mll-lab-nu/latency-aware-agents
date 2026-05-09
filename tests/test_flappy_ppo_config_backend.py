from __future__ import annotations

import pytest

from training.flappy_ppo.config_backend import (
    build_flappy_ppo_args,
    build_observation_config,
    run_from_config,
    validate_flappy_ppo_config,
)


def _config() -> dict:
    return {
        "experiment": {"name": "flappy_l3_seed0", "seed": 7},
        "backend": {
            "type": "flappy_ppo",
            "train_dir": "results/checkpoints/training_runs",
            "device": "cpu",
        },
        "env": {"name": "flappy", "frame_stack": 4},
        "latency": {
            "train_values": [0, 3],
            "train_mode": "episode_fixed",
            "train_sampling_mode": "uniform_all",
            "randomize_within_stage": False,
            "add_latency_info": False,
        },
        "training": {
            "train_for_env_steps": 300,
            "num_workers": 2,
            "num_envs_per_worker": 3,
            "rollout": 5,
            "batch_size": 10,
            "num_epochs": 4,
            "learning_rate": 0.00025,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5,
        },
        "flappy_ppo": {
            "image_size": 84,
            "grayscale": False,
            "checkpoint_freq": 123,
        },
        "logging": {
            "wandb_project": "latency-test",
            "wandb_name": "flappy-bird-l3-s0",
            "wandb_tags": ["flappy", "l3"],
        },
    }


def test_flappy_ppo_config_maps_supported_fields_to_args():
    args = build_flappy_ppo_args(_config())

    assert args.train_steps == 300
    assert args.n_envs == 6
    assert args.n_steps == 5
    assert args.batch_size == 10
    assert args.n_epochs == 4
    assert args.seed == 7
    assert args.device == "cpu"
    assert args.frame_skip_values == "0,3"
    assert args.frame_skip_training_setting == "action_delay_frame_skip"
    assert args.image_size == 84
    assert args.stack_frames == 4
    assert args.checkpoint_freq == 123
    assert args.wandb_project == "latency-test"
    assert args.wandb_run_name == "flappy-bird-l3-s0"
    assert args.wandb_tags == "flappy,l3"


def test_flappy_ppo_observation_config_is_image_only():
    obs_config = build_observation_config(_config())

    assert obs_config.image_size == 84
    assert obs_config.grayscale is False
    assert obs_config.stack_frames == 4
    assert obs_config.in_channels == 12


def test_flappy_ppo_config_rejects_sample_factory_only_training_fields():
    cfg = _config()
    cfg["training"]["async_rl"] = False

    with pytest.raises(ValueError, match="training.async_rl"):
        validate_flappy_ppo_config(cfg)


def test_flappy_ppo_config_rejects_env_screen_size():
    cfg = _config()
    cfg["env"]["screen_size"] = [84, 84]

    with pytest.raises(ValueError, match="env.screen_size"):
        validate_flappy_ppo_config(cfg)


def test_flappy_ppo_config_rejects_batch_size_larger_than_rollout_batch():
    cfg = _config()
    cfg["training"]["batch_size"] = 31

    with pytest.raises(ValueError, match="training.batch_size"):
        validate_flappy_ppo_config(cfg)


def test_flappy_ppo_config_rejects_unsupported_latency_schedule_fields():
    cfg = _config()
    cfg["latency"]["within_episode_change_prob"] = 0.5

    with pytest.raises(ValueError, match="latency.within_episode_change_prob"):
        validate_flappy_ppo_config(cfg)


def test_flappy_ppo_config_rejects_ignored_logging_fields():
    cfg = _config()
    cfg["logging"]["wandb_group"] = "would-be-ignored"

    with pytest.raises(ValueError, match="logging.wandb_group"):
        validate_flappy_ppo_config(cfg)


def test_flappy_ppo_run_from_config_writes_artifacts_and_delegates_train(monkeypatch, tmp_path):
    cfg = _config()
    cfg["backend"]["train_dir"] = str(tmp_path)
    calls = []

    class _FakeWandb:
        def finish(self):
            calls.append("finish")

    import training.flappy_ppo.logging as flappy_logging
    import training.flappy_ppo.ppo_loop as ppo_loop

    monkeypatch.setattr(flappy_logging, "setup_wandb", lambda args: _FakeWandb())
    monkeypatch.setattr(
        ppo_loop,
        "train",
        lambda args, wandb_module: calls.append(args) or "checkpoint",
    )

    assert run_from_config(cfg) == 0

    args = calls[0]
    exp_dir = tmp_path / "flappy_l3_seed0"
    assert args.unified_checkpoint_layout is True
    assert args.experiment_dir == str(exp_dir)
    assert (exp_dir / "config.json").is_file()
    assert (exp_dir / "resolved_config.json").is_file()
    assert (exp_dir / "metadata.json").is_file()
    assert calls[-1] == "finish"
