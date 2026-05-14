from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


def _write_checkpoint_config(path: Path, **overrides) -> Path:
    config = {
        "algo": "APPO",
        "env": "latency_demon_attack",
        "experiment": "demon_fixed_l4_seed0",
        "train_dir": str(path.parent),
        "restart_behavior": "resume",
        "device": "gpu",
        "seed": 7,
        "train_for_env_steps": 1024,
        "num_workers": 1,
        "num_envs_per_worker": 2,
        "num_policies": 1,
        "batch_size": 128,
        "rollout": 16,
        "recurrence": 1,
        "num_epochs": 1,
        "num_batches_per_epoch": 1,
        "worker_num_splits": 1,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_ratio": 0.1,
        "ppo_clip_value": 0.2,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.0,
        "async_rl": True,
        "batched_sampling": True,
        "use_rnn": False,
        "normalize_input": True,
        "normalize_returns": True,
        "stats_avg": 100,
        "experiment_summaries_interval": 1,
        "save_every_sec": 600,
        "keep_checkpoints": 5,
        "env_id": "ALE/DemonAttack-v5",
        "env_fps": 60.0,
        "obs_fps": 15.0,
        "screen_size": 84,
        "noop_max": 30,
        "frame_stack": 6,
        "terminal_on_life_loss": True,
        "clip_reward": True,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "max_episode_steps": 25000,
        "latency_type": "fixed",
        "fixed_latency_ms": 66.66666666666667,
        "mean_latency_ms": None,
        "std_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "latency_seed": None,
        "add_latency_info": False,
        "max_pending_actions": None,
        "hold_policy": "hold",
        "ordering_policy": "latest_ready",
        "eval_episodes": 20,
        "eval_parallel_envs": 1,
        "eval_max_steps": 3600,
        "eval_deterministic": True,
        "eval_raw_reward": True,
        "output_dir": "outputs/demon_fixed_l4_seed0",
    }
    config.update(overrides)
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_config_loader_generic_uses_current_public_loader(tmp_path):
    from training.config_loader import load_training_config_generic

    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("env:\n  name: flappy\nlatency:\n  type: zero\n", encoding="utf-8")
    child.write_text("__base__: base.yaml\nlatency:\n  type: fixed\n  fixed_latency_ms: 50\n", encoding="utf-8")

    config = load_training_config_generic(child)

    assert config["env"]["name"] == "flappy"
    assert config["latency"] == {"type": "fixed", "fixed_latency_ms": 50}


def test_rollout_config_defaults_to_checkpoint_training_latency(tmp_path):
    from scripts.export_sf_teacher_dataset import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)

    config, metadata = load_rollout_training_config(experiment_dir)

    assert config["env"]["name"] == "demon_attack"
    assert config["experiment"]["name"] == "demon_fixed_l4_seed0"
    assert config["backend"]["run_mode"] == "eval"
    assert config["backend"]["train_dir"] == str(tmp_path)
    assert config["policy"]["checkpoint_path"] is None
    assert config["latency"]["type"] == "fixed"
    assert config["latency"]["fixed_latency_ms"] == pytest.approx(66.66666666666667)
    assert config["latency"]["add_latency_info"] is True
    assert metadata["latency_source"] == "checkpoint"


def test_latency_config_override_replaces_only_latency_section(tmp_path):
    from scripts.export_sf_teacher_dataset import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)
    override_path = tmp_path / "latency_override.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "latency": {
                    "type": "uniform_distribution",
                    "min_latency_ms": 10.0,
                    "max_latency_ms": 80.0,
                    "seed": 123,
                }
            }
        ),
        encoding="utf-8",
    )

    config, metadata = load_rollout_training_config(experiment_dir, latency_config_path=override_path)

    assert config["env"]["frame_stack"] == 6
    assert config["latency"] == {
        "type": "uniform_distribution",
        "min_latency_ms": 10.0,
        "max_latency_ms": 80.0,
        "seed": 123,
        "add_latency_info": True,
    }
    assert metadata["latency_source"] == str(override_path.resolve())


def test_device_override_cpu_routes_flappy_gpu_renderer_to_cpu(tmp_path):
    from scripts.export_sf_teacher_dataset import load_rollout_training_config

    experiment_dir = tmp_path / "flappy_normal_seed0"
    _write_checkpoint_config(
        experiment_dir,
        env="latency_flappy",
        experiment="flappy_normal_seed0",
        gym_id="FlappyBird-v0",
        obs_resize="84,84",
        use_gpu_render=True,
        gpu_render_device="auto",
        gpu_render_batch_size=64,
        frame_stack=4,
        latency_type="normal_distribution",
        fixed_latency_ms=None,
        mean_latency_ms=66.7,
        std_latency_ms=33.3,
        min_latency_ms=0.0,
        max_latency_ms=500.0,
        hold_policy="one_frame_then_noop",
    )

    config, _ = load_rollout_training_config(experiment_dir, device_override="cpu")

    assert config["backend"]["device"] == "cpu"
    assert config["env"]["gpu_render_device"] == "cpu"


def test_observation_frame_uses_latest_stacked_policy_observation_without_render():
    from scripts.export_sf_teacher_dataset import frame_from_policy_observation

    observation = np.zeros((12, 2, 3), dtype=np.uint8)
    observation[9:12] = np.array(
        [
            [[11, 12, 13], [14, 15, 16]],
            [[21, 22, 23], [24, 25, 26]],
            [[31, 32, 33], [34, 35, 36]],
        ],
        dtype=np.uint8,
    )

    frame = frame_from_policy_observation(observation, frame_stack=4)

    assert frame.shape == (2, 3, 3)
    assert frame[0, 0].tolist() == [11, 21, 31]
    assert frame[1, 2].tolist() == [16, 26, 36]


def test_observation_frame_drops_single_batched_observation_axis():
    from scripts.export_sf_teacher_dataset import frame_from_policy_observation

    observation = np.zeros((1, 12, 2, 3), dtype=np.uint8)
    observation[0, 9:12] = np.array(
        [
            [[11, 12, 13], [14, 15, 16]],
            [[21, 22, 23], [24, 25, 26]],
            [[31, 32, 33], [34, 35, 36]],
        ],
        dtype=np.uint8,
    )

    frame = frame_from_policy_observation({"obs": observation}, frame_stack=4)

    assert frame.shape == (2, 3, 3)
    assert frame[0, 0].tolist() == [11, 21, 31]


def test_batch_observations_keeps_already_batched_mapping_observation(monkeypatch, tmp_path):
    import torch
    from scripts import export_sf_teacher_dataset as exporter

    observation = {"obs": np.zeros((1, 12, 84, 84), dtype=np.uint8)}

    batch = exporter._batch_observations([observation])

    assert batch["obs"].shape == (1, 12, 84, 84)

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 99, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        def __init__(self):
            self.action_space = object()
            self.step_actions: list[object] = []

        def reset(self, seed=None):
            del seed
            return np.zeros((3, 2, 2), dtype=np.uint8), {}

        def step(self, action):
            self.step_actions.append(action)
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(1.5),
                np.bool_(True),
                np.bool_(False),
                {"latency_raw_frames": 2, "latency_ms": 66.0},
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=1, env_fps=30.0, seed=11)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "_build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [fake_env],
            "env_info": object(),
            "actor_critic": fake_actor,
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.asarray([99], dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        },
    )
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [7])
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
        output_dir=tmp_path / "export",
        prompt="pick action",
        prompt_template="{prompt}",
        max_episodes=1,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert not val_rows
    assert fake_env.step_actions == [7]
    assert train_rows[0]["action_id"] == 7
    assert train_rows[0]["reward"] == 1.5
    assert train_rows[0]["done"] is True
    assert metadata["checkpoint_train_step"] == 123


def test_exporter_parser_has_no_render_image_source_switch():
    from scripts.export_sf_teacher_dataset import build_parser

    option_dests = {action.dest for action in build_parser()._actions}

    assert "image_source" not in option_dests


def test_push_token_prefers_explicit_then_environment(monkeypatch):
    from scripts.push_sf_teacher_dataset import resolve_token

    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    assert resolve_token(" explicit-token ") == "explicit-token"
    assert resolve_token("") == "env-token"


def test_push_token_falls_back_to_huggingface_hub_token(monkeypatch):
    from scripts.push_sf_teacher_dataset import resolve_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    assert resolve_token("") == "hub-token"
