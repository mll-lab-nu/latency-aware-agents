from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
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


def _env_raw_rgb_frame_stack(value: int = 5, *, frames: int = 1, height: int = 2, width: int = 2) -> np.ndarray:
    return np.stack(
        [np.full((height, width, 3), int(value) + idx, dtype=np.uint8) for idx in range(int(frames))],
        axis=0,
    )


def test_config_loader_generic_uses_current_public_loader(tmp_path):
    from training.config_loader import load_training_config_generic

    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("env:\n  name: flappy\nlatency:\n  type: zero\n", encoding="utf-8")
    child.write_text("__base__: base.yaml\nlatency:\n  type: fixed\n  fixed_latency_ms: 50\n", encoding="utf-8")

    config = load_training_config_generic(child)

    assert config["env"]["name"] == "flappy"
    assert config["latency"] == {"type": "fixed", "fixed_latency_ms": 50}


def test_rollout_config_reads_checkpoint_config_without_rebuilding_nested_training_config(tmp_path):
    from latency_bench.data.sf_teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir, simulator="gpu")

    descriptor, metadata = load_rollout_training_config(experiment_dir)

    assert descriptor["env_name"] == "demon_attack"
    assert descriptor["checkpoint_experiment_dir"] == str(experiment_dir.resolve())
    assert descriptor["checkpoint_path"] is None
    assert descriptor["flat_cfg"]["simulator"] == "gpu"
    assert descriptor["device_override"] is None
    assert descriptor["latency_override"] is None
    assert metadata["latency_source"] == "checkpoint"
    assert metadata["checkpoint_config"]["simulator"] == "gpu"


def test_latency_config_override_is_explicit_rollout_override(tmp_path):
    from latency_bench.data.sf_teacher_config import load_rollout_training_config

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

    descriptor, metadata = load_rollout_training_config(experiment_dir, latency_config_path=override_path)

    assert descriptor["flat_cfg"]["frame_stack"] == 6
    assert descriptor["latency_override"] == {
        "type": "uniform_distribution",
        "min_latency_ms": 10.0,
        "max_latency_ms": 80.0,
        "seed": 123,
    }
    assert metadata["latency_source"] == str(override_path.resolve())


def test_device_override_is_recorded_without_rewriting_checkpoint_env_config(tmp_path):
    from latency_bench.data.sf_teacher_config import load_rollout_training_config

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

    descriptor, _ = load_rollout_training_config(experiment_dir, device_override="cpu")

    assert descriptor["device_override"] == "cpu"
    assert descriptor["flat_cfg"]["gpu_render_device"] == "auto"


def test_device_override_cuda_uses_cuda_public_device_name(tmp_path):
    from latency_bench.data.sf_teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)

    descriptor, _ = load_rollout_training_config(experiment_dir, device_override="cuda")

    assert descriptor["device_override"] == "gpu"


def test_dataset_image_uses_latest_env_raw_rgb_frame():
    from latency_bench.data.rollout_image_io import frame_from_env_raw_rgb_info

    raw_stack = np.stack(
        [
            np.full((4, 5, 3), [11, 12, 13], dtype=np.uint8),
            np.full((4, 5, 3), [21, 22, 23], dtype=np.uint8),
        ],
        axis=0,
    )

    frame = frame_from_env_raw_rgb_info({"env_raw_rgb_frame_stack": raw_stack})

    assert frame.shape == (4, 5, 3)
    assert frame[0, 0].tolist() == [21, 22, 23]



def test_teacher_prompts_cover_rollout_envs():
    from latency_bench.data.prompts import BASE_PROMPTS

    assert set(BASE_PROMPTS) == {"flappy", "demon_attack", "deadly_corridor"}


def test_batch_observations_keeps_already_batched_mapping_observation():
    from latency_bench.data import sf_teacher_drivers

    observation = {"obs": np.zeros((1, 12, 84, 84), dtype=np.uint8)}

    batch = sf_teacher_drivers.batch_observations([observation])

    assert batch["obs"].shape == (1, 12, 84, 84)


def test_batch_observations_concatenates_single_agent_batched_mapping_observations():
    from latency_bench.data import sf_teacher_drivers

    observations = [
        {"obs": np.full((1, 4, 2, 2), slot, dtype=np.uint8)}
        for slot in range(2)
    ]

    batch = sf_teacher_drivers.batch_observations(observations)

    assert batch["obs"].shape == (2, 4, 2, 2)
    assert batch["obs"][0].tolist() == observations[0]["obs"][0].tolist()
    assert batch["obs"][1].tolist() == observations[1]["obs"][0].tolist()


def test_collect_teacher_dataset_handles_single_env_rollout(monkeypatch, tmp_path):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

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
            return np.zeros((3, 2, 2), dtype=np.uint8), {
                "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5)
            }

        def step(self, action):
            self.step_actions.append(action)
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(1.5),
                np.bool_(True),
                np.bool_(False),
                {
                    "latency_raw_frames": 2,
                    "latency_ms": 66.0,
                    "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6),
                },
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=1, env_fps=30.0, obs_fps=15.0, seed=11)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=tmp_path / "export",
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
    assert fake_env.step_actions == [[1]]
    assert train_rows[0]["action_id"] == 1
    assert Path(train_rows[0]["image"]).exists()
    assert "image_stack" not in train_rows[0]
    assert train_rows[0]["env_name"] == "flappy"
    assert train_rows[0]["decision_step"] == 0
    assert train_rows[0]["raw_reward"] == 1.5
    assert metadata["checkpoint_train_step"] == 123
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2


def test_collect_teacher_dataset_online_filter_retries_until_episode_is_accepted(monkeypatch, tmp_path):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            del normalized_obs
            return {
                "actions": torch.tensor([[99]], dtype=torch.long),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

        def action_distribution(self):
            return object()

    class FakeEnv:
        def __init__(self):
            self.action_space = object()
            self.reset_seeds: list[int] = []

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            return np.zeros((3, 2, 2), dtype=np.uint8), {
                "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5)
            }

        def step(self, action):
            del action
            reward = 0.0 if len(self.reset_seeds) == 1 else 2200.0
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(reward),
                np.bool_(True),
                np.bool_(False),
                {
                    "latency_raw_frames": 2,
                    "latency_ms": 66.0,
                    "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6),
                },
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=1, env_fps=35.0, obs_fps=17.5, seed=11)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [fake_env],
            "env_info": object(),
            "actor_critic": fake_actor,
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.asarray([[0, 1, 0, 0]], dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        },
    )
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(
        exporter,
        "_split_batch_actions",
        lambda processed_actions, batch_size, action_space=None: [[0, 1, 0, 0]] * batch_size,
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "deadly_corridor"},
        output_dir=tmp_path / "export",
        max_episodes=1,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        filter_preset="clean_v1",
    )

    assert not val_rows
    assert fake_env.reset_seeds == [11, 12]
    assert train_rows[0]["episode_idx"] == 0
    assert train_rows[0]["seed"] == 12
    assert train_rows[0]["raw_reward"] == 2200.0
    assert Path(train_rows[0]["image"]).exists()
    assert "_attempts" not in train_rows[0]["image"]
    assert metadata["filter_report"]["attempted_episodes"] == 2
    assert metadata["filter_report"]["accepted_episodes"] == 1
    assert metadata["filter_report"]["rejected_episodes"] == 1


def test_collect_teacher_dataset_handles_single_agent_batched_step_outputs(monkeypatch, tmp_path):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeActorCritic:
        def __init__(self):
            self.seen_obs_shapes: list[tuple[int, ...]] = []

        def __call__(self, normalized_obs, rnn_states):
            self.seen_obs_shapes.append(tuple(normalized_obs["obs"].shape))
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 99, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        def __init__(self, slot: int):
            self.slot = int(slot)
            self.action_space = object()
            self.step_actions: list[object] = []

        def reset(self, seed=None):
            del seed
            return {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)}, [
                {"reset": True, "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(self.slot, frames=4)}
            ]

        def step(self, action):
            self.step_actions.append(action)
            return (
                {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(self.slot + 10, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    fake_envs = [FakeEnv(slot) for slot in range(2)]
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=11, use_gpu_render=False)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": fake_envs,
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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=tmp_path / "export",
        max_episodes=2,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=2,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert not val_rows
    assert fake_actor.seen_obs_shapes == [(2, 12, 84, 84)]
    assert [env.step_actions for env in fake_envs] == [[[1]], [[1]]]
    assert [row["action_id"] for row in train_rows] == [1, 1]
    assert Path(train_rows[0]["image"]).exists()
    assert "image_stack" not in train_rows[0]
    assert train_rows[0]["raw_reward"] == 1.5
    assert train_rows[0]["latency_raw_frames"] == 2
    assert metadata["checkpoint_train_step"] == 123
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2


def test_collect_teacher_dataset_encodes_deadly_corridor_factorized_tuple_actions(monkeypatch, tmp_path):
    import torch
    from gymnasium.spaces import Discrete, Tuple
    from latency_bench.data import sf_teacher_rollout as exporter

    component_actions = torch.tensor(
        [
            [0, 1, 2, 1],
            [1, 2, 0, 0],
        ],
        dtype=torch.long,
    )

    class FakeActorCritic:
        def __init__(self):
            self.last_component_actions = torch.empty((0, 4), dtype=torch.long)

        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            self.last_component_actions = component_actions[:batch_size].to(rnn_states.device)
            return {
                "actions": torch.full((batch_size, 4), 99, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

        def action_distribution(self):
            return SimpleNamespace(
                argmax_actions=self.last_component_actions.reshape(1, -1),
                distributions=[
                    SimpleNamespace(argmax_actions=self.last_component_actions[:, component_index])
                    for component_index in range(self.last_component_actions.shape[1])
                ],
            )

    class FakeEnv:
        action_space = Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2)))

        def __init__(self, slot: int):
            self.slot = int(slot)
            self.step_actions: list[object] = []

        def reset(self, seed=None):
            del seed
            return {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)}, [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(self.slot, frames=4)}
            ]

        def step(self, action):
            self.step_actions.append(action)
            return (
                {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)},
                np.asarray([1.0], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 33.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(self.slot + 10, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    fake_envs = [FakeEnv(slot) for slot in range(2)]
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=4, env_fps=60.0, obs_fps=15.0, seed=1)
    fake_env_info = SimpleNamespace(action_space=fake_envs[0].action_space)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": fake_envs,
            "env_info": fake_env_info,
            "actor_critic": fake_actor,
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda distribution: distribution.argmax_actions,
            "preprocess_actions": lambda env_info, actions: actions.detach().cpu().numpy(),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        },
    )
    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "deadly_corridor"},
        output_dir=tmp_path / "export",
        max_episodes=2,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=2,
        deterministic=True,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert not val_rows
    assert [env.step_actions for env in fake_envs] == [[[[0, 1, 2, 1]]], [[[1, 2, 0, 0]]]]
    assert [row["action"] for row in train_rows] == [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ]
    assert [row["action_tuple"] for row in train_rows] == [
        [0, 1, 2, 1],
        [1, 2, 0, 0],
    ]
    assert [row["action_text"] for row in train_rows] == [
        "MOVE_FORWARD+MOVE_RIGHT+ATTACK",
        "TURN_LEFT+MOVE_BACKWARD",
    ]
    assert metadata["action_layout"] == "deadly_corridor_factorized_11"
    assert metadata["checkpoint_train_step"] == 123


def test_deadly_corridor_action_vector_encodes_factorized_eleven_dim_actions():
    from latency_bench.data.action_layouts import encode_action_for_env

    row = encode_action_for_env("deadly_corridor", [2, 0, 1, 0])

    assert row["action"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert row["action_text"] == "TURN_RIGHT+MOVE_LEFT"


def test_collect_teacher_dataset_saves_demon_attack_latest_raw_rgb_image(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeProgress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            pass

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        action_space = object()

        def reset(self, seed=None):
            del seed
            obs = {"obs": np.zeros((1, 4, 2, 2), dtype=np.uint8)}
            raw_stack = np.stack(
                [
                    np.full((3, 2, 3), [10, 11, 12], dtype=np.uint8),
                    np.full((3, 2, 3), [20, 21, 22], dtype=np.uint8),
                    np.full((3, 2, 3), [30, 31, 32], dtype=np.uint8),
                    np.full((3, 2, 3), [40, 41, 42], dtype=np.uint8),
                ],
                axis=0,
            )
            return obs, [{"env_raw_rgb_frame_stack": raw_stack}]

        def step(self, action):
            del action
            obs = {"obs": np.ones((1, 4, 2, 2), dtype=np.uint8)}
            next_stack = np.stack([np.full((3, 2, 3), 50 + idx, dtype=np.uint8) for idx in range(4)], axis=0)
            return (
                obs,
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"latency_raw_frames": 2, "latency_ms": 66.0, "env_raw_rgb_frame_stack": next_stack}],
            )

        def close(self):
            return None

    monkeypatch.setattr(
        exporter,
        "build_sample_factory_rollout_cfg",
        lambda config: (object(), SimpleNamespace(frame_stack=4, env_fps=60.0, obs_fps=15.0, seed=0, simulator="cpu")),
    )
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [FakeEnv()],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.asarray([1], dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        },
    )
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "demon_attack"},
        output_dir=tmp_path / "export",
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

    image = Image.open(train_rows[0]["image"])
    assert not val_rows
    assert metadata["image_source"] == "env_raw_rgb"
    assert "image_stack" not in train_rows[0]
    assert image.mode == "RGB"
    assert image.size == (2, 3)
    assert np.asarray(image)[0, 0].tolist() == [40, 41, 42]


def test_collect_teacher_dataset_saves_flappy_latest_raw_rgb_image(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeProgress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            pass

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        action_space = object()

        def reset(self, seed=None):
            del seed
            obs = {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}
            raw_stack = np.stack(
                [
                    np.full((6, 5, 3), [10, 11, 12], dtype=np.uint8),
                    np.full((6, 5, 3), [20, 21, 22], dtype=np.uint8),
                    np.full((6, 5, 3), [30, 31, 32], dtype=np.uint8),
                    np.full((6, 5, 3), [40, 41, 42], dtype=np.uint8),
                ],
                axis=0,
            )
            return obs, [{"env_raw_rgb_frame_stack": raw_stack}]

        def step(self, action):
            del action
            obs = {"obs": np.ones((1, 12, 2, 2), dtype=np.uint8)}
            next_stack = np.stack([np.full((6, 5, 3), 50 + idx, dtype=np.uint8) for idx in range(4)], axis=0)
            return (
                obs,
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"latency_raw_frames": 2, "latency_ms": 66.0, "env_raw_rgb_frame_stack": next_stack}],
            )

        def close(self):
            return None

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=0)))
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [FakeEnv()],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.asarray([1], dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        },
    )
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=tmp_path / "export",
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

    image = Image.open(train_rows[0]["image"])
    assert not val_rows
    assert metadata["image_source"] == "env_raw_rgb"
    assert metadata["image_resolution"] == [6, 5, 3]
    assert "image_stack" not in train_rows[0]
    assert image.size == (5, 6)
    assert np.asarray(image)[0, 0].tolist() == [40, 41, 42]


def test_collect_teacher_dataset_uses_batched_driver_for_demon_attack_gpu(monkeypatch, tmp_path):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeProgress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            pass

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.arange(batch_size, dtype=torch.int64).reshape(batch_size, 1),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeBatchedEnv:
        action_space = object()

        def __init__(self):
            self.unwrapped = self
            self.num_agents = 3
            self.step_actions = []

        def reset(self, seed=None):
            del seed
            obs = {"obs": np.zeros((3, 4, 2, 2), dtype=np.uint8)}
            infos = [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(agent_idx, frames=4)}
                for agent_idx in range(3)
            ]
            return obs, infos

        def reset_agents(self, reset_plan, *, active_mask=None):
            del active_mask
            obs = {"obs": np.zeros((3, 4, 2, 2), dtype=np.uint8)}
            infos = [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(agent_idx + 10, frames=4)}
                for agent_idx in range(3)
            ]
            return obs, infos

        def set_active_agents(self, active_mask):
            self.active_mask = list(active_mask)

        def step(self, actions):
            self.step_actions.append(list(actions))
            obs = {"obs": np.ones((3, 4, 2, 2), dtype=np.uint8)}
            infos = [
                {
                    "latency_raw_frames": 2,
                    "latency_ms": 66.0,
                    "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(agent_idx + 20, frames=4),
                }
                for agent_idx in range(3)
            ]
            return (
                obs,
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                np.asarray([True, True, True], dtype=np.bool_),
                np.asarray([False, False, False], dtype=np.bool_),
                infos,
            )

        def close(self):
            return None

    fake_env = FakeBatchedEnv()
    fake_cfg = SimpleNamespace(
        frame_stack=4,
        env_fps=60.0,
        obs_fps=15.0,
        seed=0,
        simulator="gpu",
        gpu_env_batch_size=1,
    )
    build_env_calls = []

    def fake_runtime(module, cfg, config, num_envs):
        del module, config
        build_env_calls.append(num_envs)
        assert cfg.gpu_env_batch_size == 3
        return {
            "envs": [fake_env],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.arange(3, dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(exporter, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: list(processed_actions))
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    train_rows, val_rows, _ = exporter.collect_teacher_dataset(
        {"env_name": "demon_attack"},
        output_dir=tmp_path / "export",
        max_episodes=3,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=3,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert build_env_calls == [1]
    assert not val_rows
    assert [row["episode_idx"] for row in train_rows] == [0, 1, 2]
    assert fake_env.step_actions == [[0, 1, 2]]


def test_collect_teacher_dataset_writes_episode_shards_and_resumes(monkeypatch, tmp_path):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeProgress:
        calls: list[tuple[int, int]] = []
        updates: list[int] = []

        def __init__(self, *, total, initial, unit):
            self.calls.append((total, initial))
            self.unit = unit

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            self.updates.append(value)

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 99, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        reset_seeds: list[int] = []

        def __init__(self):
            self.action_space = object()

        def reset(self, seed=None):
            self.reset_seeds.append(int(seed))
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [
                {"reset": True, "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(int(seed), frames=4)}
            ]

        def step(self, action):
            del action
            return (
                {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    fake_cfg = SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=100)

    def fake_runtime(module, cfg, config, num_envs):
        return {
            "envs": [FakeEnv()],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.asarray([99], dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(exporter, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    output_dir = tmp_path / "export"
    train_rows, val_rows, _ = exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=output_dir,
        max_episodes=2,
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
    assert [row["episode_idx"] for row in train_rows] == [0, 1]
    assert (output_dir / "episode_rows" / "train" / "episode_000000.jsonl").exists()
    assert (output_dir / "episode_rows" / "train" / "episode_000001.jsonl").exists()
    assert FakeProgress.calls[-1] == (2, 0)
    assert FakeProgress.updates == [1, 1]
    assert FakeEnv.reset_seeds == [100, 101]

    FakeEnv.reset_seeds.clear()
    FakeProgress.updates.clear()

    resumed_train_rows, resumed_val_rows, _ = exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=output_dir,
        max_episodes=3,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert not resumed_val_rows
    assert [row["episode_idx"] for row in resumed_train_rows] == [0, 1, 2]
    assert FakeProgress.calls[-1] == (3, 2)
    assert FakeProgress.updates == [1]
    assert FakeEnv.reset_seeds == [102]


def test_collect_teacher_dataset_redirects_sample_factory_init_output(monkeypatch, tmp_path, capfd):
    import torch
    from latency_bench.data import sf_teacher_rollout as exporter

    class FakeProgress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, value):
            pass

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 99, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        action_space = object()

        def reset(self, seed=None):
            del seed
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [
                {"reset": True, "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5, frames=4)}
            ]

        def step(self, action):
            del action
            return (
                {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    def fake_build_config(config):
        print("sample factory stdout")
        print("sample factory stderr", file=sys.stderr)
        os.write(2, b"native init stderr\n")
        logging.getLogger("rl").warning("sample factory logger")
        return object(), SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=0)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", fake_build_config)
    monkeypatch.setattr(
        exporter,
        "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [FakeEnv()],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        {"env_name": "flappy"},
        output_dir=output_dir,
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

    captured = capfd.readouterr()
    init_log = (output_dir / "sample_factory_init.log").read_text(encoding="utf-8")
    assert "sample factory stdout" not in captured.out
    assert "sample factory stderr" not in captured.err
    assert "native init stderr" not in captured.err
    assert "sample factory logger" not in captured.err
    assert "sample factory stdout" in init_log
    assert "sample factory stderr" in init_log
    assert "native init stderr" in init_log
    assert "sample factory logger" in init_log


def test_write_parquet_splits_handles_empty_validation_rows(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)

    row = {
        "image": str(image_path),
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 11,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "FLAP",
    }

    output_dir = tmp_path / "export"
    dataset_dict = write_parquet_splits(output_dir, [row], [], action_spec=action_spec_for_env("flappy"))

    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert len(dataset_dict["train"]) == 1
    assert len(dataset_dict["val"]) == 0
    assert "image" in dataset_dict["train"].column_names
    assert "image_stack" not in dataset_dict["train"].column_names


def test_write_parquet_splits_embeds_image_bytes(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((2, 2, 3), 3, dtype=np.uint8)).save(image_path)

    row = {
        "image": str(image_path),
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 11,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "FLAP",
    }

    output_dir = tmp_path / "export"
    write_parquet_splits(output_dir, [row], [], action_spec=action_spec_for_env("flappy"))
    image_path.unlink()

    table = pq.read_table(output_dir / "train.parquet")
    image = table.to_pylist()[0]["image"]
    assert "image_stack" not in table.column_names
    assert image["path"] is None
    assert image["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_write_parquet_splits_preserves_rows_across_large_exports(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((1, 1, 3), 7, dtype=np.uint8)).save(image_path)
    rows = [
        {
            "image": str(image_path),
            "env_name": "flappy",
            "episode_idx": episode_idx,
            "decision_step": 0,
            "split": "train",
            "seed": 11 + episode_idx,
            "prompt": "pick action",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "raw_reward": 1.5,
            "action": [0.0, 1.0],
            "action_id": 1,
            "action_text": "FLAP",
        }
        for episode_idx in range(260)
    ]

    output_dir = tmp_path / "export"
    dataset_dict = write_parquet_splits(output_dir, rows, [], action_spec=action_spec_for_env("flappy"))
    table = pq.read_table(output_dir / "train.parquet")

    assert len(dataset_dict["train"]) == 260
    assert table.num_rows == 260
    assert table.to_pylist()[259]["episode_idx"] == 259


def  test_write_parquet_splits_from_shards_writes_train_and_val_rows(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits_from_shards

    output_dir = tmp_path / "export"
    rows_by_split = {
        "train": [
            (0, 0, 3),
            (1, 0, 5),
            (1, 1, 7),
        ],
        "val": [
            (2, 0, 11),
        ],
    }
    for split, specs in rows_by_split.items():
        for episode_idx, decision_step, pixel in specs:
            image_path = (
                output_dir
                / "images"
                / split
                / f"episode_{episode_idx:06d}"
                / f"decision_step_{decision_step:06d}.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((2, 2, 3), pixel, dtype=np.uint8)).save(image_path)
            row = {
                "image": str(image_path),
                "env_name": "flappy",
                "episode_idx": episode_idx,
                "decision_step": decision_step,
                "split": split,
                "seed": 11 + episode_idx,
                "prompt": "pick action",
                "latency_raw_frames": 2,
                "latency_ms": 66.0,
                "raw_reward": 1.5,
                "action": [0.0, 1.0],
                "action_id": 1,
                "action_text": "FLAP",
            }
            shard_path = output_dir / "episode_rows" / split / f"episode_{episode_idx:06d}.jsonl"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            with shard_path.open("a", encoding="utf-8") as handle:
                json.dump(row, handle, sort_keys=True)
                handle.write("\n")

    dataset_dict = write_parquet_splits_from_shards(output_dir, action_spec=action_spec_for_env("flappy"))
    train_table = pq.read_table(output_dir / "train.parquet")
    val_table = pq.read_table(output_dir / "val.parquet")

    assert len(dataset_dict["train"]) == 3
    assert len(dataset_dict["val"]) == 1
    assert train_table.num_rows == 3
    assert val_table.num_rows == 1
    assert [row["episode_idx"] for row in train_table.to_pylist()] == [0, 1, 1]
    assert [row["decision_step"] for row in train_table.to_pylist()] == [0, 0, 1]
    assert train_table.to_pylist()[0]["image"]["bytes"].startswith(b"\x89PNG\r\n\x1a\n")


def test_write_parquet_splits_from_shards_replaces_stale_partial_file(tmp_path):
    from PIL import Image
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.hf_parquet import write_parquet_splits, write_parquet_splits_from_shards

    output_dir = tmp_path / "export"
    stale_image_path = tmp_path / "stale_image.png"
    Image.fromarray(np.full((1, 1, 3), 1, dtype=np.uint8)).save(stale_image_path)
    stale_row = {
        "image": str(stale_image_path),
        "env_name": "flappy",
        "episode_idx": 99,
        "decision_step": 0,
        "split": "train",
        "seed": 99,
        "prompt": "stale",
        "latency_raw_frames": 0,
        "latency_ms": 0.0,
        "raw_reward": 0.0,
        "action": [1.0, 0.0],
        "action_id": 0,
        "action_text": "noop",
    }
    write_parquet_splits(output_dir, [stale_row], [], action_spec=action_spec_for_env("flappy"))

    image_path = output_dir / "images" / "train" / "episode_000000" / "t_000000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((2, 2, 3), 9, dtype=np.uint8)).save(image_path)
    row = {
        "image": str(image_path),
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 11,
        "prompt": "fresh",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "FLAP",
    }
    shard_path = output_dir / "episode_rows" / "train" / "episode_000000.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")

    write_parquet_splits_from_shards(output_dir, action_spec=action_spec_for_env("flappy"))
    table = pq.read_table(output_dir / "train.parquet")

    assert table.num_rows == 1
    assert table.to_pylist()[0]["prompt"] == "fresh"
    assert table.to_pylist()[0]["episode_idx"] == 0
    assert not (output_dir / "train.parquet.tmp").exists()


def test_main_cleans_intermediate_images_and_episode_rows_after_parquet(monkeypatch, tmp_path):
    from PIL import Image
    from latency_bench.data import sf_teacher_export as exporter

    def fake_collect_teacher_dataset(*args, output_dir, **kwargs):
        del args, kwargs
        image_path = output_dir / "images" / "train" / "episode_000000" / "t_000000.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 2, 3), 5, dtype=np.uint8)).save(image_path)
        shard_dir = output_dir / "episode_rows" / "train"
        shard_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image": str(image_path),
            "env_name": "flappy",
            "episode_idx": 0,
            "decision_step": 0,
            "split": "train",
            "seed": 11,
            "prompt": "pick action",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "raw_reward": 1.5,
            "action": [0.0, 1.0],
            "action_id": 1,
            "action_text": "FLAP",
        }
        (shard_dir / "episode_000000.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return (
            [],
            [],
            {
                "action_labels": ["NOOP", "FLAP"],
                "reward_field": "raw_reward",
                "episode_return_field": "episode_raw_return",
                "reward_semantics": "raw_reward",
                "filter_report": {
                    "preset": "",
                    "attempted_episodes": 1,
                    "accepted_episodes": 1,
                    "rejected_episodes": 0,
                    "accepted_train_episodes": 1,
                    "accepted_val_episodes": 0,
                    "train_rows_before_row_filter": 1,
                    "train_rows_after_row_filter": 1,
                    "val_rows_before_row_filter": 0,
                    "val_rows_after_row_filter": 0,
                },
                "checkpoint_train_step": 123,
                "frame_stack": 1,
                "env_fps": 30.0,
                "obs_fps": 15.0,
                "obs_stride_raw_frames": 2,
                "image_source": "env_raw_rgb",
                "image_resolution": [2, 2, 3],
                "train_episodes": 1,
                "val_episodes": 0,
            },
        )

    monkeypatch.setattr(
        exporter,
        "load_rollout_training_config",
        lambda *args, **kwargs: (
            {"env_name": "flappy"},
            {"latency_source": "checkpoint", "checkpoint_config": {"env": "latency_flappy"}},
        ),
    )
    monkeypatch.setattr(exporter, "collect_teacher_dataset", fake_collect_teacher_dataset)

    output_dir = tmp_path / "export"
    metadata = exporter.export_sample_factory_teacher_dataset(
        checkpoint_root=tmp_path / "checkpoint",
        output_dir=output_dir,
        latency_config_path=None,
        env_name="",
        max_episodes=1,
        max_steps_per_episode=8,
        val_fraction=0.1,
        split_seed=0,
        num_envs=1,
        device="",
        deterministic=True,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=8192,
        filter_preset="",
        push_to_hub=False,
        hf_repo_id="",
        hf_private=False,
        hf_token="",
        hf_max_shard_size="500MB",
    )

    assert metadata == json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["filter_report"] == json.loads((output_dir / "filter_report.json").read_text(encoding="utf-8"))
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert not (output_dir / "images").exists()
    assert not (output_dir / "episode_rows").exists()
    assert metadata["total_rows"] == 1
    assert metadata["rows_unit"] == "decision_step"
    assert metadata["total_episodes"] == 1
    assert metadata["image_resolution"] == [2, 2, 3]


def test_write_parquet_splits_disables_datasets_progress(monkeypatch, tmp_path):
    import latency_bench.data.hf_parquet as hf_parquet
    from latency_bench.data.action_layouts import action_spec_for_env

    calls: list[str] = []

    monkeypatch.setattr(hf_parquet.datasets, "disable_progress_bars", lambda: calls.append("disabled"))

    hf_parquet.write_parquet_splits(tmp_path / "export", [], [], action_spec=action_spec_for_env("flappy"))

    assert calls == ["disabled"]
    assert (tmp_path / "export" / "train.parquet").exists()
    assert (tmp_path / "export" / "val.parquet").exists()


def test_exporter_parser_has_no_prompt_or_render_image_source_switches():
    from scripts.rollout_data.export_sf_teacher_dataset import build_parser

    option_dests = {action.dest for action in build_parser()._actions}

    assert "image_source" not in option_dests
    assert "prompt" not in option_dests
    assert "prompt_template" not in option_dests
    assert "config" not in option_dests


def test_main_writes_prompt_source_and_timing_metadata(monkeypatch, tmp_path):
    from latency_bench.data import sf_teacher_export as exporter

    def fake_write_parquet_splits_from_shards(output_dir, **kwargs):
        del kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"train": [], "val": []}

    monkeypatch.setattr(
        exporter,
        "load_rollout_training_config",
        lambda *args, **kwargs: (
            {"env_name": "flappy"},
            {"latency_source": "checkpoint", "checkpoint_config": {"env": "latency_flappy"}},
        ),
    )
    monkeypatch.setattr(
        exporter,
        "collect_teacher_dataset",
        lambda *args, **kwargs: (
            [],
            [],
            {
                "action_labels": ["NOOP", "FLAP"],
                "reward_field": "raw_reward",
                "episode_return_field": "episode_raw_return",
                "reward_semantics": "raw_reward",
                "filter_report": {
                    "preset": "",
                    "attempted_episodes": 0,
                    "accepted_episodes": 0,
                    "rejected_episodes": 0,
                    "accepted_train_episodes": 0,
                    "accepted_val_episodes": 0,
                    "train_rows_before_row_filter": 0,
                    "train_rows_after_row_filter": 0,
                    "val_rows_before_row_filter": 0,
                    "val_rows_after_row_filter": 0,
                },
                "checkpoint_train_step": 123,
                "frame_stack": 4,
                "env_fps": 30.0,
                "obs_fps": 15.0,
                "obs_stride_raw_frames": 2,
                "image_source": "env_raw_rgb",
                "image_resolution": [84, 84, 3],
                "train_episodes": 0,
                "val_episodes": 0,
            },
        ),
    )
    monkeypatch.setattr(exporter, "write_parquet_splits_from_shards", fake_write_parquet_splits_from_shards)
    monkeypatch.setattr(exporter, "image_metadata_from_shards", lambda output_dir: {"image_resolution": [84, 84, 3]})

    output_dir = tmp_path / "export"
    metadata = exporter.export_sample_factory_teacher_dataset(
        checkpoint_root=tmp_path / "checkpoint",
        output_dir=output_dir,
        latency_config_path=None,
        env_name="",
        max_episodes=1,
        max_steps_per_episode=8,
        val_fraction=0.1,
        split_seed=0,
        num_envs=1,
        device="",
        deterministic=True,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=8192,
        filter_preset="",
        push_to_hub=False,
        hf_repo_id="",
        hf_private=False,
        hf_token="",
        hf_max_shard_size="500MB",
    )

    assert metadata == json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_source"] == "latency_bench.data.prompts"
    assert metadata["base_prompt"] == exporter.BASE_PROMPTS["flappy"]
    assert "env_fps" in metadata["prompt_timing_suffix"]
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2
    assert metadata["image_source"] == "env_raw_rgb"
    assert metadata["total_rows"] == 0
    assert metadata["rows_unit"] == "decision_step"
    assert metadata["total_episodes"] == 0
    assert metadata["image_resolution"] == [84, 84, 3]


def test_push_token_prefers_explicit_then_environment(monkeypatch):
    from scripts.rollout_data.push_sf_teacher_dataset import resolve_token

    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    assert resolve_token(" explicit-token ") == "explicit-token"
    assert resolve_token("") == "env-token"


def test_push_token_falls_back_to_huggingface_hub_token(monkeypatch):
    from scripts.rollout_data.push_sf_teacher_dataset import resolve_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-token")

    assert resolve_token("") == "hub-token"


def test_prepare_starvla_lerobot_dataset_converts_from_raw_export(monkeypatch, tmp_path):
    from scripts.rollout_data import prepare_starvla_lerobot_dataset as prepare

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(json.dumps({"env_name": "demon_attack"}), encoding="utf-8")
    (raw_export / "train.parquet").write_bytes(b"train")
    (raw_export / "val.parquet").write_bytes(b"val")

    calls = []

    def fake_convert(raw_export_dir, output_dir, **kwargs):
        calls.append((raw_export_dir, output_dir, kwargs))
        (output_dir / "meta").mkdir(parents=True)
        (output_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
        (output_dir.with_name(f"{output_dir.name}__val") / "meta").mkdir(parents=True)
        return {"dataset_name": output_dir.name}

    monkeypatch.setattr(prepare, "convert_canonical_rollout_to_lerobot", fake_convert)

    result = prepare.main([
        str(raw_export),
        str(tmp_path / "lerobot"),
        "demon_attack_train",
    ])

    assert result == 0
    assert calls == [
        (
            raw_export,
            tmp_path / "lerobot" / "demon_attack_train",
            {"force": True},
        )
    ]


def test_prepare_starvla_lerobot_dataset_uploads_converted_root(monkeypatch, tmp_path):
    from scripts.rollout_data import prepare_starvla_lerobot_dataset as prepare

    lerobot_root = tmp_path / "lerobot"
    (lerobot_root / "demon_attack_train").mkdir(parents=True)
    (lerobot_root / "demon_attack_train__val").mkdir()

    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(prepare, "HfApi", lambda: FakeApi())
    monkeypatch.setenv("HF_TOKEN", "env-token")

    prepare.push_lerobot_dataset(
        lerobot_root=lerobot_root,
        data_mix="demon_attack_train",
        repo_id="user/demon_attack_train_lerobot",
        private=True,
    )

    assert calls == [
        (
            "create_repo",
            {
                "repo_id": "user/demon_attack_train_lerobot",
                "repo_type": "dataset",
                "private": True,
                "exist_ok": True,
                "token": "env-token",
            },
        ),
        (
            "upload_folder",
            {
                "folder_path": str(lerobot_root),
                "repo_id": "user/demon_attack_train_lerobot",
                "repo_type": "dataset",
                "allow_patterns": ["demon_attack_train/**", "demon_attack_train__val/**"],
                "commit_message": "Upload StarVLA LeRobot dataset",
                "token": "env-token",
            },
        ),
    ]
