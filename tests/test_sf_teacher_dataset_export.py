from __future__ import annotations

import json
import logging
import os
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


def test_rollout_config_defaults_to_checkpoint_training_latency(tmp_path):
    from scripts.rollout_data.export_sf_teacher_dataset import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)

    config, metadata = load_rollout_training_config(experiment_dir)

    assert config["env"]["name"] == "demon_attack"
    assert config["experiment"]["name"] == "demon_fixed_l4_seed0"
    assert config["backend"]["run_mode"] == "eval"
    assert config["backend"]["device"] == "cuda"
    assert config["backend"]["train_dir"] == str(tmp_path)
    assert config["policy"]["checkpoint_path"] is None
    assert config["policy"]["device"] == "cuda"
    assert config["latency"]["type"] == "fixed"
    assert config["latency"]["fixed_latency_ms"] == pytest.approx(66.66666666666667)
    assert config["latency"]["add_latency_info"] is True
    assert metadata["latency_source"] == "checkpoint"


def test_latency_config_override_replaces_only_latency_section(tmp_path):
    from scripts.rollout_data.export_sf_teacher_dataset import load_rollout_training_config

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
    from scripts.rollout_data.export_sf_teacher_dataset import load_rollout_training_config

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


def test_device_override_cuda_uses_cuda_public_device_name(tmp_path):
    from scripts.rollout_data.export_sf_teacher_dataset import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)

    config, _ = load_rollout_training_config(experiment_dir, device_override="cuda")

    assert config["backend"]["device"] == "cuda"
    assert config["policy"]["device"] == "cuda"


def test_dataset_frames_use_env_raw_rgb_info():
    from scripts.rollout_data.export_sf_teacher_dataset import frames_for_dataset_image

    raw_stack = np.stack(
        [
            np.full((4, 5, 3), [11, 12, 13], dtype=np.uint8),
            np.full((4, 5, 3), [21, 22, 23], dtype=np.uint8),
        ],
        axis=0,
    )

    frames = frames_for_dataset_image({"env_raw_rgb_frame_stack": raw_stack})

    assert len(frames) == 2
    assert frames[0].shape == (4, 5, 3)
    assert frames[0][0, 0].tolist() == [11, 12, 13]
    assert frames[1][0, 0].tolist() == [21, 22, 23]



def test_teacher_prompts_cover_rollout_envs():
    from scripts.rollout_data.prompts import BASE_PROMPTS

    assert set(BASE_PROMPTS) == {"flappy", "demon_attack", "deadly_corridor"}


def test_batch_observations_keeps_already_batched_mapping_observation(monkeypatch, tmp_path):
    import torch
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    observation = {"obs": np.zeros((1, 12, 84, 84), dtype=np.uint8)}

    batch = exporter._batch_observations([observation])

    assert batch["obs"].shape == (1, 12, 84, 84)


def test_batch_observations_concatenates_single_agent_batched_mapping_observations():
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    observations = [
        {"obs": np.full((1, 4, 2, 2), slot, dtype=np.uint8)}
        for slot in range(2)
    ]

    batch = exporter._batch_observations(observations)

    assert batch["obs"].shape == (2, 4, 2, 2)
    assert batch["obs"][0].tolist() == observations[0]["obs"][0].tolist()
    assert batch["obs"][1].tolist() == observations[1]["obs"][0].tolist()


def test_collect_teacher_dataset_handles_single_env_rollout(monkeypatch, tmp_path):
    import torch
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [7] * batch_size)
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
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
    assert fake_env.step_actions == [[7]]
    assert train_rows[0]["action_id"] == 7
    assert len(train_rows[0]["image_stack"]) == 1
    assert Path(train_rows[0]["image_stack"][-1]).exists()
    assert "image" not in train_rows[0]
    assert train_rows[0]["reward"] == 1.5
    assert train_rows[0]["done"] is True
    assert metadata["checkpoint_train_step"] == 123
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2


def test_collect_teacher_dataset_handles_single_agent_batched_step_outputs(monkeypatch, tmp_path):
    import torch
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
    fake_cfg = SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=11)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        exporter,
        "_build_env_and_policy",
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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [7] * batch_size)
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
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
    assert [env.step_actions for env in fake_envs] == [[[7]], [[7]]]
    assert [row["action_id"] for row in train_rows] == [7, 7]
    assert len(train_rows[0]["image_stack"]) == 4
    assert Path(train_rows[0]["image_stack"][-1]).exists()
    assert "image" not in train_rows[0]
    assert train_rows[0]["reward"] == 1.5
    assert train_rows[0]["done"] is True
    assert train_rows[0]["latency_raw_frames"] == 2
    assert metadata["checkpoint_train_step"] == 123
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2


def test_collect_teacher_dataset_stacks_deadly_corridor_deterministic_tuple_actions(monkeypatch, tmp_path):
    import torch
    from gymnasium.spaces import Discrete, Tuple
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
        "_build_env_and_policy",
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
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "deadly_corridor"}},
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
    assert [row["action_text"] for row in train_rows] == [
        "TURN_NOOP+MOVE_FORWARD+MOVE_RIGHT+ATTACK",
        "TURN_LEFT+MOVE_BACKWARD+STRAFE_NOOP+ATTACK_NOOP",
    ]
    assert metadata["checkpoint_train_step"] == 123


def test_deadly_corridor_action_vector_encodes_factorized_tuple():
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    vector, label = exporter.action_vector_and_text_for_env("deadly_corridor", [2, 0, 1, 0], {})

    assert vector == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert label == "TURN_RIGHT+MOVE_NOOP+MOVE_LEFT+ATTACK_NOOP"


def test_collect_teacher_dataset_saves_demon_attack_raw_rgb_stack(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", lambda config: (object(), SimpleNamespace(frame_stack=4, env_fps=60.0, obs_fps=15.0, seed=0)))
    monkeypatch.setattr(
        exporter,
        "_build_env_and_policy",
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
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "demon_attack"}},
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

    stack_images = [Image.open(path) for path in train_rows[0]["image_stack"]]
    assert not val_rows
    assert metadata["image_source"] == "env_raw_rgb"
    assert metadata["image_stack_source"] == "env_raw_rgb_frame_stack"
    assert "image" not in train_rows[0]
    assert [img.mode for img in stack_images] == ["RGB", "RGB", "RGB", "RGB"]
    assert [img.size for img in stack_images] == [(2, 3), (2, 3), (2, 3), (2, 3)]
    assert [np.asarray(img)[0, 0].tolist() for img in stack_images] == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_collect_teacher_dataset_saves_flappy_env_raw_rgb_stack(monkeypatch, tmp_path):
    import torch
    from PIL import Image
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
        "_build_env_and_policy",
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
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    train_rows, val_rows, metadata = exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
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

    stack_images = [Image.open(path) for path in train_rows[0]["image_stack"]]
    assert not val_rows
    assert metadata["image_source"] == "env_raw_rgb"
    assert metadata["image_stack_source"] == "env_raw_rgb_frame_stack"
    assert metadata["image_resolution"] == [6, 5, 3]
    assert [img.size for img in stack_images] == [(5, 6), (5, 6), (5, 6), (5, 6)]
    assert [np.asarray(img)[0, 0].tolist() for img in stack_images] == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_collect_teacher_dataset_writes_episode_shards_and_resumes(monkeypatch, tmp_path):
    import torch
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
    monkeypatch.setattr(exporter, "_build_env_and_policy", fake_runtime)
    monkeypatch.setattr(exporter, "_normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [7] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    output_dir = tmp_path / "export"
    train_rows, val_rows, _ = exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
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
        {"env": {"name": "flappy"}},
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
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
        print("sample factory stderr", file=exporter.sys.stderr)
        os.write(2, b"native init stderr\n")
        logging.getLogger("rl").warning("sample factory logger")
        return object(), SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=0)

    monkeypatch.setattr(exporter, "build_sample_factory_rollout_cfg", fake_build_config)
    monkeypatch.setattr(
        exporter,
        "_build_env_and_policy",
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
    monkeypatch.setattr(exporter, "_split_batch_actions", lambda processed_actions, batch_size, action_space=None: [7] * batch_size)
    monkeypatch.setattr(exporter, "tqdm", FakeProgress)
    monkeypatch.setattr(
        exporter,
        "sample_factory_latency_params_from_flat_config",
        lambda cfg, env_fps: SimpleNamespace(fixed_latency_raw_frames=0),
    )

    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        {"env": {"name": "flappy"}},
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
    from scripts.rollout_data.export_sf_teacher_dataset import _write_parquet_splits

    stack_paths = [tmp_path / f"stack_{idx:02d}.png" for idx in range(2)]
    for idx, path in enumerate(stack_paths):
        Image.fromarray(np.full((2, 2, 3), idx, dtype=np.uint8)).save(path)
    image_path = stack_paths[-1]
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)

    row = {
        "image_stack": [str(path) for path in stack_paths],
        "split": "train",
        "prompt": "pick action",
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "flap",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "reward": 1.5,
        "done": True,
        "episode_idx": 0,
        "t": 0,
        "seed": 11,
        "deterministic": True,
    }

    output_dir = tmp_path / "export"
    dataset_dict = _write_parquet_splits(output_dir, [row], [])

    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert len(dataset_dict["train"]) == 1
    assert len(dataset_dict["val"]) == 0
    assert len(dataset_dict["train"][0]["image_stack"]) == 2
    assert "image" not in dataset_dict["train"].column_names


def test_write_parquet_splits_embeds_image_stack_bytes(tmp_path):
    from PIL import Image
    from scripts.rollout_data.export_sf_teacher_dataset import _write_parquet_splits

    stack_paths = [tmp_path / f"stack_{idx:02d}.png" for idx in range(2)]
    for idx, path in enumerate(stack_paths):
        Image.fromarray(np.full((2, 2, 3), idx + 3, dtype=np.uint8)).save(path)

    row = {
        "image_stack": [str(path) for path in stack_paths],
        "split": "train",
        "prompt": "pick action",
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "flap",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "reward": 1.5,
        "done": True,
        "episode_idx": 0,
        "t": 0,
        "seed": 11,
        "deterministic": True,
    }

    output_dir = tmp_path / "export"
    _write_parquet_splits(output_dir, [row], [])
    for path in stack_paths:
        path.unlink()

    table = pq.read_table(output_dir / "train.parquet")
    image_stack = table.to_pylist()[0]["image_stack"]
    assert "image" not in table.column_names
    assert [item["path"] for item in image_stack] == [None, None]
    assert all(item["bytes"].startswith(b"\x89PNG\r\n\x1a\n") for item in image_stack)


def test_write_parquet_splits_preserves_rows_across_large_exports(tmp_path):
    from PIL import Image
    from scripts.rollout_data.export_sf_teacher_dataset import _write_parquet_splits

    stack_path = tmp_path / "stack_00.png"
    Image.fromarray(np.full((1, 1, 3), 7, dtype=np.uint8)).save(stack_path)
    rows = [
        {
            "image_stack": [str(stack_path)],
            "split": "train",
            "prompt": "pick action",
            "action": [0.0, 1.0],
            "action_id": 1,
            "action_text": "flap",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "reward": 1.5,
            "done": True,
            "episode_idx": episode_idx,
            "t": 0,
            "seed": 11 + episode_idx,
            "deterministic": True,
        }
        for episode_idx in range(260)
    ]

    output_dir = tmp_path / "export"
    dataset_dict = _write_parquet_splits(output_dir, rows, [])
    table = pq.read_table(output_dir / "train.parquet")

    assert len(dataset_dict["train"]) == 260
    assert table.num_rows == 260
    assert table.to_pylist()[259]["episode_idx"] == 259


def  test_write_parquet_splits_from_shards_writes_train_and_val_rows(tmp_path):
    from PIL import Image
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

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
        for episode_idx, t, pixel in specs:
            stack_dir = output_dir / "images" / split / f"episode_{episode_idx:06d}" / f"t_{t:06d}"
            stack_dir.mkdir(parents=True, exist_ok=True)
            stack_paths = []
            for frame_idx in range(2):
                stack_path = stack_dir / f"stack_{frame_idx:02d}.png"
                Image.fromarray(np.full((2, 2, 3), pixel + frame_idx, dtype=np.uint8)).save(stack_path)
                stack_paths.append(str(stack_path))
            row = {
                "image_stack": stack_paths,
                "split": split,
                "prompt": "pick action",
                "action": [0.0, 1.0],
                "action_id": 1,
                "action_text": "flap",
                "latency_raw_frames": 2,
                "latency_ms": 66.0,
                "reward": 1.5,
                "done": True,
                "episode_idx": episode_idx,
                "t": t,
                "seed": 11 + episode_idx,
                "deterministic": True,
            }
            shard_path = output_dir / "episode_rows" / split / f"episode_{episode_idx:06d}.jsonl"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            with shard_path.open("a", encoding="utf-8") as handle:
                json.dump(row, handle, sort_keys=True)
                handle.write("\n")

    dataset_dict = exporter._write_parquet_splits_from_shards(output_dir)
    train_table = pq.read_table(output_dir / "train.parquet")
    val_table = pq.read_table(output_dir / "val.parquet")

    assert len(dataset_dict["train"]) == 3
    assert len(dataset_dict["val"]) == 1
    assert train_table.num_rows == 3
    assert val_table.num_rows == 1
    assert [row["episode_idx"] for row in train_table.to_pylist()] == [0, 1, 1]
    assert [row["t"] for row in train_table.to_pylist()] == [0, 0, 1]
    assert all(item["bytes"].startswith(b"\x89PNG\r\n\x1a\n") for item in train_table.to_pylist()[0]["image_stack"])


def test_write_parquet_splits_from_shards_replaces_stale_partial_file(tmp_path):
    from PIL import Image
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    output_dir = tmp_path / "export"
    stale_stack_path = tmp_path / "stale_stack.png"
    Image.fromarray(np.full((1, 1, 3), 1, dtype=np.uint8)).save(stale_stack_path)
    stale_row = {
        "image_stack": [str(stale_stack_path)],
        "split": "train",
        "prompt": "stale",
        "action": [1.0, 0.0],
        "action_id": 0,
        "action_text": "noop",
        "latency_raw_frames": 0,
        "latency_ms": 0.0,
        "reward": 0.0,
        "done": True,
        "episode_idx": 99,
        "t": 0,
        "seed": 99,
        "deterministic": True,
    }
    exporter._write_parquet_splits(output_dir, [stale_row], [])

    stack_dir = output_dir / "images" / "train" / "episode_000000" / "t_000000"
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stack_dir / "stack_00.png"
    Image.fromarray(np.full((2, 2, 3), 9, dtype=np.uint8)).save(stack_path)
    row = {
        "image_stack": [str(stack_path)],
        "split": "train",
        "prompt": "fresh",
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "flap",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "reward": 1.5,
        "done": True,
        "episode_idx": 0,
        "t": 0,
        "seed": 11,
        "deterministic": True,
    }
    shard_path = output_dir / "episode_rows" / "train" / "episode_000000.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")

    exporter._write_parquet_splits_from_shards(output_dir)
    table = pq.read_table(output_dir / "train.parquet")

    assert table.num_rows == 1
    assert table.to_pylist()[0]["prompt"] == "fresh"
    assert table.to_pylist()[0]["episode_idx"] == 0
    assert not (output_dir / "train.parquet.tmp").exists()


def test_main_cleans_intermediate_images_and_episode_rows_after_parquet(monkeypatch, tmp_path):
    from PIL import Image
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    def fake_collect_teacher_dataset(*args, output_dir, **kwargs):
        del args, kwargs
        stack_dir = output_dir / "images" / "train" / "episode_000000" / "t_000000"
        stack_dir.mkdir(parents=True, exist_ok=True)
        stack_path = stack_dir / "stack_00.png"
        Image.fromarray(np.full((2, 2, 3), 5, dtype=np.uint8)).save(stack_path)
        shard_dir = output_dir / "episode_rows" / "train"
        shard_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "image_stack": [str(stack_path)],
            "split": "train",
            "prompt": "pick action",
            "action": [0.0, 1.0],
            "action_id": 1,
            "action_text": "flap",
            "latency_raw_frames": 2,
            "latency_ms": 66.0,
            "reward": 1.5,
            "done": True,
            "episode_idx": 0,
            "t": 0,
            "seed": 11,
            "deterministic": True,
        }
        (shard_dir / "episode_000000.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return (
            [],
            [],
            {
                "action_labels": {0: "noop", 1: "flap"},
                "checkpoint_train_step": 123,
                "frame_stack": 1,
                "env_fps": 30.0,
                "obs_fps": 15.0,
                "obs_stride_raw_frames": 2,
                "image_source": "env_raw_rgb",
                "image_stack_source": "env_raw_rgb_frame_stack",
                "image_stack_size": 1,
                "image_resolution": [2, 2, 3],
                "train_episodes": 1,
                "val_episodes": 0,
            },
        )

    monkeypatch.setattr(
        exporter,
        "load_rollout_training_config",
        lambda *args, **kwargs: (
            {"env": {"name": "flappy"}},
            {"latency_source": "checkpoint", "latency_config": {"type": "zero"}},
        ),
    )
    monkeypatch.setattr(exporter, "collect_teacher_dataset", fake_collect_teacher_dataset)

    output_dir = tmp_path / "export"
    result = exporter.main(
        [
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(output_dir),
            "--max-episodes",
            "1",
            "--max-steps-per-episode",
            "8",
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert result == 0
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert not (output_dir / "images").exists()
    assert not (output_dir / "episode_rows").exists()
    assert metadata["total_rows"] == 1
    assert metadata["rows_unit"] == "decision_step"
    assert metadata["total_episodes"] == 1
    assert metadata["image_stack_size"] == 1
    assert metadata["image_resolution"] == [2, 2, 3]


def test_write_parquet_splits_disables_datasets_progress(monkeypatch, tmp_path):
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    calls: list[str] = []

    monkeypatch.setattr(exporter.datasets, "disable_progress_bars", lambda: calls.append("disabled"))

    exporter._write_parquet_splits(tmp_path / "export", [], [])

    assert calls == ["disabled"]
    assert (tmp_path / "export" / "train.parquet").exists()
    assert (tmp_path / "export" / "val.parquet").exists()


def test_exporter_parser_has_no_prompt_or_render_image_source_switches():
    from scripts.rollout_data.export_sf_teacher_dataset import build_parser

    option_dests = {action.dest for action in build_parser()._actions}

    assert "image_source" not in option_dests
    assert "prompt" not in option_dests
    assert "prompt_template" not in option_dests


def test_main_writes_prompt_source_and_timing_metadata(monkeypatch, tmp_path):
    from scripts.rollout_data import export_sf_teacher_dataset as exporter

    def fake_write_parquet_splits_from_shards(output_dir, **kwargs):
        del kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"train": [], "val": []}

    monkeypatch.setattr(
        exporter,
        "load_rollout_training_config",
        lambda *args, **kwargs: (
            {"env": {"name": "flappy"}},
            {"latency_source": "checkpoint", "latency_config": {"type": "zero"}},
        ),
    )
    monkeypatch.setattr(
        exporter,
        "collect_teacher_dataset",
        lambda *args, **kwargs: (
            [],
            [],
            {
                "action_labels": {0: "noop", 1: "flap"},
                "checkpoint_train_step": 123,
                "frame_stack": 4,
                "env_fps": 30.0,
                "obs_fps": 15.0,
                "obs_stride_raw_frames": 2,
                "image_source": "env_raw_rgb",
                "image_stack_source": "env_raw_rgb_frame_stack",
                "image_stack_size": 4,
                "image_resolution": [84, 84, 3],
                "train_episodes": 0,
                "val_episodes": 0,
            },
        ),
    )
    monkeypatch.setattr(exporter, "_write_parquet_splits_from_shards", fake_write_parquet_splits_from_shards)

    result = exporter.main(
        [
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path / "export"),
            "--max-episodes",
            "1",
            "--max-steps-per-episode",
            "8",
        ]
    )

    metadata = json.loads((tmp_path / "export" / "metadata.json").read_text(encoding="utf-8"))
    assert result == 0
    assert metadata["prompt_source"] == "scripts/rollout_data/prompts.py"
    assert metadata["base_prompt"] == exporter.BASE_PROMPTS["flappy"]
    assert "env_fps" in metadata["prompt_timing_suffix"]
    assert metadata["env_fps"] == 30.0
    assert metadata["obs_fps"] == 15.0
    assert metadata["obs_stride_raw_frames"] == 2
    assert metadata["image_source"] == "env_raw_rgb"
    assert metadata["image_stack_source"] == "env_raw_rgb_frame_stack"
    assert metadata["total_rows"] == 0
    assert metadata["rows_unit"] == "decision_step"
    assert metadata["total_episodes"] == 0
    assert metadata["image_stack_size"] == 4
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

    def fake_convert(dataset_name, output_dir, **kwargs):
        calls.append((dataset_name, output_dir, kwargs))
        (output_dir / "meta").mkdir(parents=True)
        (output_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
        (output_dir.with_name(f"{output_dir.name}__val") / "meta").mkdir(parents=True)
        return {"dataset_name": output_dir.name}

    monkeypatch.setitem(prepare.CONVERTERS, "demon_attack", fake_convert)

    result = prepare.main([
        str(raw_export),
        str(tmp_path / "lerobot"),
        "demon_attack_train",
    ])

    assert result == 0
    assert calls == [
        (
            str(raw_export),
            tmp_path / "lerobot" / "demon_attack_train",
            {"force": True, "require_latency_prompt_map": False},
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
