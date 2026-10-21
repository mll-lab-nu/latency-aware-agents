from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
import yaml
from helpers import REPO_ROOT


def _rollout_descriptor(env_name: str) -> dict:
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.prompts import BASE_PROMPTS

    return {
        "env_name": env_name,
        "integration_name": env_name,
        "action_spec": action_spec_for_env(env_name),
        "base_prompt": BASE_PROMPTS[env_name],
        "flat_cfg": {},
    }


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
        "reset_mode": "fire",
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


def test_export_cli_accepts_flappy_ghost_trail_image_transform():
    from scripts.rollout_data.export_sf_teacher_dataset import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint-root",
            "ckpt",
            "--output-dir",
            "out",
            "--max-episodes",
            "1",
            "--max-steps-per-episode",
            "10",
            "--image-transform",
            "flappy_ghost_trail",
            "--ghost-trail-history-frames",
            "5",
            "--ghost-trail-gamma",
            "1.3",
            "--hf-data-dir",
            "latency-aware/flappy/ghost-trail",
        ]
    )

    assert args.image_transform == "flappy_ghost_trail"
    assert args.ghost_trail_history_frames == 5
    assert args.ghost_trail_gamma == 1.3
    assert args.hf_data_dir == "latency-aware/flappy/ghost-trail"


def _read_shard_rows(output_dir: Path, split: str) -> list[dict]:
    from latency_bench.data.hf_parquet import iter_episode_rows

    return list(iter_episode_rows(output_dir, split))


def _load_sf_teacher_drivers_for_direct_test(monkeypatch):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=type("FakeTensor", (), {})))
    sample_factory_actions = ModuleType("latency_bench.integrations.sample_factory.actions")
    sample_factory_actions.to_python_action = lambda value: value.item() if hasattr(value, "item") else value
    monkeypatch.setitem(sys.modules, sample_factory_actions.__name__, sample_factory_actions)
    module_path = REPO_ROOT / "latency_bench" / "integrations" / "sample_factory" / "teacher_drivers.py"
    spec = importlib.util.spec_from_file_location("_test_sf_teacher_drivers", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rollout_config_reads_checkpoint_config_without_rebuilding_nested_training_config(tmp_path):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

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


@pytest.mark.parametrize("method", ["iid", "temporal"])
def test_profile_latency_override_preserves_sampler_method(method):
    from latency_bench.integrations.sample_factory.teacher_runtime import _apply_latency_to_flat_cfg
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(simulated_worker_capacity=None)
    _apply_latency_to_flat_cfg(
        cfg,
        {"method": method, "profile_path": "profile.json", "profile_worker_slot": 0, "seed": 0},
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=10)
    assert params.profile_method == method


def test_latency_config_override_is_explicit_rollout_override(tmp_path):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)
    override_path = tmp_path / "latency_override.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "latency": {
                    "method": "uniform",
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
        "method": "uniform",
        "min_latency_ms": 10.0,
        "max_latency_ms": 80.0,
        "seed": 123,
    }
    assert metadata["latency_source"] == str(override_path.resolve())


@pytest.mark.parametrize(
    ("env", "experiment", "expected_ms"),
    [
        ("latency_flappy", "flappy_sf_same_latency_l1_seed0", 1000.0 / 30.0),
        ("latency_demon_attack", "demon_attack_sf_same_latency_l1_seed0", 4 * 1000.0 / 60.0),
        ("latency_demon_attack", "demon_attack_sf_same_latency_l2_seed0", 8 * 1000.0 / 60.0),
        ("doom_deadly_corridor", "deadly_corridor_sf_same_latency_l1_seed0", 4 * 1000.0 / 35.0),
    ],
)
def test_rollout_config_infers_legacy_same_latency_checkpoint_fixed_latency(
    tmp_path,
    env,
    experiment,
    expected_ms,
):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / experiment
    _write_checkpoint_config(
        experiment_dir,
        env=env,
        experiment=experiment,
        latency_type="fixed",
        fixed_latency_ms=1.0,
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        queue_latency_values="1",
        env_fps=None,
        obs_fps=None,
    )

    descriptor, metadata = load_rollout_training_config(experiment_dir)

    assert descriptor["latency_override"] == {
        "method": "fixed",
        "fixed_latency_ms": expected_ms,
        "add_latency_info": False,
    }
    assert metadata["latency_override"] == descriptor["latency_override"]
    assert metadata["latency_source"] == "legacy_checkpoint_name"


def test_explicit_latency_config_takes_precedence_over_legacy_checkpoint_name(tmp_path):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "flappy_sf_same_latency_l1_seed0"
    _write_checkpoint_config(
        experiment_dir,
        env="latency_flappy",
        experiment="flappy_sf_same_latency_l1_seed0",
        latency_type=None,
        fixed_latency_ms=None,
        latency_mode="episode_fixed",
        queue_latency_values="1",
        env_fps=None,
        obs_fps=None,
    )
    override_path = tmp_path / "latency_override.yaml"
    override_path.write_text(
        yaml.safe_dump({"latency": {"method": "fixed", "fixed_latency_ms": 99.0}}),
        encoding="utf-8",
    )

    descriptor, metadata = load_rollout_training_config(experiment_dir, latency_config_path=override_path)

    assert descriptor["latency_override"] == {"method": "fixed", "fixed_latency_ms": 99.0}
    assert metadata["latency_source"] == str(override_path.resolve())


def test_device_override_is_recorded_without_rewriting_checkpoint_env_config(tmp_path):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

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
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "demon_fixed_l4_seed0"
    _write_checkpoint_config(experiment_dir)

    descriptor, _ = load_rollout_training_config(experiment_dir, device_override="cuda")

    assert descriptor["device_override"] == "gpu"


def test_batch_observations_keeps_already_batched_mapping_observation():
    from latency_bench.integrations.sample_factory import teacher_drivers as sf_teacher_drivers

    observation = {"obs": np.zeros((1, 12, 84, 84), dtype=np.uint8)}

    batch = sf_teacher_drivers.batch_observations([observation])

    assert batch["obs"].shape == (1, 12, 84, 84)


def test_batch_observations_concatenates_single_agent_batched_mapping_observations():
    from latency_bench.integrations.sample_factory import teacher_drivers as sf_teacher_drivers

    observations = [
        {"obs": np.full((1, 4, 2, 2), slot, dtype=np.uint8)}
        for slot in range(2)
    ]

    batch = sf_teacher_drivers.batch_observations(observations)

    assert batch["obs"].shape == (2, 4, 2, 2)
    assert batch["obs"][0].tolist() == observations[0]["obs"][0].tolist()
    assert batch["obs"][1].tolist() == observations[1]["obs"][0].tolist()


def test_batch_observations_preserves_decision_history_batch_axis():
    from latency_bench.integrations.sample_factory.teacher_drivers import batch_observations

    observations = [
        {"obs": np.zeros((1, 4, 2, 2), dtype=np.uint8),
         "transport": np.full((1, 24), slot, dtype=np.float32)}
        for slot in range(2)
    ]

    batch = batch_observations(observations)

    assert batch["transport"].shape == (2, 24)
    assert batch["transport"][:, 0].tolist() == [0.0, 1.0]


def test_serial_rollout_driver_continues_life_loss_with_seedless_reset(monkeypatch):
    sf_teacher_drivers = _load_sf_teacher_drivers_for_direct_test(monkeypatch)

    class FakeEnv:
        action_space = object()

        def __init__(self):
            self.reset_seeds = []

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            return {"obs": np.asarray([len(self.reset_seeds)])}, [{"reset_seed": seed}]

        def close(self):
            return None

    env = FakeEnv()
    driver = sf_teacher_drivers.SerialTeacherRolloutDriver([env])

    obs, info = driver.continue_after_life_loss(0)

    assert env.reset_seeds == [None]
    assert obs["obs"].tolist() == [1]
    assert info == {"reset_seed": None}


def test_serial_teacher_replay_starts_each_seed_in_a_fresh_environment(monkeypatch):
    sf_teacher_drivers = _load_sf_teacher_drivers_for_direct_test(monkeypatch)

    class StatefulEnv:
        action_space = object()

        def __init__(self):
            self.resets = 0

        def reset(self, seed=None):
            self.resets += 1
            return {"obs": np.asarray([seed, self.resets])}, [{}]

        def close(self):
            return None

    driver = sf_teacher_drivers.SerialTeacherRolloutDriver(
        [StatefulEnv()], recreate_env=lambda _slot: StatefulEnv()
    )

    driver.reset_slots({0: 42})
    observation, _ = driver.reset_slots({0: 43})[0]

    assert observation["obs"].tolist() == [43, 1]


def test_batched_rollout_driver_continues_life_loss_from_last_step_result(monkeypatch):
    sf_teacher_drivers = _load_sf_teacher_drivers_for_direct_test(monkeypatch)

    class FakeBatchedEnv:
        action_space = object()

        def __init__(self):
            self.unwrapped = self

        def step(self, action):
            return (
                {"obs": np.asarray([[11], [22]])},
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([True, True], dtype=np.bool_),
                np.asarray([False, False], dtype=np.bool_),
                [{"slot": 0}, {"slot": 1}],
            )

        def close(self):
            return None

    driver = sf_teacher_drivers.BatchedTeacherRolloutDriver(FakeBatchedEnv(), 2)
    driver.step([0, 1], [3, 4])

    obs, info = driver.continue_after_life_loss(1)

    assert obs["obs"].tolist() == [[22]]
    assert info == {"slot": 1}


def test_collect_teacher_dataset_handles_single_env_rollout(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

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
            return np.zeros((3, 2, 2), dtype=np.uint8), [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5)}
            ]

        def step(self, action):
            self.step_actions.append(action)
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(1.5),
                np.bool_(True),
                np.bool_(False),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6),
                    }
                ],
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(
        frame_stack=1,
        env_fps=30.0,
        obs_fps=15.0,
        seed=11,
        use_gpu_render=False,
    )

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert not val_rows
    assert fake_env.step_actions == [[1], [1]]
    assert train_rows[0]["action_id"] == 1
    assert Path(train_rows[0]["image"]).exists()
    assert train_rows[0]["env_name"] == "flappy"
    assert train_rows[0]["decision_step"] == 0
    assert train_rows[0]["raw_reward"] == 1.5
    assert metadata["runtime_metadata"]["checkpoint_train_step"] == 123
    assert metadata["runtime_metadata"]["env_fps"] == 30.0
    assert metadata["runtime_metadata"]["obs_fps"] == 15.0


def test_collect_teacher_dataset_probe_retries_until_episode_is_accepted(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
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
            return np.zeros((3, 2, 2), dtype=np.uint8), [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5)}
            ]

        def step(self, action):
            reward = 3000.0 if len(self.reset_seeds) == 1 else 3000.001
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(reward),
                np.bool_(True),
                np.bool_(False),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6),
                    }
                ],
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_actor = FakeActorCritic()
    fake_cfg = SimpleNamespace(frame_stack=1, env_fps=35.0, obs_fps=17.5, seed=11)

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(
        teacher_runtime, "split_batch_actions",
        lambda processed_actions, batch_size, action_space=None: [[0, 1, 0, 0]] * batch_size,
    )

    output_dir = tmp_path / "export"
    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("deadly_corridor"),
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
        episode_return_gt=3000.0,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert not val_rows
    assert fake_env.reset_seeds == [11, 12, 12]
    assert train_rows[0]["episode_idx"] == 0
    assert train_rows[0]["seed"] == 12
    assert train_rows[0]["raw_reward"] == pytest.approx(3000.001)
    assert Path(train_rows[0]["image"]).exists()
    assert "_attempts" not in train_rows[0]["image"]
    assert sorted(path.name for path in (output_dir / "images" / "train" / "episode_000000").glob("*.png")) == [
        "decision_step_000000.png"
    ]
    assert not (output_dir / "images" / "_attempts").exists()
    state = json.loads((output_dir / "probe_state.json").read_text(encoding="utf-8"))
    assert state["next_attempt_idx"] == 2
    assert state["attempted_episodes"] == 2
    assert state["rejected_episodes"] == 1
    assert metadata["attempted_episodes"] == 2
    assert len(metadata["accepted_specs"]) == 1
    assert metadata["rejected_episodes"] == 1


def test_collect_teacher_dataset_probes_without_rgb_then_replays_accepted_seed(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class ProbeEnv:
        reset_seeds: list[int] = []
        transitions: list[tuple[int, object, float, int]] = []
        step_count = 0
        action_space = object()

        def reset(self, seed=None):
            self.seed = int(seed)
            self.reset_seeds.append(int(seed))
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [{"probe": True}]

        def step(self, action):
            self.__class__.step_count += 1
            self.transitions.append((self.seed, action, 1.5, 2))
            return (
                {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"latency_raw_frames": 2, "latency_ms": 66.0}],
            )

        def close(self):
            return None

    class ReplayEnv:
        reset_seeds: list[int] = []
        transitions: list[tuple[int, object, float, int]] = []
        step_count = 0
        action_space = object()

        def reset(self, seed=None):
            self.seed = int(seed)
            self.reset_seeds.append(int(seed))
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(int(seed), frames=4)}
            ]

        def step(self, action):
            self.__class__.step_count += 1
            self.transitions.append((self.seed, action, 1.5, 2))
            return (
                {"obs": np.ones((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(9, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    build_env_calls: list[tuple[bool, int]] = []

    def fake_build_cfg(config):
        return object(), SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=100, use_gpu_render=False)

    def fake_runtime(module, cfg, config, num_envs):
        export_rgb = bool(cfg.export_env_raw_rgb_frames)
        build_env_calls.append((export_rgb, int(num_envs)))
        env_cls = ReplayEnv if export_rgb else ProbeEnv
        return {
            "envs": [env_cls() for _ in range(int(num_envs))],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.ones(int(actions.shape[0]), dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", fake_build_cfg)
    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)

    output_dir = tmp_path / "export"
    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=3,
        replay_num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")

    assert build_env_calls == [(False, 3), (True, 1)]
    assert ProbeEnv.reset_seeds == [100, 101, 102]
    assert ProbeEnv.step_count == 3
    assert ReplayEnv.reset_seeds == [100]
    assert ReplayEnv.step_count == 1
    assert [transition for transition in ProbeEnv.transitions if transition[0] == 100] == ReplayEnv.transitions
    assert [row["seed"] for row in train_rows] == [100]
    assert Path(train_rows[0]["image"]).exists()
    assert metadata["attempted_episodes"] == 1
    assert len(metadata["accepted_specs"]) == 1
    assert metadata["rejected_episodes"] == 0
    state = json.loads((output_dir / "probe_state.json").read_text(encoding="utf-8"))
    assert state["next_attempt_idx"] == 3
    assert state["attempted_episodes"] == 1
    assert state["rejected_episodes"] == 0
    assert state["accepted_specs"] == [
        {
            "attempt_idx": 0,
            "episode_idx": 0,
            "episode_length": 1,
            "episode_raw_frames": 2,
            "episode_raw_return": 1.5,
            "seed": 100,
            "split": "train",
            "termination_reason": "terminated",
        }
    ]
    assert state["runtime_metadata"] == {
        "checkpoint_train_step": 123,
        "env_fps": 30.0,
        "frame_stack": 4,
        "obs_fps": 15.0,
    }


def test_collect_teacher_dataset_handles_single_agent_batched_step_outputs(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

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

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=2,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=2,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert not val_rows
    assert fake_actor.seen_obs_shapes == [(2, 12, 84, 84), (2, 12, 84, 84)]
    assert [env.step_actions for env in fake_envs] == [[[1], [1]], [[1], [1]]]
    assert [row["action_id"] for row in train_rows] == [1, 1]
    assert Path(train_rows[0]["image"]).exists()
    assert train_rows[0]["raw_reward"] == 1.5
    assert train_rows[0]["latency_raw_frames"] == 2
    assert metadata["runtime_metadata"]["checkpoint_train_step"] == 123
    assert metadata["runtime_metadata"]["env_fps"] == 30.0
    assert metadata["runtime_metadata"]["obs_fps"] == 15.0


def test_collect_teacher_dataset_encodes_deadly_corridor_factorized_tuple_actions(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from gymnasium.spaces import Discrete, Tuple
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

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
            return {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)}, [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(self.slot, frames=4)}
            ]

        def step(self, action):
            self.step_actions.append(action)
            return (
                {"obs": np.full((1, 12, 84, 84), self.slot, dtype=np.uint8)},
                np.asarray([2200.0], dtype=np.float32),
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

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    output_dir = tmp_path / "export"
    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("deadly_corridor"),
        output_dir=output_dir,
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
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert not val_rows
    assert [env.step_actions for env in fake_envs] == [
        [[[0, 1, 2, 1]], [[0, 1, 2, 1]]],
        [[[1, 2, 0, 0]], [[1, 2, 0, 0]]],
    ]
    assert [row["action_id"] for row in train_rows] == [11, 30]
    assert [row["action_text"] for row in train_rows] == [
        "MOVE_FORWARD + MOVE_RIGHT + ATTACK",
        "TURN_LEFT + MOVE_BACKWARD",
    ]
    assert metadata["runtime_metadata"]["checkpoint_train_step"] == 123


def test_collect_teacher_dataset_saves_demon_attack_latest_raw_rgb_image(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from PIL import Image
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


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
            obs = {"obs": np.ones((1, 4, 2, 2), dtype=np.uint8)}
            next_stack = np.stack([np.full((3, 2, 3), 50 + idx, dtype=np.uint8) for idx in range(4)], axis=0)
            return (
                obs,
                np.asarray([51.0], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "raw_reward": 8000.0,
                        "lives": 0,
                        "is_true_episode_end": True,
                        "env_raw_rgb_frame_stack": next_stack,
                    }
                ],
            )

        def close(self):
            return None

    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: (object(), SimpleNamespace(frame_stack=4, env_fps=60.0, obs_fps=15.0, seed=0, simulator="cpu")),
    )
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("demon_attack"),
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
        demon_episode_threshold=7999.0,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    image = Image.open(train_rows[0]["image"])
    assert not val_rows
    assert image.mode == "RGB"
    assert image.size == (2, 3)
    assert np.asarray(image)[0, 0].tolist() == [40, 41, 42]


@pytest.mark.parametrize("export_history", [False, True])
def test_collect_teacher_dataset_saves_flappy_latest_raw_rgb_image(monkeypatch, tmp_path, export_history):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from gymnasium.spaces import Discrete
    from PIL import Image
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            assert set(normalized_obs) == {"obs"}
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        action_space = Discrete(2)

        def reset(self, seed=None):
            self.steps = 0
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
            self.steps += 1
            obs = {"obs": np.ones((1, 12, 2, 2), dtype=np.uint8)}
            next_stack = np.stack([np.full((6, 5, 3), 50 + idx, dtype=np.uint8) for idx in range(4)], axis=0)
            return (
                obs,
                np.asarray([1.5], dtype=np.float32),
                np.asarray([self.steps == 2], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"latency_raw_frames": 0, "latency_ms": 0.0, "env_raw_rgb_frame_stack": next_stack}],
            )

        def close(self):
            return None

    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: (
            object(),
            SimpleNamespace(
                frame_stack=4,
                env_fps=30.0,
                obs_fps=15.0,
                seed=0,
                use_gpu_render=False, latency_type="fixed", fixed_latency_ms=0.0,
            ),
        ),
    )
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy") | ({"export_zero_action_history": 8} if export_history else {}),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=4,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    image = Image.open(train_rows[0]["image"])
    assert not val_rows
    assert image.size == (5, 6)
    assert np.asarray(image)[0, 0].tolist() == [40, 41, 42]

    if export_history:
        assert train_rows[0]["state"] == [0.] * 24
        assert train_rows[1]["state"] == [0.] * 21 + [1., 1., 1.]
        assert train_rows[0]["prompt"] == _rollout_descriptor("flappy")["base_prompt"]


def test_collect_teacher_dataset_saves_historical_context_images(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from PIL import Image
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class FakeEnv:
        action_space = object()

        def __init__(self):
            self.step_count = 0

        def reset(self, seed=None):
            self.step_count = 0
            obs = {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}
            return obs, [{"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(10, frames=1)}]

        def step(self, action):
            self.step_count += 1
            obs = {"obs": np.ones((1, 12, 2, 2), dtype=np.uint8)}
            next_value = 10 * (self.step_count + 1)
            return (
                obs,
                np.asarray([1.5], dtype=np.float32),
                np.asarray([self.step_count >= 4], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(next_value, frames=1),
                    }
                ],
            )

        def close(self):
            return None

    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: (
            object(),
            SimpleNamespace(
                frame_stack=1,
                env_fps=30.0,
                obs_fps=15.0,
                seed=0,
                use_gpu_render=False,
            ),
        ),
    )
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"

    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
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
        filter_preset="none",
        context_window=4,
    )
    train_rows = _read_shard_rows(output_dir, "train")

    assert [len(row["context_images"]) for row in train_rows] == [3, 3, 3, 3]

    context_values = []
    current_values = []
    for row in train_rows:
        context_values.append([
            np.asarray(Image.open(path))[0, 0, 0].item()
            for path in row["context_images"]
        ])
        current_values.append(np.asarray(Image.open(row["image"]))[0, 0, 0].item())

    assert context_values == [
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 20],
        [10, 20, 30],
    ]
    assert current_values == [10, 20, 30, 40]


def test_collect_teacher_dataset_uses_batched_driver_for_demon_attack_gpu(monkeypatch, tmp_path):
    episodes = 5
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


    class FakeActorCritic:
        inference_batches = []

        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            self.inference_batches.append(batch_size)
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
            obs = {"obs": np.zeros((3, 4, 2, 2), dtype=np.uint8)}
            infos = [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(agent_idx, frames=4)}
                for agent_idx in range(3)
            ]
            return obs, infos

        def reset_agents(self, reset_plan, *, active_mask=None):
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
                    "raw_reward": 8000.0 + agent_idx,
                    "lives": 0,
                    "is_true_episode_end": True,
                    "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(agent_idx + 20, frames=4),
                }
                for agent_idx in range(3)
            ]
            return (
                obs,
                np.asarray([51.0, 52.0, 53.0], dtype=np.float32),
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
        build_env_calls.append(num_envs)
        assert cfg.gpu_env_batch_size == 3
        return {
            "envs": [fake_env],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: actions.reshape(-1).numpy(),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: list(processed_actions))
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("demon_attack"),
        output_dir=output_dir,
        max_episodes=episodes,
        max_attempts=episodes,
        max_steps_per_episode=8,
        val_fraction=0.0,
        split_seed=0,
        num_envs=3,
        replay_num_envs=3,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert build_env_calls == [1, 1]
    assert not val_rows
    assert [row["episode_idx"] for row in train_rows] == list(range(episodes))
    phase_actions = [[0, 1, 2], [0, 1, 0]]
    assert fake_env.step_actions == phase_actions * 2
    # Inactive slots must not shrink the GPU inference batch between phases.
    assert FakeActorCritic.inference_batches == [3] * (2 * len(phase_actions))


@pytest.mark.parametrize("export_history", [False, True])
def test_collect_teacher_dataset_demon_attack_exports_full_game_across_life_losses(monkeypatch, tmp_path, export_history):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    torch = pytest.importorskip("torch")
    from gymnasium.spaces import Discrete
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


    class FakeActorCritic:
        rnn_inputs: list[list[list[float]]] = []

        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            self.rnn_inputs.append(rnn_states.detach().cpu().numpy().tolist())
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class ScriptedDemonEnv:
        reset_seeds: list[int | None] = []
        step_actions: list[object] = []
        action_space = Discrete(6)

        transitions = [
            {"reward": 10.0, "raw_reward": 1000.0, "lives": 2, "true_end": False},
            {"reward": 20.0, "raw_reward": 3000.0, "lives": 1, "true_end": False},
            {"reward": 30.0, "raw_reward": 4000.0, "lives": 0, "true_end": True},
        ]

        def __init__(self):
            self.life_idx = 0

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            if seed is None:
                self.life_idx += 1
            else:
                self.life_idx = 0
            return (
                {"obs": np.full((1, 4, 2, 2), self.life_idx, dtype=np.uint8)},
                [{"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(100 + self.life_idx, frames=4)}],
            )

        def step(self, action):
            self.step_actions.append(action)
            transition = self.transitions[self.life_idx]
            info = {
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "raw_reward": transition["raw_reward"],
                "lives": transition["lives"],
                "is_true_episode_end": transition["true_end"],
                "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(200 + self.life_idx, frames=4),
            }
            return (
                {"obs": np.full((1, 4, 2, 2), self.life_idx + 10, dtype=np.uint8)},
                np.asarray([transition["reward"]], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [info],
            )

        def close(self):
            return None

    fake_cfg = SimpleNamespace(
        frame_stack=4,
        env_fps=60.0,
        obs_fps=15.0,
        seed=100,
        simulator="cpu",
        use_gpu_render=False, latency_type="fixed", fixed_latency_ms=0.0,
    )

    def fake_runtime(module, cfg, config, num_envs):
        return {
            "envs": [ScriptedDemonEnv() for _ in range(int(num_envs))],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.ones(int(actions.shape[0]), dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)

    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("demon_attack") | ({"export_zero_action_history": 8} if export_history else {}),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=12,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        filter_preset="none",
    )
    train_rows = _read_shard_rows(output_dir, "train")
    state = json.loads((output_dir / "probe_state.json").read_text(encoding="utf-8"))

    assert ScriptedDemonEnv.reset_seeds == [100, None, None, 100, None, None]
    assert [row["decision_step"] for row in train_rows] == [0, 1, 2]
    assert [row["raw_reward"] for row in train_rows] == [10.0, 20.0, 30.0]
    assert state["accepted_specs"] == [
        {
            "attempt_idx": 0,
            "episode_idx": 0,
            "episode_length": 3,
            "episode_raw_frames": 12,
            "episode_raw_return": 8000.0,
            "seed": 100,
            "split": "train",
            "termination_reason": "terminated",
        }
    ]
    assert all(rnn_input == [[0.0]] for rnn_input in FakeActorCritic.rnn_inputs)

    if export_history:
        assert all(row["state"] == [0.] * 88 for row in train_rows)


def test_collect_teacher_dataset_writes_episode_shards_and_resumes(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


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

    fake_cfg = SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=100, use_gpu_render=False)

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

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=2,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")
    val_rows = _read_shard_rows(output_dir, "val")

    assert not val_rows
    assert [row["episode_idx"] for row in train_rows] == [0, 1]
    assert (output_dir / "episode_rows" / "train" / "episode_000000.jsonl").exists()
    assert (output_dir / "episode_rows" / "train" / "episode_000001.jsonl").exists()
    assert FakeEnv.reset_seeds == [100, 101, 100, 101]

    FakeEnv.reset_seeds.clear()

    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=3,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    resumed_train_rows = _read_shard_rows(output_dir, "train")
    resumed_val_rows = _read_shard_rows(output_dir, "val")

    assert not resumed_val_rows
    assert [row["episode_idx"] for row in resumed_train_rows] == [0, 1, 2]
    assert FakeEnv.reset_seeds == [102, 102]


def test_collect_teacher_dataset_skips_runtime_when_probe_and_shards_complete(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    from PIL import Image
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter
    from latency_bench.data.hf_parquet import write_episode_row_shard

    output_dir = tmp_path / "export"
    image_path = output_dir / "images" / "train" / "episode_000000" / "decision_step_000000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
    row = {
        "image": str(image_path),
        "env_name": "flappy",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 100,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 66.0,
        "raw_reward": 1.5,
        "action": [0.0, 1.0],
        "action_id": 1,
        "action_text": "FLAP",
    }
    write_episode_row_shard(output_dir, [row])
    (output_dir / "probe_state.json").write_text(
        json.dumps(
            {
                "next_attempt_idx": 1,
                "attempted_episodes": 1,
                "rejected_episodes": 0,
                "accepted_specs": [
                    {
                        "attempt_idx": 0,
                        "episode_idx": 0,
                        "episode_length": 1,
                        "episode_raw_return": 1.5,
                        "seed": 100,
                        "split": "train",
                    }
                ],
                "runtime_metadata": {
                    "checkpoint_train_step": 123,
                    "frame_stack": 4,
                    "env_fps": 30.0,
                    "obs_fps": 15.0,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: pytest.fail("runtime should not be built when accepted specs and shards are complete"),
    )

    metadata = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=8,
        replay_num_envs=4,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )

    assert metadata["runtime_metadata"]["checkpoint_train_step"] == 123
    assert metadata["runtime_metadata"]["frame_stack"] == 4
    assert metadata["runtime_metadata"]["env_fps"] == 30.0
    assert metadata["runtime_metadata"]["obs_fps"] == 15.0


def test_collect_teacher_dataset_redirects_sample_factory_init_output(monkeypatch, tmp_path, capfd):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    monkeypatch.setattr(logging.getLogger("rl"), "disabled", False)
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter


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
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [
                {"reset": True, "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5, frames=4)}
            ]

        def step(self, action):
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
        return object(), SimpleNamespace(
            frame_stack=4,
            env_fps=30.0,
            obs_fps=15.0,
            seed=0,
            use_gpu_render=False,
        )

    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", fake_build_config)
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)
    output_dir = tmp_path / "export"
    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=1,
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


def test_sample_factory_init_log_restores_logger_handlers_added_inside_context(tmp_path, monkeypatch):
    monkeypatch.setattr(logging.getLogger("rl"), "disabled", False)
    from latency_bench.integrations.sample_factory.teacher_runtime import sample_factory_init_log

    logger = logging.getLogger("rl")
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    leaked_handler = None
    try:
        with sample_factory_init_log(tmp_path):
            leaked_handler = logging.StreamHandler()
            logger.addHandler(leaked_handler)
            logger.warning("first init")

        with sample_factory_init_log(tmp_path):
            logger.warning("second init")

        assert logger.handlers == original_handlers
        assert logger.propagate is original_propagate
        init_log = (tmp_path / "sample_factory_init.log").read_text(encoding="utf-8")
        assert "first init" in init_log
        assert "second init" in init_log
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate
        if leaked_handler is not None:
            leaked_handler.close()


@pytest.mark.parametrize("env_name,sf_env,registration", [
    ("flappy", "latency_flappy", "register_flappy_envs"),
    ("deadly_corridor", "doom_deadly_corridor", "register_deadly_corridor_components"),
])
def test_rollout_cfg_registers_integration_and_parses_evaluation_args(
    monkeypatch, tmp_path, env_name, sf_env, registration,
):
    from sample_factory.cfg import arguments as sf_arguments
    from latency_bench.integrations.sample_factory import teacher_runtime

    registered = []
    parser_args = []
    parsed_cfg = SimpleNamespace(policy_index=0)
    module = SimpleNamespace(
        SAMPLE_FACTORY_CONFIG_PARSER=SimpleNamespace(
            parse_eval=lambda argv: parser_args.extend(argv) or parsed_cfg
        ),
    )
    setattr(module, registration, lambda: registered.append(True))
    monkeypatch.setattr(teacher_runtime, "import_integration_module", lambda config: module)
    monkeypatch.setattr(sf_arguments, "load_from_checkpoint", lambda cfg: cfg)

    result_module, result_cfg = teacher_runtime.build_sample_factory_rollout_cfg({
        "env_name": env_name, "integration_name": env_name,
        "checkpoint_experiment_dir": str(tmp_path / "teacher"), "checkpoint_path": None,
        "flat_cfg": {"algo": "APPO", "env": sf_env, "load_checkpoint_kind": "latest"},
        "device_override": None, "latency_override": None,
    })

    assert result_module is module
    assert result_cfg is parsed_cfg
    assert registered
    assert parser_args[parser_args.index("--mode") + 1] == "eval"


@pytest.mark.parametrize(
    ("num_envs", "runtime_num_envs", "driver_kind"),
    [(1, 1, "serial"), (3, 1, "batched")],
)
def test_flappy_gpu_rollout_uses_requested_render_batch_size(
    monkeypatch,
    tmp_path,
    num_envs,
    runtime_num_envs,
    driver_kind,
):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    from contextlib import nullcontext


    cfg = SimpleNamespace(
        frame_stack=4,
        use_gpu_render=True,
        simulator="gpu",
        gpu_render_batch_size=128,
    )
    runtime = {"envs": [object()]}
    captured = {}

    monkeypatch.setattr(teacher_runtime, "sample_factory_init_log", lambda output_dir: nullcontext())
    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: (object(), cfg),
    )

    def fake_build_env_and_policy(module, rollout_cfg, config, build_num_envs):
        captured["gpu_render_batch_size"] = rollout_cfg.gpu_render_batch_size
        captured["num_envs"] = build_num_envs
        return runtime

    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_build_env_and_policy)
    monkeypatch.setattr(
        teacher_runtime, "SerialTeacherRolloutDriver",
        lambda envs: SimpleNamespace(kind="serial"),
    )
    monkeypatch.setattr(
        teacher_runtime, "BatchedTeacherRolloutDriver",
        lambda env, slot_count: SimpleNamespace(kind="batched"),
    )

    _, _, driver, frame_stack = teacher_runtime.build_rollout_driver_runtime(
        _rollout_descriptor("flappy"),
        output_dir=tmp_path,
        num_envs=num_envs,
        export_env_raw_rgb_frames=True,
    )

    assert captured == {
        "gpu_render_batch_size": num_envs,
        "num_envs": runtime_num_envs,
    }
    assert driver.kind == driver_kind
    assert frame_stack == 4


def test_gymnasium_atlantis_gpu_rollout_uses_one_native_batched_env(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    from contextlib import nullcontext


    cfg = SimpleNamespace(frame_stack=4, device="gpu", cli_args={})
    serial_env = SimpleNamespace(close=lambda: captured.update(serial_closed=True))
    runtime = {"envs": [serial_env]}
    captured = {}

    class FakeAtlantisGpuEnv:
        def __init__(self, **kwargs):
            captured["gpu_env_kwargs"] = kwargs

    monkeypatch.setattr(teacher_runtime, "sample_factory_init_log", lambda output_dir: nullcontext())
    monkeypatch.setattr(
        teacher_runtime, "build_sample_factory_rollout_cfg",
        lambda config: (object(), cfg),
    )

    def fake_build_env_and_policy(module, rollout_cfg, config, build_num_envs):
        captured["runtime_num_envs"] = build_num_envs
        return runtime

    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_build_env_and_policy)
    monkeypatch.setattr(
        teacher_runtime, "_GYMNASIUM_ATARI_GPU_ENVS",
        {"atlantis": FakeAtlantisGpuEnv},
    )
    monkeypatch.setattr(
        teacher_runtime, "BatchedTeacherRolloutDriver",
        lambda env, slot_count: SimpleNamespace(kind="batched", env=env, slot_count=slot_count),
    )

    config = {
        "env_name": "atlantis",
        "integration_name": "gymnasium",
        "flat_cfg": {
            "gym_make_kwargs_json": json.dumps({
                "noop_max": 30,
                "base_make_kwargs": {"max_num_frames_per_episode": 108_000},
            }),
        },
    }
    _, _, driver, frame_stack = teacher_runtime.build_rollout_driver_runtime(
        config,
        output_dir=tmp_path,
        num_envs=128,
        export_env_raw_rgb_frames=False,
    )

    assert captured["runtime_num_envs"] == 1
    assert captured["serial_closed"] is True
    assert captured["gpu_env_kwargs"]["gpu_env_device"] == "cuda"
    assert captured["gpu_env_kwargs"]["gpu_env_batch_size"] == 128
    assert driver.kind == "batched"
    assert driver.slot_count == 128
    assert frame_stack == 4


def test_collect_teacher_dataset_replays_existing_accepted_specs_without_new_probe_steps(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter

    class FakeActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(normalized_obs["obs"].shape[0])
            return {
                "actions": torch.full((batch_size, 1), 1, dtype=torch.int64),
                "new_rnn_states": torch.ones_like(rnn_states),
            }

    class ProbeEnv:
        step_count = 0
        close_count = 0
        action_space = object()

        def reset(self, seed=None):
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [{"probe": True}]

        def step(self, action):
            self.__class__.step_count += 1
            return (
                {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"latency_raw_frames": 2, "latency_ms": 66.0}],
            )

        def close(self):
            self.__class__.close_count += 1

    class ReplayEnv:
        reset_seeds: list[int] = []
        action_space = object()

        def reset(self, seed=None):
            self.seed = int(seed)
            self.reset_seeds.append(int(seed))
            return {"obs": np.zeros((1, 12, 2, 2), dtype=np.uint8)}, [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(int(seed), frames=4)}
            ]

        def step(self, action):
            return (
                {"obs": np.ones((1, 12, 2, 2), dtype=np.uint8)},
                np.asarray([1.5], dtype=np.float32),
                np.asarray([True], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [
                    {
                        "latency_raw_frames": 2,
                        "latency_ms": 66.0,
                        "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(9, frames=4),
                    }
                ],
            )

        def close(self):
            return None

    def fake_build_cfg(config):
        return object(), SimpleNamespace(frame_stack=4, env_fps=30.0, obs_fps=15.0, seed=100, use_gpu_render=False)

    def fake_runtime(module, cfg, config, num_envs):
        env_cls = ReplayEnv if cfg.export_env_raw_rgb_frames else ProbeEnv
        return {
            "envs": [env_cls() for _ in range(int(num_envs))],
            "env_info": object(),
            "actor_critic": FakeActorCritic(),
            "device": torch.device("cpu"),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "argmax_actions": lambda action_distribution: action_distribution,
            "preprocess_actions": lambda env_info, actions: np.ones(int(actions.shape[0]), dtype=np.int64),
            "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim),
            "rnn_size": 1,
            "checkpoint": {"train_step": 123},
        }

    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / "probe_state.json").write_text(
        json.dumps(
            {
                "next_attempt_idx": 1,
                "attempted_episodes": 1,
                "rejected_episodes": 0,
                "accepted_specs": [
                    {
                        "attempt_idx": 0,
                        "episode_idx": 0,
                        "episode_length": 1,
                        "episode_raw_return": 1.5,
                        "seed": 100,
                        "split": "train",
                    }
                ],
                "runtime_metadata": {
                    "checkpoint_train_step": 123,
                    "frame_stack": 4,
                    "env_fps": 30.0,
                    "obs_fps": 15.0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", fake_build_cfg)
    monkeypatch.setattr(teacher_runtime, "build_env_and_policy", fake_runtime)
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)

    exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=1,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=3,
        replay_num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
    )
    train_rows = _read_shard_rows(output_dir, "train")

    assert ProbeEnv.step_count == 0
    assert ProbeEnv.close_count == 0
    assert ReplayEnv.reset_seeds == [100]
    assert [row["seed"] for row in train_rows] == [100]
    assert Path(train_rows[0]["image"]).exists()


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


def test_merge_probe_states_reindexes_dedupes_and_assigns_splits():
    from latency_bench.data.rollout_probe import merge_probe_states, build_episode_plan

    def spec(seed, length):
        return {
            "attempt_idx": seed,
            "episode_idx": 999,
            "episode_length": length,
            "episode_score": float(length),
            "seed": seed,
            "split": "_attempts",
        }

    shard0 = {
        "accepted_specs": [spec(0, 5040), spec(1, 6000), spec(2, 7000)],
        "attempted_episodes": 100,
        "rejected_episodes": 97,
        "next_attempt_idx": 100,
        "runtime_metadata": {"env_fps": 30.0, "obs_fps": 30.0, "frame_stack": 1, "checkpoint_train_step": 7},
    }
    shard1 = {
        "accepted_specs": [spec(1, 6000), spec(100_000_000, 5500), spec(100_000_001, 5800)],
        "attempted_episodes": 90,
        "rejected_episodes": 88,
        "next_attempt_idx": 90,
    }

    merged = merge_probe_states([shard0, shard1], max_episodes=4, val_fraction=0.5, split_seed=0)
    specs = merged["accepted_specs"]

    # Contiguous reindex and deterministic dedupe-by-seed in shard order (seed 1 dropped from shard1).
    assert [s["episode_idx"] for s in specs] == [0, 1, 2, 3]
    assert [s["seed"] for s in specs] == [0, 1, 2, 100_000_000]
    # Original per-episode payload (length/return) preserved.
    assert specs[1]["episode_length"] == 6000
    assert specs[3]["episode_score"] == 5500.0
    # Splits come from the same plan the single-process probe uses.
    plan = {
        p["episode_idx"]: p["split"]
        for p in build_episode_plan(max_episodes=4, val_fraction=0.5, split_seed=0, base_seed=0)
    }
    assert {s["episode_idx"]: s["split"] for s in specs} == plan
    assert sum(1 for s in specs if s["split"] == "val") == round(4 * 0.5)
    # Counters summed; runtime_metadata carried from the first shard that has it.
    assert merged["attempted_episodes"] == 190
    assert merged["rejected_episodes"] == 185
    assert merged["runtime_metadata"]["checkpoint_train_step"] == 7
    # Deterministic given the same inputs.
    again = merge_probe_states([shard0, shard1], max_episodes=4, val_fraction=0.5, split_seed=0)
    assert again["accepted_specs"] == specs


def test_merge_probe_states_raises_when_insufficient():
    from latency_bench.data.rollout_probe import merge_probe_states

    shard = {
        "accepted_specs": [{"seed": 0, "episode_length": 5040, "episode_idx": 0, "split": "_attempts", "attempt_idx": 0}],
        "attempted_episodes": 1,
        "rejected_episodes": 0,
        "next_attempt_idx": 1,
    }
    with pytest.raises(ValueError):
        merge_probe_states([shard], max_episodes=5, val_fraction=0.0, split_seed=0)


def test_collect_teacher_dataset_probe_only_persists_state_without_replay(monkeypatch, tmp_path):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch
    from latency_bench.integrations.sample_factory import teacher_rollout as exporter
    from latency_bench.data.hf_parquet import completed_episode_indices_from_shards

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
            self.reset_seeds: list[int] = []
            self.seed = 0

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            self.seed = int(seed)
            return np.zeros((3, 2, 2), dtype=np.uint8), [
                {"env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(5)}
            ]

        def step(self, action):
            return (
                np.zeros((3, 2, 2), dtype=np.uint8),
                np.float32(1.5 if self.seed % 3 == 0 else 0.5),
                np.bool_(True),
                np.bool_(False),
                [{"latency_raw_frames": 2, "latency_ms": 66.0, "env_raw_rgb_frame_stack": _env_raw_rgb_frame_stack(6)}],
            )

        def close(self):
            return None

    fake_env = FakeEnv()
    fake_cfg = SimpleNamespace(
        frame_stack=1,
        env_fps=30.0,
        obs_fps=15.0,
        seed=11,
        use_gpu_render=False,
    )
    monkeypatch.setattr(teacher_runtime, "build_sample_factory_rollout_cfg", lambda config: (object(), fake_cfg))
    monkeypatch.setattr(
        teacher_runtime, "build_env_and_policy",
        lambda module, cfg, config, num_envs: {
            "envs": [fake_env],
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
    monkeypatch.setattr(teacher_runtime, "normalize_actions_for_space", lambda actions, **kwargs: actions)
    monkeypatch.setattr(teacher_runtime, "split_batch_actions", lambda processed_actions, batch_size, action_space=None: [1] * batch_size)

    output_dir = tmp_path / "shard"
    result = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=12,
        max_attempts=12,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        rollout_seed_offset=1000,
        probe_only=True,
        episode_return_gt=1.0,
    )

    # Probe-only returns its marker and writes the resume state, but does not replay.
    assert len(result["accepted_specs"]) == 4
    assert result["attempted_episodes"] == 12
    assert result["rejected_episodes"] == 8
    state = json.loads((output_dir / "probe_state.json").read_text(encoding="utf-8"))
    assert len(state["accepted_specs"]) == 4
    assert state["attempted_episodes"] == 12
    assert state["rejected_episodes"] == 8
    assert state["next_attempt_idx"] == 12
    # Reset seed carries the rollout_seed_offset (base_seed cfg.seed=11 + offset 1000).
    assert state["accepted_specs"][0]["seed"] == 11 + 1000
    assert fake_env.reset_seeds[0] == 11 + 1000
    resumed = exporter.collect_teacher_dataset(
        _rollout_descriptor("flappy"),
        output_dir=output_dir,
        max_episodes=12,
        max_attempts=12,
        max_steps_per_episode=1,
        val_fraction=0.0,
        split_seed=0,
        num_envs=1,
        deterministic=False,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        rollout_seed_offset=1000,
        probe_only=True,
        episode_return_gt=1.0,
    )
    assert len(resumed["accepted_specs"]) == 4
    assert len(fake_env.reset_seeds) == 12
    # No episode row shards were materialized (replay skipped).
    assert completed_episode_indices_from_shards(output_dir) == set()


@pytest.mark.parametrize("probe_only", [False, True])
def test_teacher_export_reports_persisted_collection_and_materialized_rows(tmp_path, monkeypatch, probe_only):
    """Long-lived export contract: resume decisions, final reports and embedded storage agree."""
    from PIL import Image
    from latency_bench.integrations.sample_factory import teacher_export as sf_teacher_export
    from latency_bench.data.ghost_trail import GhostTrailConfig
    from latency_bench.data.hf_parquet import write_episode_row_shard, load_parquet_splits
    from latency_bench.data.rollout_probe import write_probe_state
    from datasets import DatasetDict

    uploads = []
    monkeypatch.setattr(DatasetDict, "push_to_hub", lambda self, **kwargs: uploads.append(kwargs))
    monkeypatch.setattr(sf_teacher_export, "resolve_token", lambda token: None)

    output = tmp_path / "export"
    output.mkdir()
    image = output / "image.png"
    Image.fromarray(np.full((2, 2, 3), 42, dtype=np.uint8)).save(image)
    row = {"image": str(image), "context_images": [str(image)] * 3,
           "env_name": "flappy", "episode_idx": 0, "decision_step": 0,
           "split": "train", "seed": 11, "prompt": "flap", "latency_raw_frames": 0,
           "latency_ms": 0.0, "raw_reward": 5.0, "action_id": 1, "state": [0.0] * 24,
           "action": [0.0, 1.0], "action_text": "FLAP"}
    write_episode_row_shard(output, [row])
    write_probe_state(output, {
        "next_attempt_idx": 3, "attempted_episodes": 3, "rejected_episodes": 2,
        "accepted_specs": [{"episode_idx": 0, "attempt_idx": 2, "seed": 11,
                            "episode_length": 1, "episode_raw_frames": 2,
                            "episode_raw_return": 5.0, "split": "train"}],
        "runtime_metadata": {"checkpoint_train_step": 123, "frame_stack": 4,
                             "env_fps": 30.0, "obs_fps": 15.0},
    })
    descriptor = _rollout_descriptor("flappy") | {"flat_cfg": {}}
    monkeypatch.setattr(sf_teacher_export, "load_rollout_training_config",
                        lambda *args, **kwargs: (descriptor, {"latency_source": "checkpoint"}))
    metadata = sf_teacher_export.export_sample_factory_teacher_dataset(
        checkpoint_root=tmp_path / "checkpoint", output_dir=output, latency_config_path=None,
        env_name="", max_episodes=1, max_attempts=3, max_steps_per_episode=10,
        val_fraction=0.0, split_seed=0, num_envs=1, replay_num_envs=1, device="cpu",
        deterministic=True, export_image_size=None, image_writer_workers=0, row_buffer_size=2,
        push_to_hub=True, hf_repo_id="organization/data", hf_private=False, hf_token="", hf_max_shard_size="500MB",
        hf_config_name="flappy", hf_data_dir="latency-aware/flappy/teacher-rollouts",
        filter_preset="none", context_window=4, episode_return_gt=0.0, probe_only=probe_only,
        export_zero_action_history=8,
        image_transform_config=GhostTrailConfig(image_transform="flappy_ghost_trail"),
    )
    assert metadata["env_name"] == "flappy"
    assert metadata["episode_return_gate"] == {"field": "episode_raw_return", "operator": ">", "threshold": 0.0}
    if probe_only:
        assert uploads == []
        assert metadata["probe_only"] is True
        assert metadata["accepted_episodes"] == 1
        assert metadata["attempted_episodes"] == 3
        assert metadata["rejected_episodes"] == 2
        assert metadata["attempt_budget"] == 3
        assert (output / "probe_state.json").exists()
        assert not (output / "train.parquet").exists()
        return
    assert uploads[0]["repo_id"] == "organization/data"
    assert uploads[0]["config_name"] == "flappy"
    assert uploads[0]["data_dir"] == "latency-aware/flappy/teacher-rollouts"
    assert json.loads((output / "metadata.json").read_text()) == metadata
    report = json.loads((output / "filter_report.json").read_text())
    assert metadata["filter_report"] == report
    assert report == {
        "preset": "none", "attempted_episodes": 3, "accepted_episodes": 1,
        "rejected_episodes": 2, "accepted_train_episodes": 1, "accepted_val_episodes": 0,
        "train_rows_before_row_filter": 1, "train_rows_after_row_filter": 1,
        "val_rows_before_row_filter": 0, "val_rows_after_row_filter": 0,
        "episode_return_gate": metadata["episode_return_gate"],
    }
    assert metadata["action_history_decisions"] == 8
    assert metadata["prompt_source"] == "sample_factory_checkpoint:base_prompt"
    assert metadata["checkpoint_train_step"] == 123
    assert metadata["frame_stack"] == 4
    assert (metadata["env_fps"], metadata["obs_fps"], metadata["obs_stride_raw_frames"]) == (30.0, 15.0, 2)
    assert metadata["max_steps_unit"] == "raw_frame"
    assert metadata["replay_num_envs"] == 1
    assert metadata["image_source"] == "env_raw_rgb_flappy_ghost_trail"
    assert metadata["image_resolution"] == [2, 2, 3]
    assert (metadata["context_window"], metadata["context_image_count"]) == (4, 3)
    assert metadata["action_layout"] == "flappy_discrete"
    assert (metadata["train_rows"], metadata["val_rows"], metadata["total_rows"]) == (1, 0, 1)
    assert (metadata["train_episodes"], metadata["val_episodes"], metadata["total_episodes"]) == (1, 0, 1)
    splits = load_parquet_splits(output)
    assert len(splits["train"]) == 1 and len(splits["val"]) == 0
    assert not (output / "probe_state.json").exists()
    assert not (output / "episode_rows").exists()
