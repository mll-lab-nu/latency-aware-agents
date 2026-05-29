from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np


def test_flappy_teacher_export_minimal_cfg_uses_checkpoint_visual_settings(tmp_path):
    from scripts.export_sf_teacher_dataset import _minimal_export_cfg

    class FakeFlappyIntegration:
        SF_ENV_NAME = "latency_flappy"

        @staticmethod
        def parse_cfg(cli_args, evaluation):
            return SimpleNamespace(cli_args=cli_args, evaluation=evaluation)

    experiment_dir = tmp_path / "flappy_sf_same_latency_l0_seed0"
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "env": "latency_flappy",
                "device": "cpu",
                "gym_id": "FlappyBird-v0",
                "screen_size": "288,512",
                "policy_screen_size": "96,96",
                "use_gpu_render": True,
                "gpu_render_device": "cuda",
                "gpu_render_batch_size": 8,
                "gpu_render_profile": True,
                "gpu_render_profile_interval": 7,
                "frame_stack": 4,
                "use_lidar": False,
                "normalize_obs": True,
                "audio_on": False,
                "pipe_gap": 120,
                "bird_color": "red",
                "pipe_color": "green",
                "background": "night",
                "score_limit": 42,
                "debug": False,
                "max_episode_steps": 10000,
            }
        ),
        encoding="utf-8",
    )

    cfg = _minimal_export_cfg(
        FakeFlappyIntegration,
        experiment_dir=experiment_dir,
        device_override=None,
    )

    assert cfg.evaluation is True
    expected_flags = {
        "--env": "latency_flappy",
        "--device": "cpu",
        "--gym_id": "FlappyBird-v0",
        "--screen_size": "288,512",
        "--policy_screen_size": "96,96",
        "--use_gpu_render": "True",
        "--gpu_render_device": "cuda",
        "--gpu_render_batch_size": "8",
        "--gpu_render_profile": "True",
        "--gpu_render_profile_interval": "7",
        "--frame_stack": "4",
        "--use_lidar": "False",
        "--normalize_obs": "True",
        "--audio_on": "False",
        "--pipe_gap": "120",
        "--bird_color": "red",
        "--pipe_color": "green",
        "--background": "night",
        "--score_limit": "42",
        "--debug": "False",
        "--max_episode_steps": "10000",
    }
    for flag, value in expected_flags.items():
        assert cfg.cli_args[cfg.cli_args.index(flag) + 1] == value


def test_flappy_teacher_export_render_source_uses_rendered_rgb_frame():
    from scripts.export_sf_teacher_dataset import _frame_for_image_source

    class RenderEnv:
        def render(self):
            return np.full((2, 3, 3), 9, dtype=np.uint8)

    obs = np.zeros((3, 2, 3), dtype=np.uint8)

    frame = _frame_for_image_source(
        obs=obs,
        env=RenderEnv(),
        env_name="flappy",
        frame_stack=1,
        image_source="render",
    )

    assert np.all(frame == 9)


def test_flappy_teacher_export_observation_source_does_not_render():
    from scripts.export_sf_teacher_dataset import _frame_for_image_source

    class RenderEnv:
        def render(self):
            raise AssertionError("observation image source should not render")

    obs = np.full((3, 2, 3), 7, dtype=np.uint8)

    frame = _frame_for_image_source(
        obs=obs,
        env=RenderEnv(),
        env_name="flappy",
        frame_stack=1,
        image_source="observation",
    )

    assert np.all(frame == 7)


def test_flappy_teacher_export_collects_replacement_episodes(monkeypatch, tmp_path):
    import torch

    import scripts.export_sf_teacher_dataset as exporter

    class FakeEnv:
        def reset(self, *, seed=None):
            return np.zeros((1, 3, 2, 2), dtype=np.uint8), {}

        def step(self, action):
            return (
                np.zeros((1, 3, 2, 2), dtype=np.uint8),
                np.array([1.0], dtype=np.float32),
                np.array([True]),
                np.array([False]),
                [{"queue_latency": 0}],
            )

        def close(self):
            return None

    class FakeActor:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(rnn_states.shape[0])
            return {
                "actions": torch.zeros((batch_size, 1), dtype=torch.int64),
                "new_rnn_states": torch.zeros_like(rnn_states),
            }

    def fake_runtime(module, cfg, env_name, num_envs):
        return {
            "actor_critic": FakeActor(),
            "argmax_actions": None,
            "device": torch.device("cpu"),
            "env_info": SimpleNamespace(
                action_space=None,
                all_discrete=True,
                gpu_actions=False,
            ),
            "envs": [FakeEnv() for _ in range(int(num_envs))],
            "preprocess_actions": lambda env_info, actions: actions.cpu().numpy(),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "unsqueeze_tensor": lambda actions, dim: torch.unsqueeze(actions, dim),
            "rnn_size": 1,
        }

    monkeypatch.setattr(
        exporter,
        "_prepare_export_cfg",
        lambda module, env_name, base_cfg, latency, jitter_spec: base_cfg,
    )
    monkeypatch.setattr(exporter, "_build_env_and_policy", fake_runtime)
    monkeypatch.setattr(
        exporter,
        "_frame_for_image_source",
        lambda obs, env, env_name, frame_stack, image_source: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    train_rows, val_rows, metadata = exporter._collect_latency_dataset_batched(
        env_name="flappy",
        module=SimpleNamespace(),
        base_cfg=SimpleNamespace(eval_raw_reward=False, frame_stack=1, seed=0),
        latency=0,
        episode_count=5,
        val_indices={1},
        max_steps_per_episode=1,
        prompt="play flappy",
        mixed_prompt_template="{prompt}",
        dataset_mode="fixed_latency",
        output_dir=tmp_path,
        global_episode_start=0,
        deterministic=False,
        export_image_size=None,
        image_source="observation",
        jitter_spec=None,
        num_envs=3,
        image_writer_workers=0,
        row_buffer_size=8,
    )

    episode_ids = {int(row["episode_idx"]) for row in train_rows + val_rows}

    assert episode_ids == {0, 1, 2, 3, 4}
    assert metadata["action_labels"] == {0: "NOOP", 1: "FLAP"}
    assert len(train_rows) == 4
    assert len(val_rows) == 1


def test_flappy_gpu_teacher_export_collects_inner_batched_agents(monkeypatch, tmp_path):
    import torch

    import scripts.export_sf_teacher_dataset as exporter

    class FakeGpuEnv:
        num_agents = 3

        def __init__(self):
            self.active_mask = [False, False, False]
            self.rendered_agents: list[int] = []

        def reset(self, *, seed=None):
            return {"obs": np.zeros((3, 3, 2, 2), dtype=np.uint8)}, [{} for _ in range(3)]

        def reset_agents(self, reset_plan, *, active_mask):
            self.active_mask = list(active_mask)
            return {"obs": np.zeros((3, 3, 2, 2), dtype=np.uint8)}, [{} for _ in range(3)]

        def set_auto_reset_done(self, enabled):
            self.auto_reset_done = bool(enabled)

        def set_active_agents(self, active_mask):
            self.active_mask = list(active_mask)

        def step(self, actions):
            active = np.asarray(self.active_mask, dtype=np.bool_)
            return (
                {"obs": np.zeros((3, 3, 2, 2), dtype=np.uint8)},
                active.astype(np.float32),
                active.copy(),
                np.zeros(3, dtype=np.bool_),
                [{"queue_latency": 0} for _ in range(3)],
            )

        def render(self, agent_idx=0):
            self.rendered_agents.append(int(agent_idx))
            return np.full((2, 2, 3), int(agent_idx), dtype=np.uint8)

        def close(self):
            return None

    class FakeActor:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(rnn_states.shape[0])
            return {
                "actions": torch.zeros((batch_size, 1), dtype=torch.int64),
                "new_rnn_states": torch.zeros_like(rnn_states),
            }

        def action_distribution(self):
            return None

    fake_env = FakeGpuEnv()

    def fake_runtime(module, cfg, env_name, num_envs):
        assert int(num_envs) == 1
        assert int(cfg.gpu_render_batch_size) == 3
        return {
            "actor_critic": FakeActor(),
            "argmax_actions": lambda distribution: torch.zeros((3, 1), dtype=torch.int64),
            "device": torch.device("cpu"),
            "env_info": SimpleNamespace(
                action_space=None,
                all_discrete=True,
                gpu_actions=False,
            ),
            "envs": [fake_env],
            "preprocess_actions": lambda env_info, actions: actions.cpu().numpy(),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "unsqueeze_tensor": lambda actions, dim: torch.unsqueeze(actions, dim),
            "rnn_size": 1,
        }

    monkeypatch.setattr(exporter, "_build_env_and_policy", fake_runtime)

    train_rows, val_rows, metadata = exporter._collect_flappy_gpu_latency_dataset_batched(
        module=SimpleNamespace(),
        cfg=SimpleNamespace(
            eval_raw_reward=False,
            frame_stack=1,
            seed=0,
            use_gpu_render=True,
            gpu_render_batch_size=1,
        ),
        latency=0,
        episode_count=5,
        val_indices={1},
        max_steps_per_episode=1,
        prompt="play flappy",
        mixed_prompt_template="{prompt}",
        dataset_mode="fixed_latency",
        output_dir=tmp_path,
        global_episode_start=0,
        deterministic=False,
        export_image_size=None,
        image_source="render",
        num_envs=3,
        image_writer_workers=0,
        row_buffer_size=8,
    )

    episode_ids = {int(row["episode_idx"]) for row in train_rows + val_rows}

    assert episode_ids == {0, 1, 2, 3, 4}
    assert set(fake_env.rendered_agents) == {0, 1, 2}
    assert metadata["action_labels"] == {0: "NOOP", 1: "FLAP"}
    assert len(train_rows) == 4
    assert len(val_rows) == 1
