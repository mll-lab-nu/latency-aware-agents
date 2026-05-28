from __future__ import annotations

from latency_bench.core.config import load_config
from scripts.generate_sample_factory_same_latency_sweep import _build_eval_config, _build_train_config
from training.sample_factory_common.config_adapter import (
    apply_training_overrides,
    build_common_sf_args,
    load_training_config,
)


def test_sample_factory_training_configs_load():
    cfg = load_training_config("configs/experiments/demon_attack/sf_fixed_latency_4.yaml")
    assert cfg["backend"]["type"] == "sample_factory"
    assert cfg["env"]["name"] == "demon_attack"
    assert cfg["latency"]["train_values"] == [4]


def test_build_common_sf_args_emits_non_zero_latency_and_eval_settings():
    cfg = load_config("configs/experiments/deadly_corridor/sf_fixed_latency_4.yaml")
    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")
    assert "--queue-latency-values" in args
    assert "4" in args
    assert "--eval-queue-latency-values" in args
    assert "--experiment" in args
    assert "deadly_corridor_sf_fixed_latency_4" in args


def test_apply_training_overrides_updates_run_name_train_dir_and_mode():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    updated = apply_training_overrides(
        cfg,
        run_name="override_run",
        train_dir="tmp/sample_factory",
        mode="train",
    )
    assert updated["experiment"]["name"] == "override_run"
    assert updated["backend"]["train_dir"] == "tmp/sample_factory"
    assert updated["backend"]["run_mode"] == "train"


def test_queue_latency_wrapper_exposes_fixed_non_zero_latency():
    import gymnasium as gym
    from training.sample_factory_common.latency_wrapper import QueueLatencyWrapper

    env = gym.make("CartPole-v1")
    wrapped = QueueLatencyWrapper(
        env,
        queue_latency_values=[4],
        latency_mode="episode_fixed",
        latency_info_keys=("queue_latency",),
    )
    _, info = wrapped.reset(seed=123)
    assert info["queue_latency"] == 4
    _, _, _, _, step_info = wrapped.step(0)
    assert step_info["queue_latency"] == 4
    wrapped.close()


def test_action_latency_wrapper_distribution_samples_deterministically():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    class NeverDoneEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 15})
    first = ActionLatencyWrapper(
        NeverDoneEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=123,
    )
    second = ActionLatencyWrapper(
        NeverDoneEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=123,
    )
    other = ActionLatencyWrapper(
        NeverDoneEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=124,
    )

    try:
        first.reset()
        second.reset()
        other.reset()
        first_values = [first.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        second_values = [second.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        other_values = [other.step(0)[4]["latency_raw_frames"] for _ in range(20)]
    finally:
        first.close()
        second.close()
        other.close()

    assert first_values == second_values
    assert first_values != other_values
    assert all(0 <= value <= 15 for value in first_values)


def test_action_latency_wrapper_distribution_resets_episode_rng():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    class NeverDoneEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    wrapped = ActionLatencyWrapper(
        NeverDoneEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=parse_latency_distribution({"type": "uniform", "min": 0, "max": 15}),
        seed=99,
    )

    try:
        wrapped.reset()
        first_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        wrapped.reset()
        second_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
    finally:
        wrapped.close()

    assert first_episode != second_episode


def test_action_latency_wrapper_distribution_rejects_too_small_pending_action_bound():
    import gymnasium as gym
    import pytest

    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    env = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="max_pending_actions"):
        ActionLatencyWrapper(
            env,
            fixed_latency_raw_frames=0,
            env_fps=30,
            obs_fps=30,
            latency_distribution_spec=parse_latency_distribution({"type": "uniform", "min": 0, "max": 5}),
            max_pending_actions=5,
        )
    env.close()


def test_deadly_corridor_teacher_export_cfg_uses_raw_frame_latency_ms():
    from scripts.export_sf_teacher_dataset import _prepare_deadly_corridor_export_cfg
    from training.sample_factory_common.common import maybe_set_cli_override
    from training.sample_factory_common.latency_config import raw_frames_to_latency_ms

    class FakeDeadlyCorridorIntegration:
        @staticmethod
        def _prepare_eval_cfg(base_cfg, latency_value):
            cfg = SimpleNamespace(**vars(base_cfg))
            maybe_set_cli_override(cfg, "latency_type", "zero" if int(latency_value) == 0 else "fixed")
            maybe_set_cli_override(
                cfg,
                "fixed_latency_ms",
                raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps"))),
            )
            maybe_set_cli_override(cfg, "env_frameskip", 1)
            maybe_set_cli_override(cfg, "eval_env_frameskip", 1)
            maybe_set_cli_override(cfg, "fps", 35)
            maybe_set_cli_override(cfg, "max_episode_steps", int(getattr(base_cfg, "eval_max_steps")))
            return cfg

    base_cfg = SimpleNamespace(
        env="doom_deadly_corridor",
        experiment="deadly_corridor_sf_same_latency_l0_seed0",
        train_dir="runs",
        env_fps=60,
        obs_fps=15,
        env_frameskip=4,
        eval_env_frameskip=4,
        no_render=False,
        save_video=True,
        fps=0,
        eval_deterministic=True,
        eval_raw_reward=False,
        eval_max_steps=100,
        max_episode_steps=0,
        cli_args={},
    )

    cfg = _prepare_deadly_corridor_export_cfg(
        FakeDeadlyCorridorIntegration,
        base_cfg,
        latency_raw_frames=4,
        load_checkpoint=False,
        jitter_spec=None,
    )

    assert cfg.latency_type == "fixed"
    assert cfg.fixed_latency_ms == pytest.approx(4 * 1000.0 / 60.0)
    assert cfg.env_frameskip == 1
    assert cfg.eval_env_frameskip == 1
    assert cfg.fps == 35
    assert cfg.max_episode_steps == 100


def test_deadly_corridor_teacher_export_cfg_uses_uniform_latency_distribution():
    from scripts.export_sf_teacher_dataset import JitterLatencySpec, _prepare_deadly_corridor_export_cfg
    from training.sample_factory_common.common import maybe_set_cli_override
    from training.sample_factory_common.latency_config import raw_frames_to_latency_ms

    class FakeDeadlyCorridorIntegration:
        @staticmethod
        def _prepare_eval_cfg(base_cfg, latency_value):
            cfg = SimpleNamespace(**vars(base_cfg))
            maybe_set_cli_override(cfg, "latency_type", "zero" if int(latency_value) == 0 else "fixed")
            maybe_set_cli_override(
                cfg,
                "fixed_latency_ms",
                raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps"))),
            )
            return cfg

    cfg = _prepare_deadly_corridor_export_cfg(
        FakeDeadlyCorridorIntegration,
        SimpleNamespace(env_fps=60, cli_args={}),
        latency_raw_frames=0,
        load_checkpoint=False,
        jitter_spec=JitterLatencySpec(
            distribution="uniform",
            mean=None,
            std=None,
            min_latency=2,
            max_latency=5,
        ),
    )

    assert cfg.latency_type == "uniform_distribution"
    assert cfg.min_latency_ms == pytest.approx(2 * 1000.0 / 60.0)
    assert cfg.max_latency_ms == pytest.approx(5 * 1000.0 / 60.0)
    assert cfg.add_latency_info is True


def test_deadly_corridor_teacher_export_cfg_uses_normal_latency_distribution():
    from scripts.export_sf_teacher_dataset import JitterLatencySpec, _prepare_deadly_corridor_export_cfg
    from training.sample_factory_common.common import maybe_set_cli_override
    from training.sample_factory_common.latency_config import raw_frames_to_latency_ms

    class FakeDeadlyCorridorIntegration:
        @staticmethod
        def _prepare_eval_cfg(base_cfg, latency_value):
            cfg = SimpleNamespace(**vars(base_cfg))
            maybe_set_cli_override(cfg, "latency_type", "zero" if int(latency_value) == 0 else "fixed")
            maybe_set_cli_override(
                cfg,
                "fixed_latency_ms",
                raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps"))),
            )
            return cfg

    cfg = _prepare_deadly_corridor_export_cfg(
        FakeDeadlyCorridorIntegration,
        SimpleNamespace(env_fps=60, cli_args={}),
        latency_raw_frames=0,
        load_checkpoint=False,
        jitter_spec=JitterLatencySpec(
            distribution="normal",
            mean=3.0,
            std=1.5,
            min_latency=0,
            max_latency=6,
        ),
    )

    assert cfg.latency_type == "normal_distribution"
    assert cfg.mean_latency_ms == pytest.approx(3 * 1000.0 / 60.0)
    assert cfg.std_latency_ms == pytest.approx(1.5 * 1000.0 / 60.0)
    assert cfg.min_latency_ms == pytest.approx(0.0)
    assert cfg.max_latency_ms == pytest.approx(6 * 1000.0 / 60.0)
    assert cfg.add_latency_info is True


def test_deadly_corridor_teacher_export_jitter_plan_uses_max_episodes():
    from scripts.export_sf_teacher_dataset import _build_plan

    args = SimpleNamespace(
        dataset_mode="jitter_latency",
        max_episodes=100,
        val_fraction=0.2,
        fixed_latency=4,
        latency_start=0,
        latency_end=0,
        episodes_per_latency=0,
        val_episodes_per_latency=0,
    )

    assert _build_plan(args) == [(0, 100, 20)]


def test_deadly_corridor_teacher_export_finds_normal_jitter_checkpoint_dir(tmp_path):
    from scripts.export_sf_teacher_dataset import JitterLatencySpec, _find_best_checkpoint_name, _find_experiment_dir

    experiment_dir = tmp_path / "deadly_corridor_sf_latency_normal_mean_2_std_0p5_min_0_max_15_seed0"
    checkpoint_dir = experiment_dir / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    best_checkpoint = checkpoint_dir / "best_000011204_22945792_reward_1107.689.pth"
    best_checkpoint.touch()
    jitter_spec = JitterLatencySpec(
        distribution="normal",
        mean=2.0,
        std=0.5,
        min_latency=0,
        max_latency=15,
    )

    assert _find_experiment_dir(tmp_path, "deadly_corridor", 0, 0, jitter_spec) == experiment_dir
    assert _find_experiment_dir(experiment_dir, "deadly_corridor", 0, 0, jitter_spec) == experiment_dir
    assert _find_best_checkpoint_name(experiment_dir) == best_checkpoint.name


def test_deadly_corridor_teacher_export_finds_uniform_jitter_checkpoint_dir(tmp_path):
    from scripts.export_sf_teacher_dataset import JitterLatencySpec, _find_experiment_dir

    experiment_dir = tmp_path / "deadly_corridor_sf_latency_uniform_min_0_max_4_seed0"
    experiment_dir.mkdir()
    jitter_spec = JitterLatencySpec(
        distribution="uniform",
        mean=None,
        std=None,
        min_latency=0,
        max_latency=4,
    )

    assert _find_experiment_dir(tmp_path, "deadly_corridor", 0, 0, jitter_spec) == experiment_dir


def test_deadly_corridor_teacher_export_finds_fixed_latency_checkpoint_dir(tmp_path):
    from scripts.export_sf_teacher_dataset import _find_experiment_dir

    experiment_dir = tmp_path / "deadly_corridor_sf_same_latency_l4_seed0"
    experiment_dir.mkdir()

    assert _find_experiment_dir(tmp_path, "deadly_corridor", 4, 0, None) == experiment_dir


def test_deadly_corridor_teacher_export_selects_single_action_without_batch_axis():
    import numpy as np

    from scripts.export_sf_teacher_dataset import _select_batch_item

    actions = np.array([[0, 1, 0, 0], [2, 0, 1, 0]], dtype=np.int32)

    selected = _select_batch_item(actions, 1)

    assert selected.shape == (4,)
    assert selected.tolist() == [2, 0, 1, 0]


def test_deadly_corridor_teacher_export_keeps_tuple_action_components_in_single_env_batch():
    import torch

    from scripts.export_sf_teacher_dataset import _ensure_batched_policy_actions, _select_batch_item

    action_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(2),
        )
    )
    action = torch.tensor([2, 1, 0, 1], dtype=torch.int64)

    batched = _ensure_batched_policy_actions(action, action_space)
    selected = _select_batch_item(batched.cpu().numpy(), 0)

    assert tuple(batched.shape) == (1, 4)
    assert selected.shape == (4,)
    assert selected.tolist() == [2, 1, 0, 1]


def test_deadly_corridor_teacher_export_unflattens_tuple_argmax_actions_for_multiple_envs():
    import torch

    from scripts.export_sf_teacher_dataset import _ensure_batched_policy_actions, _select_batch_item

    action_space = gym.spaces.Tuple(
        (
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(2),
        )
    )
    flattened_actions = torch.tensor([[2, 1, 1, 2, 0, 1, 1, 0]], dtype=torch.int64)

    batched = _ensure_batched_policy_actions(flattened_actions, action_space)
    first = _select_batch_item(batched.cpu().numpy(), 0)
    second = _select_batch_item(batched.cpu().numpy(), 1)

    assert tuple(batched.shape) == (2, 4)
    assert first.tolist() == [2, 1, 0, 1]
    assert second.tolist() == [1, 2, 1, 0]


def test_deadly_corridor_teacher_export_preserves_agent_axis_for_batched_env_step():
    import numpy as np

    from scripts.export_sf_teacher_dataset import _select_env_step_action

    processed_actions = np.array([[2, 1, 0, 1], [0, 2, 1, 0]], dtype=np.int32)

    selected = _select_env_step_action(processed_actions, 0)

    assert selected.shape == (1, 4)
    assert selected[0].tolist() == [2, 1, 0, 1]


def test_deadly_corridor_teacher_export_uses_render_frames():
    from scripts.export_sf_teacher_dataset import _frame_for_image_source, _render_mode_for_image_source

    class RenderEnv:
        def render(self):
            return np.full((2, 3, 3), 9, dtype=np.uint8)

    obs = np.zeros((3, 2, 3), dtype=np.uint8)

    assert _render_mode_for_image_source("render") == "rgb_array"
    assert np.all(_frame_for_image_source(obs, RenderEnv(), 1, "render") == 9)


def test_deadly_corridor_teacher_export_observation_frames_do_not_request_render():
    from scripts.export_sf_teacher_dataset import _frame_for_image_source, _render_mode_for_image_source

    class RenderEnv:
        def render(self):
            raise AssertionError("observation image source should not render")

    obs = np.full((3, 2, 3), 7, dtype=np.uint8)

    assert _render_mode_for_image_source("observation") is None
    assert np.all(_frame_for_image_source(obs, RenderEnv(), 1, "observation") == 7)


def test_deadly_corridor_teacher_export_collects_replacement_episodes(monkeypatch, tmp_path):
    import torch

    import scripts.export_sf_teacher_dataset as exporter

    class FakeEnv:
        def __init__(self):
            self.action_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(3),
                    gym.spaces.Discrete(3),
                    gym.spaces.Discrete(3),
                    gym.spaces.Discrete(2),
                )
            )

        def reset(self, *, seed=None):
            return {"obs": np.zeros((1, 3, 2, 2), dtype=np.uint8)}, {}

        def step(self, action):
            return {"obs": np.zeros((1, 3, 2, 2), dtype=np.uint8)}, np.array([0.0]), np.array([False]), np.array([False]), [{}]

        def close(self):
            pass

    class FakeActor:
        def __call__(self, normalized_obs, rnn_states):
            batch_size = int(rnn_states.shape[0])
            actions = torch.tensor([[0, 0, 0, 0] for _ in range(batch_size)], dtype=torch.int64)
            return {"actions": actions, "new_rnn_states": torch.zeros_like(rnn_states)}

    def fake_runtime(module, cfg, num_envs, image_source):
        return {
            "actor_critic": FakeActor(),
            "argmax_actions": None,
            "device": torch.device("cpu"),
            "env_info": SimpleNamespace(
                action_space=FakeEnv().action_space,
                all_discrete=True,
                gpu_actions=False,
            ),
            "envs": [FakeEnv() for _ in range(int(num_envs))],
            "preprocess_actions": lambda env_info, actions: actions.cpu().numpy(),
            "prepare_and_normalize_obs": lambda actor_critic, obs_batch: obs_batch,
            "rnn_size": 1,
        }

    monkeypatch.setattr(
        exporter,
        "_prepare_deadly_corridor_export_cfg",
        lambda module, base_cfg, latency_raw_frames, load_checkpoint, jitter_spec: base_cfg,
    )
    monkeypatch.setattr(exporter, "_build_env_and_policy", fake_runtime)
    monkeypatch.setattr(exporter, "_deadly_action_labels", lambda env: {0: "NOOP"})
    monkeypatch.setattr(exporter, "_frame_for_image_source", lambda obs, env, frame_stack, image_source: np.zeros((2, 2, 3), dtype=np.uint8))

    train_rows, val_rows, _ = exporter._collect_deadly_corridor_dataset(
        module=SimpleNamespace(),
        base_cfg=SimpleNamespace(eval_raw_reward=False, frame_stack=1, env_fps=60, seed=0),
        latency=0,
        episode_count=5,
        val_indices={1},
        max_steps_per_episode=1,
        prompt="play",
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
    assert len(train_rows) + len(val_rows) == 5


def test_action_latency_wrapper_advances_obs_stride_with_one_frame_noop():
    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    base_env = _ActionTraceEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        fixed_latency_raw_frames=3,
        env_fps=60,
        obs_fps=10,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        add_latency_info=True,
    )

    try:
        wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(1)
    finally:
        wrapped.close()

    assert obs.tolist() == [6]
    assert reward == pytest.approx(6.0)
    assert not terminated
    assert not truncated
    assert base_env.actions == [0, 0, 0, 1, 0, 0]
    assert info["latency_raw_frames"] == 3
    assert info["obs_stride_raw_frames"] == 6
    assert info["raw_frames_advanced"] == 6
    applied_events = wrapped.action_scheduler.get_applied_events()
    assert len(applied_events) == 1
    assert applied_events[0].obs_env_step == 0
    assert applied_events[0].ready_raw_frame == 3
    assert applied_events[0].apply_raw_frame == 3


def test_action_latency_wrapper_hold_policy_holds_for_observation_stride():
    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    base_env = _ActionTraceEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=10,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        add_latency_info=True,
    )

    try:
        wrapped.reset()
        _, reward, _, _, info = wrapped.step(1)
    finally:
        wrapped.close()

    assert reward == pytest.approx(3.0)
    assert base_env.actions == [1, 1, 1]
    assert info["obs_stride_raw_frames"] == 3


def test_action_latency_wrapper_sums_raw_reward_across_observation_stride():
    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper, RewardTraceWrapper

    wrapped = ActionLatencyWrapper(
        RewardTraceWrapper(_RawRewardTraceEnv([1.0, 2.0, 3.0])),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=10,
        hold_policy="hold",
        ordering_policy="latest_ready",
        add_latency_info=True,
    )

    try:
        wrapped.reset()
        _, reward, _, _, info = wrapped.step(1)
    finally:
        wrapped.close()

    assert reward == pytest.approx(6.0)
    assert info["raw_reward"] == pytest.approx(6.0)


def test_action_latency_wrapper_auto_seeds_underlying_env_resets():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.sample_factory_common.latency_wrapper import QueueLatencyWrapper

    class SeedEchoEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def __init__(self):
            self.seeds: list[int | None] = []

        def reset(self, *, seed=None, options=None):
            self.seeds.append(seed)
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    env = SeedEchoEnv()
    wrapped = QueueLatencyWrapper(
        env,
        queue_latency_values=[0],
        latency_mode="episode_fixed",
        seed=11,
        env_reset_seed_base=500,
    )
    wrapped.reset()
    wrapped.reset()
    wrapped.reset(seed=999)
    assert env.seeds == [500, 501, 999]


def test_same_latency_sweep_generator_emits_sample_factory_queue_eval_configs():
    train_cfg = _build_train_config(
        env_name="demon_attack",
        seed=0,
        latency=4,
        max_steps=3600,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=1234,
    )
    eval_cfg = _build_eval_config(
        train_config=train_cfg,
        env_name="demon_attack",
        seed=0,
        latency=4,
        max_steps=3600,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )
    assert train_cfg["training"]["train_for_env_steps"] == 1234
    assert train_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["evaluation"]["eval_episodes"] == 10
    assert eval_cfg["evaluation"]["eval_latency_values"] == [4]
    assert eval_cfg["evaluation"]["eval_max_steps"] == 3600
