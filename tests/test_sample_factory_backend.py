from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import yaml

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
from scripts.generate_sample_factory_same_latency_sweep import (
    _build_eval_config,
    _build_full_run_config,
    _build_latency_bench_eval_config,
    _build_train_config,
    _latency_steps_to_ms,
    main as generate_same_latency_sweep,
)
from training.sample_factory_common.config_adapter import (
    apply_runtime_logging_config,
    apply_training_overrides,
    build_common_sf_args,
    load_training_config,
)


class _ScriptedAtariInfoEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0
        self.reset_count = 0

    def reset(self, **kwargs):
        self.reset_count += 1
        return np.full((2, 2), self.reset_count, dtype=np.uint8), {}

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        obs = np.full((2, 2), self.index + 10, dtype=np.uint8)
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)


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


def test_build_common_sf_args_does_not_emit_unsupported_wandb_unique_id_arg():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["logging"]["wandb_name"] = "flappy-bird-l0-s0"

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    assert "--wandb_unique_id" not in args


def test_apply_runtime_logging_config_maps_wandb_name_to_sample_factory_unique_id():
    cfg = SimpleNamespace()

    apply_runtime_logging_config(
        cfg,
        {
            "output_dir": "outputs/latency_sweeps/train/flappy/latency_0/seed_0",
            "wandb_name": "flappy-bird-l0-s0",
        },
    )

    assert cfg.output_dir == "outputs/latency_sweeps/train/flappy/latency_0/seed_0"
    assert cfg.wandb_unique_id == "flappy-bird-l0-s0"


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


def test_queue_latency_wrapper_auto_seeds_underlying_env_resets():
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


def test_episode_metrics_wrapper_suppresses_life_loss_episode_event(tmp_path):
    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        _ScriptedAtariInfoEnv([(1.0, True, False, {"lives": 2})]),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(0)

    assert terminated is True
    assert truncated is False
    assert info["is_true_episode_end"] is False
    assert "episodic_return" not in info
    assert "episodic_length" not in info
    assert "episode_extra_stats" not in info
    assert not metrics_path.exists()


def test_episode_metrics_wrapper_reports_game_over_full_game_return(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        _ScriptedAtariInfoEnv([(2.0, True, False, {"lives": 0})]),
        metrics_path=str(metrics_path),
        frames_per_decision=4,
        worker_index=1,
        vector_index=2,
        env_id=3,
    )

    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(0)

    assert terminated is True
    assert truncated is False
    assert info["is_true_episode_end"] is True
    assert info["episodic_return"] == 2.0
    assert info["episodic_length"] == 1
    assert info["episode_extra_stats"]["episode/return"] == 2.0
    assert info["episode_extra_stats"]["episode/raw_frames"] == 4
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["return"] == 2.0
    assert event["length"] == 1
    assert event["raw_frames"] == 4
    assert event["worker_index"] == 1
    assert event["vector_index"] == 2
    assert event["env_id"] == 3


def test_episode_metrics_wrapper_separates_raw_and_clipped_full_game_returns(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import (
        ClipRewardWithRawInfoWrapper,
        EpisodeMetricsWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        ClipRewardWithRawInfoWrapper(_ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})])),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)

    assert reward == 1.0
    assert info["episodic_raw_return"] == 10.0
    assert info["episodic_clipped_return"] == 1.0
    assert info["episodic_return"] == 10.0
    assert info["episode_extra_stats"]["episode/raw_return"] == 10.0
    assert info["episode_extra_stats"]["episode/clipped_return"] == 1.0
    assert info["episode_extra_stats"]["episode/return"] == 10.0
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 10.0
    assert event["clipped_return"] == 1.0
    assert event["return"] == 10.0


def test_episode_metrics_wrapper_accumulates_raw_and_clipped_across_steps(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import (
        ClipRewardWithRawInfoWrapper,
        EpisodeMetricsWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        ClipRewardWithRawInfoWrapper(
            _ScriptedAtariInfoEnv(
                [
                    (10.0, False, False, {"lives": 3}),
                    (-5.0, False, False, {"lives": 3}),
                    (0.0, True, False, {"lives": 0}),
                ]
            )
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    wrapped.step(0)
    wrapped.step(0)
    _, _, _, _, info = wrapped.step(0)

    assert info["episodic_raw_return"] == 5.0
    assert info["episodic_clipped_return"] == 0.0
    assert info["episodic_return"] == 5.0
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 5.0
    assert event["clipped_return"] == 0.0
    assert event["return"] == 5.0


def test_episode_metrics_wrapper_does_not_emit_raw_return_on_life_loss(tmp_path):
    from training.sample_factory_common.latency_wrapper import (
        ClipRewardWithRawInfoWrapper,
        EpisodeMetricsWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        ClipRewardWithRawInfoWrapper(_ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 2})])),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["is_true_episode_end"] is False
    assert "episodic_raw_return" not in info
    assert "episodic_clipped_return" not in info
    assert "episodic_return" not in info
    assert not metrics_path.exists()


def test_episode_metrics_wrapper_emits_raw_full_game_return_on_game_over(tmp_path):
    from training.sample_factory_common.latency_wrapper import (
        ClipRewardWithRawInfoWrapper,
        EpisodeMetricsWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        ClipRewardWithRawInfoWrapper(_ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})])),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["is_true_episode_end"] is True
    assert info["episodic_raw_return"] == 10.0
    assert info["episodic_clipped_return"] == 1.0
    assert info["episodic_return"] == 10.0


def test_episode_metrics_wrapper_accumulates_return_across_life_losses(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        _ScriptedAtariInfoEnv(
            [
                (1.0, True, False, {"lives": 2}),
                (2.0, True, False, {"lives": 1}),
                (3.0, True, False, {"lives": 0}),
            ]
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, _, _, first_info = wrapped.step(0)
    wrapped.reset()
    _, _, _, _, second_info = wrapped.step(0)
    wrapped.reset()
    _, _, _, _, final_info = wrapped.step(0)

    assert "episodic_return" not in first_info
    assert "episodic_return" not in second_info
    assert final_info["episodic_return"] == 6.0
    assert final_info["episodic_length"] == 3
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["return"] == 6.0
    assert events[0]["length"] == 3


def test_episode_metrics_wrapper_marks_truncation_as_true_episode_end(tmp_path):
    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        _ScriptedAtariInfoEnv([(1.0, False, True, {"lives": 2})]),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(0)

    assert terminated is False
    assert truncated is True
    assert info["is_true_episode_end"] is True
    assert info["episodic_return"] == 1.0
    assert info["episodic_length"] == 1
    assert metrics_path.exists()


def test_episode_metrics_aggregator_logs_raw_visible_metrics_with_clipped_debug_key(tmp_path):
    from training.sample_factory_common.episode_metrics import EpisodeMetricsAggregator

    aggregator = EpisodeMetricsAggregator(path=tmp_path / "episodes.jsonl", rolling_window=10)

    payload = aggregator.process_event(
        {
            "return": 30.0,
            "clipped_return": 2.0,
            "raw_return": 30.0,
            "length": 3,
            "raw_frames": 12,
        }
    )

    assert payload["charts/episodic_return"] == 30.0
    assert payload["charts/episodic_return_raw"] == 30.0
    assert payload["rollout/ep_rew_mean"] == 30.0
    assert payload["rollout/ep_raw_rew_mean"] == 30.0
    assert payload["episode_train/latest_return"] == 30.0
    assert payload["episode_train/latest_raw_return"] == 30.0
    assert payload["episode_train/rolling_mean_return"] == 30.0
    assert payload["episode_train/rolling_mean_raw_return"] == 30.0
    assert payload["episode_train/cumulative_mean_return"] == 30.0
    assert payload["episode_train/cumulative_mean_raw_return"] == 30.0
    assert payload["rollout/ep_clipped_rew_mean"] == 2.0
    assert [key for key in payload if "clipped" in key] == ["rollout/ep_clipped_rew_mean"]


def test_sample_factory_atari_episode_patch_suppresses_native_life_loss_report():
    pytest.importorskip("sample_factory")
    from sample_factory.algo.sampling.non_batched_sampling import ActorState

    from training.atari_episode_stats import patch_sample_factory_atari_episode_stats

    patch_sample_factory_atari_episode_stats()
    state = ActorState.__new__(ActorState)
    state.curr_traj_buffer = {
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=np.bool_),
        "time_outs": np.zeros(1, dtype=np.bool_),
        "policy_id": np.zeros(1, dtype=np.int32),
    }
    state.is_active = True
    state.curr_policy_id = 0
    state.cfg = SimpleNamespace(summaries_use_frameskip=False)
    state.env_info = SimpleNamespace(frameskip=1)
    state.last_episode_reward = 7.0
    state.last_episode_duration = 2.0
    state.training_info = [None]
    state.agent_idx = 0
    state.env_idx = 0
    state.global_env_idx = 0
    state.policy_mgr = SimpleNamespace(get_policy_for_agent=lambda *args: 0)

    report = state.record_env_step(
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"lives": 2},
        rollout_step=0,
    )

    assert report is None
    assert state.curr_traj_buffer["dones"][0]
    assert state.last_episode_reward == 7.0
    assert state.last_episode_duration == 3.0


def test_sample_factory_non_batched_episode_patch_reports_raw_full_game_return():
    pytest.importorskip("sample_factory")
    from sample_factory.algo.sampling.non_batched_sampling import ActorState
    from sample_factory.algo.utils.misc import EPISODIC

    from training.atari_episode_stats import patch_sample_factory_atari_episode_stats

    patch_sample_factory_atari_episode_stats()
    state = ActorState.__new__(ActorState)
    state.curr_traj_buffer = {
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=np.bool_),
        "time_outs": np.zeros(1, dtype=np.bool_),
        "policy_id": np.zeros(1, dtype=np.int32),
    }
    state.is_active = True
    state.curr_policy_id = 0
    state.cfg = SimpleNamespace(summaries_use_frameskip=False)
    state.env_info = SimpleNamespace(frameskip=1)
    state.last_episode_reward = 2.0
    state.last_episode_duration = 1.0
    state.training_info = [None]
    state.agent_idx = 0
    state.env_idx = 0
    state.global_env_idx = 0
    state.policy_mgr = SimpleNamespace(get_policy_for_agent=lambda *args: 0)

    report = state.record_env_step(
        reward=1.0,
        terminated=True,
        truncated=False,
        info={
            "lives": 0,
            "episodic_return": 30.0,
            "episode_extra_stats": {
                "episode/raw_return": 30.0,
                "episode/clipped_return": 2.0,
                "episode/return": 30.0,
            },
        },
        rollout_step=0,
    )

    stats = report[EPISODIC]
    assert stats["reward"] == 30.0
    assert stats["len"] == 2.0
    assert stats["episode_extra_stats"]["episode/raw_return"] == 30.0
    assert stats["episode_extra_stats"]["episode/return"] == 30.0
    assert "episode/clipped_return" not in stats["episode_extra_stats"]


def test_sample_factory_batched_episode_patch_accumulates_across_life_loss():
    pytest.importorskip("sample_factory")
    torch = pytest.importorskip("torch")
    from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
    from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY

    from training.atari_episode_stats import patch_sample_factory_atari_episode_stats

    patch_sample_factory_atari_episode_stats()
    runner = BatchedVectorEnvRunner.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(summaries_use_frameskip=False)
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=1)
    runner.policy_id = 4
    runner.curr_episode_reward = torch.zeros(1)
    runner.curr_episode_len = torch.zeros(1, dtype=torch.int32)
    runner.min_raw_rewards = torch.tensor([1.0])
    runner.max_raw_rewards = torch.tensor([1.0])

    life_loss_reports = runner._process_env_step(
        rewards=torch.tensor([1.0]),
        dones_orig=torch.tensor([True]),
        infos=[{"lives": 2, "is_true_episode_end": False}],
    )

    assert life_loss_reports == []
    assert runner.curr_episode_reward.tolist() == [1.0]
    assert runner.curr_episode_len.tolist() == [1]

    game_over_reports = runner._process_env_step(
        rewards=torch.tensor([3.0]),
        dones_orig=torch.tensor([True]),
        infos=[
            {
                "lives": 0,
                "is_true_episode_end": True,
                "episode_extra_stats": {
                    "episode/raw_return": 10.0,
                    "episode/clipped_return": 4.0,
                    "episode/return": 10.0,
                },
            }
        ],
    )

    assert len(game_over_reports) == 1
    assert game_over_reports[0][POLICY_ID_KEY] == 4
    assert game_over_reports[0][EPISODIC]["reward"].tolist() == [10.0]
    assert game_over_reports[0][EPISODIC]["len"].tolist() == [2]
    assert "episode/clipped_return" not in game_over_reports[0][EPISODIC]
    assert runner.curr_episode_reward.tolist() == [0.0]
    assert runner.curr_episode_len.tolist() == [0]


def test_sample_factory_batched_episode_patch_reports_truncation_with_lives_remaining():
    pytest.importorskip("sample_factory")
    torch = pytest.importorskip("torch")
    from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
    from sample_factory.algo.utils.misc import EPISODIC

    from training.atari_episode_stats import patch_sample_factory_atari_episode_stats

    patch_sample_factory_atari_episode_stats()
    runner = BatchedVectorEnvRunner.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(summaries_use_frameskip=False)
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=1)
    runner.policy_id = 0
    runner.curr_episode_reward = torch.zeros(1)
    runner.curr_episode_len = torch.zeros(1, dtype=torch.int32)
    runner.min_raw_rewards = torch.tensor([2.0])
    runner.max_raw_rewards = torch.tensor([2.0])

    reports = runner._process_env_step(
        rewards=torch.tensor([2.0]),
        dones_orig=torch.tensor([True]),
        infos=[{"lives": 2, "is_true_episode_end": True}],
    )

    assert len(reports) == 1
    assert reports[0][EPISODIC]["reward"].tolist() == [2.0]
    assert reports[0][EPISODIC]["len"].tolist() == [1]


def test_sample_factory_batched_patch_forwards_episode_extra_raw_stats():
    pytest.importorskip("sample_factory")
    torch = pytest.importorskip("torch")
    from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
    from sample_factory.algo.utils.misc import EPISODIC

    from training.atari_episode_stats import patch_sample_factory_atari_episode_stats

    patch_sample_factory_atari_episode_stats()
    runner = BatchedVectorEnvRunner.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(summaries_use_frameskip=False)
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=1)
    runner.policy_id = 0
    runner.curr_episode_reward = torch.zeros(1)
    runner.curr_episode_len = torch.zeros(1, dtype=torch.int32)
    runner.min_raw_rewards = torch.tensor([1.0])
    runner.max_raw_rewards = torch.tensor([1.0])

    reports = runner._process_env_step(
        rewards=torch.tensor([1.0]),
        dones_orig=torch.tensor([True]),
        infos=[
            {
                "lives": 0,
                "is_true_episode_end": True,
                "episode_extra_stats": {
                    "episode/raw_return": 30.0,
                    "episode/clipped_return": 2.0,
                    "episode/return": 30.0,
                    "episode/label": "ignored",
                },
            }
        ],
    )

    assert len(reports) == 1
    stats = reports[0][EPISODIC]
    assert stats["reward"].tolist() == [30.0]
    assert stats["episode/raw_return"].tolist() == [30.0]
    assert stats["episode/return"].tolist() == [30.0]
    assert "episode/clipped_return" not in stats
    assert "episode/label" not in stats


def test_demon_attack_env_factory_patches_sample_factory_stats_in_worker_process(monkeypatch):
    from training.demon_attack_sf import integration

    patch_calls = []
    monkeypatch.setattr(
        integration,
        "ensure_sample_factory_raw_reward_console_stats_installed",
        lambda *, env_name: patch_calls.append(env_name),
    )
    monkeypatch.setattr(
        integration,
        "make_demon_attack_benchmark_env",
        lambda **kwargs: _ScriptedAtariInfoEnv([]),
    )

    env = integration.make_demon_attack_sf_env(
        integration.SF_ENV_NAME,
        cfg=SimpleNamespace(),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )
    try:
        assert patch_calls == [integration.SF_ENV_NAME]
    finally:
        env.close()


def test_doom_frame_stack_wrapper_supports_box_observations():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.deadly_corridor_sf.integration import DoomFrameStackWrapper

    class ImageEnv(gym.Env):
        observation_space = Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        action_space = Discrete(2)

        def __init__(self):
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            self.step_count = 0
            return np.ones((3, 2, 2), dtype=np.uint8), {}

        def step(self, action):
            self.step_count += 1
            obs = np.full((3, 2, 2), self.step_count + 1, dtype=np.uint8)
            return obs, 0.0, False, False, {}

    wrapped = DoomFrameStackWrapper(ImageEnv(), num_frames=4)

    assert wrapped.observation_space.shape == (12, 2, 2)
    obs, _ = wrapped.reset()
    assert obs.shape == (12, 2, 2)
    assert np.all(obs == 1)
    obs, *_ = wrapped.step(0)
    assert np.all(obs[:9] == 1)
    assert np.all(obs[9:] == 2)


def test_deadly_corridor_prepare_eval_cfg_uses_synchronous_fps_when_base_is_async():
    from training.deadly_corridor_sf import integration

    base_cfg = SimpleNamespace(
        fps=0,
        env_frameskip=4,
        eval_deterministic=True,
        eval_raw_reward=False,
    )

    cfg = integration._prepare_eval_cfg(base_cfg, latency_value=3)

    assert cfg.fps == 35
    assert cfg.cli_args["fps"] == 35
    assert (cfg.fps == 0) is False


def test_deadly_corridor_prepare_eval_cfg_keeps_positive_base_fps_synchronous():
    from training.deadly_corridor_sf import integration

    base_cfg = SimpleNamespace(
        fps=50,
        env_frameskip=4,
        eval_deterministic=True,
        eval_raw_reward=False,
    )

    cfg = integration._prepare_eval_cfg(base_cfg, latency_value=0)

    assert cfg.fps == 50
    assert cfg.cli_args["fps"] == 50
    assert (cfg.fps == 0) is False


@pytest.mark.parametrize(
    ("env_name", "seed", "latency", "expected_wandb_name"),
    (
        ("flappy", 0, 0, "flappy-bird-l0-s0"),
        ("demon_attack", 0, 4, "demon-attack-l4-s0"),
        ("deadly_corridor", 1, 40, "deadly-corridor-l40-s1"),
    ),
)
def test_same_latency_sweep_generator_emits_train_wandb_names(
    env_name, seed, latency, expected_wandb_name
):
    train_cfg = _build_train_config(
        env_name=env_name,
        seed=seed,
        latency=latency,
        max_steps=3600,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
    )

    assert train_cfg["logging"]["wandb_name"] == expected_wandb_name
    assert train_cfg["experiment"]["name"] == f"{env_name}_sf_same_latency_l{latency}_seed{seed}"


def test_same_latency_sweep_generator_emits_sample_factory_queue_eval_configs():
    full_cfg = _build_full_run_config(
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
        train_config=full_cfg,
        env_name="demon_attack",
        seed=0,
        latency=4,
        max_steps=3600,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )
    expected_eval_latencies = [0, 1, 2, 3, 4, 5, 10, 15]
    assert train_cfg["training"]["train_for_env_steps"] == 1234
    assert train_cfg["latency"]["train_values"] == [4]
    assert train_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert train_cfg["logging"]["wandb_name"] == "demon-attack-l4-s0"
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["evaluation"]["eval_episodes"] == 10
    assert eval_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert eval_cfg["evaluation"]["eval_max_steps"] == 3600
    assert eval_cfg["logging"] == {
        "output_dir": "/workspace/latency_sweeps/eval/demon_attack/latency_4/seed_0"
    }


@pytest.mark.parametrize(
    ("env_name", "max_steps", "expected_policy", "expected_ms", "expected_max_env_steps"),
    (
        ("flappy", 1800, "flappy_sf", 100.0, 1800),
        ("demon_attack", 25000, "demon_attack_sf", 50.0, 3600),
        ("deadly_corridor", 2100, "deadly_corridor_sf", 85.7142857143, 2100),
    ),
)
def test_same_latency_sweep_generator_emits_latency_bench_eval_configs(
    tmp_path, env_name, max_steps, expected_policy, expected_ms, expected_max_env_steps
):
    full_cfg = _build_full_run_config(
        env_name=env_name,
        seed=0,
        latency=3,
        max_steps=max_steps,
        train_dir="results/checkpoints/sample_factory_runs",
        output_root=str(tmp_path / "outputs"),
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
        frame_stack_override=None,
    )
    bench_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
        env_name=env_name,
        seed=0,
        latency=3,
        max_steps=max_steps,
        output_root=str(tmp_path / "outputs"),
        eval_episodes=10,
    )
    cfg_path = tmp_path / f"{env_name}_latency_bench_eval.yaml"
    cfg_path.write_text(yaml.safe_dump(bench_cfg, sort_keys=False), encoding="utf-8")

    loaded = load_config(cfg_path)

    assert loaded["experiment"]["name"] == f"{env_name}_sf_same_latency_l3_seed0"
    assert loaded["experiment"]["num_episodes"] == 10
    assert loaded["experiment"]["max_env_steps"] == expected_max_env_steps
    assert loaded["backend"]["train_dir"] == "results/checkpoints/sample_factory_runs"
    assert loaded["backend"]["run_mode"] == "eval"
    assert loaded["policy"]["type"] == expected_policy
    assert loaded["policy"]["checkpoint_path"] == ""
    assert loaded["policy"]["device"] == "auto"
    assert loaded["policy"]["deterministic"] is True
    assert loaded["latency"]["type"] == "fixed"
    assert loaded["latency"]["train_values"] == [3]
    assert math.isclose(loaded["latency"]["fixed_latency_ms"], expected_ms, rel_tol=1e-9)
    assert loaded["logging"]["output_dir"] == str(
        tmp_path / "outputs" / "latency_bench" / env_name / "train_latency_3" / "seed_0"
    )
    expected_eval_latencies = {
        "flappy": [0, 1, 2, 3, 4, 5],
        "demon_attack": [0, 1, 2, 3, 4, 5, 10, 15],
        "deadly_corridor": [0, 1, 2, 3, 4, 5, 10, 15, 40],
    }
    assert loaded["evaluation"]["eval_latency_values"] == expected_eval_latencies[env_name]
    if env_name == "deadly_corridor":
        assert loaded["env"]["fps_env"] == 35
        assert loaded["env"]["noop_action"] == [0, 0, 0, 0]


def test_same_latency_sweep_generator_maps_zero_latency_to_zero_source(tmp_path):
    full_cfg = _build_full_run_config(
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        train_dir="results/checkpoints/sample_factory_runs",
        output_root=str(tmp_path / "outputs"),
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
        frame_stack_override=None,
    )
    bench_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        output_root=str(tmp_path / "outputs"),
        eval_episodes=10,
    )

    assert bench_cfg["latency"]["type"] == "zero"
    assert "fixed_latency_ms" not in bench_cfg["latency"]
    assert bench_cfg["evaluation"]["eval_latency_values"] == [0, 1, 2, 3, 4, 5]
    assert bench_cfg["logging"]["output_dir"] == str(
        tmp_path / "outputs" / "latency_bench" / "flappy" / "train_latency_0" / "seed_0"
    )
    assert _latency_steps_to_ms(full_cfg, 3) == 100.0


def test_same_latency_sweep_generator_preserves_deadly_corridor_eval_latencies_for_latency_zero(tmp_path):
    full_cfg = _build_full_run_config(
        env_name="deadly_corridor",
        seed=0,
        latency=0,
        max_steps=2100,
        train_dir="results/checkpoints/sample_factory_runs",
        output_root=str(tmp_path / "outputs"),
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
        frame_stack_override=None,
    )
    eval_cfg = _build_eval_config(
        train_config=full_cfg,
        env_name="deadly_corridor",
        seed=0,
        latency=0,
        max_steps=2100,
        output_root=str(tmp_path / "outputs"),
        eval_episodes=100,
    )
    bench_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
        env_name="deadly_corridor",
        seed=0,
        latency=0,
        max_steps=2100,
        output_root=str(tmp_path / "outputs"),
        eval_episodes=100,
    )
    expected_eval_latencies = [0, 1, 2, 3, 4, 5, 10, 15, 40]

    assert full_cfg["latency"]["train_values"] == [0]
    assert eval_cfg["latency"]["train_values"] == [0]
    assert bench_cfg["latency"]["train_values"] == [0]
    assert eval_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert bench_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert eval_cfg["env"]["fps_env"] == 35
    assert bench_cfg["env"]["fps_env"] == 35
    assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]


def test_same_latency_sweep_generator_writes_three_configs_and_commands(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"

    generate_same_latency_sweep(
        [
            "--envs",
            "flappy,demon_attack,deadly_corridor",
            "--seeds",
            "0",
            "--flappy-latencies",
            "0,3",
            "--demon-attack-latencies",
            "0,3",
            "--deadly-corridor-latencies",
            "0,3",
            "--generated-root",
            str(generated_root),
            "--train-dir",
            "results/checkpoints/sample_factory_runs",
            "--output-root",
            str(output_root),
            "--device",
            "gpu",
            "--restart-behavior",
            "resume",
            "--eval-episodes",
            "10",
        ]
    )

    expected_policy_types = {
        "flappy": "flappy_sf",
        "demon_attack": "demon_attack_sf",
        "deadly_corridor": "deadly_corridor_sf",
    }
    expected_fixed_ms = {
        "flappy": 100.0,
        "demon_attack": 50.0,
        "deadly_corridor": 85.7142857143,
    }
    expected_eval_latencies = {
        "flappy": [0, 1, 2, 3, 4, 5],
        "demon_attack": [0, 1, 2, 3, 4, 5, 10, 15],
        "deadly_corridor": [0, 1, 2, 3, 4, 5, 10, 15, 40],
    }
    for env_name, policy_type in expected_policy_types.items():
        for latency in (0, 3):
            run_dir = generated_root / env_name / f"latency_{latency}" / "seed_0"
            train_path = run_dir / "train.yaml"
            eval_path = run_dir / "eval.yaml"
            assert train_path.is_file()
            assert eval_path.is_file()
            bench_path = run_dir / "latency_bench_eval.yaml"
            assert bench_path.is_file()

            train_loaded = yaml.safe_load(train_path.read_text(encoding="utf-8"))
            eval_loaded = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
            loaded = load_config(bench_path)
            assert train_loaded["latency"]["train_values"] == [latency]
            assert train_loaded["evaluation"]["eval_latency_values"] == expected_eval_latencies[env_name]
            assert eval_loaded["latency"]["train_values"] == [latency]
            assert eval_loaded["evaluation"]["eval_latency_values"] == expected_eval_latencies[env_name]
            assert loaded["experiment"]["name"] == f"{env_name}_sf_same_latency_l{latency}_seed0"
            assert loaded["backend"]["train_dir"] == "results/checkpoints/sample_factory_runs"
            assert loaded["policy"]["type"] == policy_type
            assert loaded["policy"]["checkpoint_path"] == ""
            assert loaded["evaluation"]["eval_latency_values"] == expected_eval_latencies[env_name]
            assert loaded["logging"]["output_dir"] == str(
                output_root / "latency_bench" / env_name / f"train_latency_{latency}" / "seed_0"
            )
            if env_name == "deadly_corridor":
                assert train_loaded["env"]["fps_env"] == 35
                assert eval_loaded["env"]["fps_env"] == 35
                assert loaded["env"]["fps_env"] == 35
                assert loaded["env"]["noop_action"] == [0, 0, 0, 0]
            if latency == 0:
                assert loaded["latency"]["type"] == "zero"
                assert "fixed_latency_ms" not in loaded["latency"]
            else:
                assert loaded["latency"]["type"] == "fixed"
                assert math.isclose(
                    loaded["latency"]["fixed_latency_ms"],
                    expected_fixed_ms[env_name],
                    rel_tol=1e-9,
                )

    manifest = (generated_root / "manifest.tsv").read_text(encoding="utf-8")
    assert "latency_bench_eval_config" in manifest.splitlines()[0]
    assert "latency_bench_eval_command" in manifest.splitlines()[0]
    latency_bench_commands = (generated_root / "latency_bench_eval_commands.txt").read_text(
        encoding="utf-8"
    )
    assert "python -m latency_bench.run --config" in latency_bench_commands
