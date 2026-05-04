from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import yaml

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
import scripts.generate_sample_factory_same_latency_sweep as same_latency_sweep
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
    validate_training_config,
)


def _patch_flappy_base_config(tmp_path, monkeypatch, *, worker_num_splits: int) -> None:
    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["training"]["worker_num_splits"] = int(worker_num_splits)
    path = tmp_path / f"flappy_worker_splits_{worker_num_splits}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setitem(same_latency_sweep.BASE_CONFIGS, "flappy", path)


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


class _RenderedRgbEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)

    def __init__(self, shape=(512, 288, 3)):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.screen_size = (288, 512)
        self.shape = tuple(shape)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros((2,), dtype=np.float32), 0.0, False, False, {}

    def render(self):
        frame = np.zeros(self.shape, dtype=np.uint8)
        frame[..., 0] = 17
        frame[..., 1] = 91
        frame[..., 2] = 203
        return frame


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


def test_flappy_sf_cli_maps_explicit_env_max_episode_steps():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"]["max_episode_steps"] = 1800

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--max_episode_steps") + 1] == "1800"


def test_flappy_sf_cli_omits_max_episode_steps_when_env_field_is_absent():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"].pop("max_episode_steps", None)

    args = build_cli_args_from_config(cfg)

    assert "--max_episode_steps" not in args


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
    cfg.setdefault("backend", {})["extra_args"] = ["--some-existing-flag", "1"]
    updated = apply_training_overrides(
        cfg,
        run_name="override_run",
        train_dir="tmp/sample_factory",
        mode="train",
        checkpoint_kind="best",
        output_dir="outputs/eval/best/flappy/latency_0/seed_0",
    )
    assert updated["experiment"]["name"] == "override_run"
    assert updated["backend"]["train_dir"] == "tmp/sample_factory"
    assert updated["backend"]["run_mode"] == "train"
    assert updated["backend"]["extra_args"] == [
        "--some-existing-flag",
        "1",
        "--load_checkpoint_kind",
        "best",
    ]
    assert updated["logging"]["output_dir"] == "outputs/eval/best/flappy/latency_0/seed_0"


def test_flappy_sf_obs_resize_single_frame_shape():
    from training.flappy_sf.integration import FlappyImageObsWrapper

    wrapped = FlappyImageObsWrapper(_RenderedRgbEnv(), obs_resize=(84, 84))

    assert wrapped.observation_space.shape == (3, 84, 84)
    obs, _ = wrapped.reset()
    assert obs.shape == (3, 84, 84)


def test_flappy_sf_frame_stack_after_resize_shape():
    from training.flappy_sf.integration import FlappyFrameStackWrapper, FlappyImageObsWrapper

    image_env = FlappyImageObsWrapper(_RenderedRgbEnv(), obs_resize=(84, 84))
    wrapped = FlappyFrameStackWrapper(image_env, num_frames=4)

    assert wrapped.observation_space.shape == (12, 84, 84)
    obs, _ = wrapped.reset()
    assert obs.shape == (12, 84, 84)


def test_flappy_sf_resize_preserves_uint8_rgb():
    from training.flappy_sf.integration import FlappyImageObsWrapper

    wrapped = FlappyImageObsWrapper(_RenderedRgbEnv(), obs_resize=(84, 84))
    obs, _ = wrapped.reset()

    assert obs.dtype == np.uint8
    assert obs.shape[0] == 3


def test_flappy_sf_resize_does_not_change_env_screen_size():
    from training.flappy_sf.integration import FlappyImageObsWrapper

    base_env = _RenderedRgbEnv()
    wrapped = FlappyImageObsWrapper(base_env, obs_resize=(84, 84))
    wrapped.reset()

    assert base_env.screen_size == (288, 512)


def test_flappy_sf_rejects_screen_size_config():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["screen_size"] = [84, 84]

    with pytest.raises(ValueError, match="env\\.obs_resize"):
        validate_training_config(cfg)


def test_flappy_sf_native_eval_uses_same_obs_resize(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

    full_cfg = _build_full_run_config(
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
    )
    train_cfg = _build_train_config(
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
    )
    eval_cfg = _build_eval_config(
        train_config=full_cfg,
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )

    assert train_cfg["env"]["obs_resize"] == [84, 84]
    assert eval_cfg["env"]["obs_resize"] == [84, 84]
    assert train_cfg["env"]["obs_mode"] == eval_cfg["env"]["obs_mode"] == "image_only"


def test_flappy_sf_sweep_config_preserves_gpu_render_fields(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

    full_cfg = _build_full_run_config(
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
    )
    train_cfg = _build_train_config(
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=None,
    )
    eval_cfg = _build_eval_config(
        train_config=full_cfg,
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )

    assert train_cfg["env"]["use_gpu_render"] is True
    assert train_cfg["env"]["gpu_render_device"] == "auto"
    assert train_cfg["env"]["gpu_render_profile"] == full_cfg["env"]["gpu_render_profile"]
    assert train_cfg["env"]["gpu_render_profile_interval"] == full_cfg["env"]["gpu_render_profile_interval"]
    assert eval_cfg["env"]["use_gpu_render"] is True
    assert eval_cfg["env"]["gpu_render_device"] == "auto"
    assert eval_cfg["env"]["gpu_render_profile"] == full_cfg["env"]["gpu_render_profile"]
    assert eval_cfg["env"]["gpu_render_profile_interval"] == full_cfg["env"]["gpu_render_profile_interval"]
    assert train_cfg["training"]["worker_num_splits"] == full_cfg["training"]["worker_num_splits"] == 1
    assert eval_cfg["training"]["worker_num_splits"] == full_cfg["training"]["worker_num_splits"] == 1


def test_flappy_sf_sweep_config_rejects_gpu_worker_splits_without_rewrite(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=2)

    with pytest.raises(ValueError, match="worker_num_splits"):
        _build_full_run_config(
            env_name="flappy",
            seed=0,
            latency=0,
            max_steps=1800,
            train_dir="/workspace/sample_factory_runs",
            output_root="/workspace/latency_sweeps",
            device="gpu",
            restart_behavior="resume",
            train_for_env_steps=None,
        )


def test_flappy_sf_latency_channel_shape():
    from modeling.flappy_ppo.preprocessing import latency_channel_value
    from training.flappy_sf.integration import (
        FlappyFrameStackWrapper,
        FlappyImageObsWrapper,
        FlappyLatencyChannelWrapper,
    )

    image_env = FlappyImageObsWrapper(_RenderedRgbEnv(), obs_resize=(84, 84))
    stacked = FlappyFrameStackWrapper(image_env, num_frames=4)
    wrapped = FlappyLatencyChannelWrapper(stacked, max_frame_skip=5)

    assert wrapped.observation_space.shape == (13, 84, 84)
    obs, info = wrapped.reset()
    assert obs.shape == (13, 84, 84)
    assert np.all(obs[-1] == latency_channel_value(0, 5))
    latency_obs = wrapped.observation(obs[:12], {"frame_skip": 3})
    assert np.all(latency_obs[-1] == latency_channel_value(3, 5))


def test_flappy_sf_shape_mismatch_fails_fast():
    from latency_bench.policy.sample_factory_common import validate_sample_factory_obs_shape

    with pytest.raises(ValueError, match="observation shape mismatch"):
        validate_sample_factory_obs_shape(
            {"obs": np.zeros((1, 12, 512, 288), dtype=np.uint8)},
            expected_shape=(12, 84, 84),
            policy_type="flappy_sf",
        )


def test_flappy_sf_update_ratio_metadata():
    from training.sample_factory_common.config_adapter import estimate_optimizer_steps_per_1m_env_steps

    assert estimate_optimizer_steps_per_1m_env_steps(
        {"batch_size": 1024, "num_epochs": 4, "num_batches_per_epoch": 2}
    ) == pytest.approx(3906.25)


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


def test_episode_metrics_wrapper_emits_flappy_timeout_and_issued_action_debug_metrics(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        _ScriptedAtariInfoEnv(
            [
                (0.5, False, False, {"score": 1}),
                (0.5, False, False, {"score": 1, "applied_action": 0}),
                (1.0, False, True, {"score": 2, "pipes_passed": 2}),
            ]
        ),
        metrics_path=str(metrics_path),
        issued_action_labels={0: "noop", 1: "flap"},
    )

    wrapped.reset()
    wrapped.step(0)
    wrapped.step(0)
    _, _, terminated, truncated, info = wrapped.step(1)

    assert terminated is False
    assert truncated is True
    extra = info["episode_extra_stats"]
    assert extra["episode/length"] == 3
    assert extra["episode/terminated"] == 0.0
    assert extra["episode/truncated"] == 1.0
    assert extra["episode/timeout"] == 1.0
    assert extra["episode/return_per_step"] == pytest.approx(2.0 / 3.0)
    assert extra["episode/score"] == 2.0
    assert extra["episode/pipes_passed"] == 2.0
    assert extra["action/issued_noop_ratio"] == pytest.approx(2.0 / 3.0)
    assert extra["action/issued_flap_ratio"] == pytest.approx(1.0 / 3.0)
    assert extra["action/issued_entropy"] > 0.0
    assert extra["action/issued_consecutive_noop_mean"] == 2.0
    assert extra["action/issued_consecutive_noop_max"] == 2.0
    assert extra["action/issued_consecutive_flap_mean"] == 1.0
    assert extra["action/issued_consecutive_flap_max"] == 1.0
    assert "done_reason" not in extra

    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["terminated"] is False
    assert event["truncated"] is True
    assert event["time_limit_truncated"] is True
    assert event["done_reason"] == "truncated"
    assert event["action/issued_flap_ratio"] == pytest.approx(1.0 / 3.0)


def test_flappy_reward_trace_keeps_raw_and_transformed_returns_equal(tmp_path):
    import json

    from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper, RewardTraceWrapper

    metrics_path = tmp_path / "flappy_rewards.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv(
                [
                    (0.5, False, False, {}),
                    (1.5, True, False, {}),
                ]
            )
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    wrapped.step(0)
    _, _, _, _, info = wrapped.step(0)

    assert info["episode_extra_stats"]["episode/raw_return"] == 2.0
    assert info["episode_extra_stats"]["episode/clipped_return"] == 2.0
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 2.0
    assert event["clipped_return"] == 2.0


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


def test_episode_metrics_aggregator_logs_numeric_debug_metrics_and_omits_done_reason(tmp_path):
    from training.sample_factory_common.episode_metrics import EpisodeMetricsAggregator

    aggregator = EpisodeMetricsAggregator(path=tmp_path / "episodes.jsonl", rolling_window=10)

    payload = aggregator.process_event(
        {
            "return": 12.0,
            "raw_return": 12.0,
            "clipped_return": 12.0,
            "length": 4,
            "raw_frames": 4,
            "terminated": False,
            "truncated": True,
            "timeout": True,
            "done_reason": "truncated",
            "score": 3,
            "action/issued_flap_ratio": 0.25,
            "action/issued_noop_ratio": 0.75,
            "action/issued_entropy": 0.562,
            "action/issued_consecutive_noop_mean": 3.0,
            "action/issued_consecutive_noop_max": 3.0,
            "action/issued_consecutive_flap_mean": 1.0,
            "action/issued_consecutive_flap_max": 1.0,
        }
    )

    assert payload["episode/length"] == 4
    assert payload["episode/terminated_rate"] == 0.0
    assert payload["episode/truncated_rate"] == 1.0
    assert payload["episode/timeout_rate"] == 1.0
    assert payload["episode/return_per_step"] == 3.0
    assert payload["episode/score"] == 3.0
    assert payload["action/issued_flap_ratio"] == 0.25
    assert payload["action/issued_noop_ratio"] == 0.75
    assert payload["action/issued_entropy"] == 0.562
    assert payload["action/issued_consecutive_noop_mean"] == 3.0
    assert payload["action/issued_consecutive_noop_max"] == 3.0
    assert payload["action/issued_consecutive_flap_mean"] == 1.0
    assert payload["action/issued_consecutive_flap_max"] == 1.0
    assert "done_reason" not in payload


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


def test_demon_attack_reference_env_is_sample_factory_compatible(monkeypatch):
    from sample_factory.algo.utils.env_info import extract_env_info

    from training.demon_attack_sf import integration

    monkeypatch.setattr(integration, "_ENVS_REGISTERED", False)
    monkeypatch.setattr(
        integration,
        "make_demon_attack_benchmark_env",
        lambda **kwargs: _ScriptedAtariInfoEnv([]),
    )

    cfg = SimpleNamespace(
        env=integration.SF_ENV_NAME,
        env_gpu_actions=False,
        env_gpu_observations=False,
        env_frameskip=1,
        seed=0,
        action_repeat=4,
        frame_stack=4,
        terminal_on_life_loss=True,
        clip_reward=True,
        queue_latency_values="0",
        latency_mode="episode_fixed",
        latency_sampling_mode="uniform_all",
        episodes_per_stage=10,
        randomize_within_stage=False,
        within_episode_change_interval=1,
        within_episode_change_prob=1.0,
        add_latency_info=False,
        max_episode_steps=0,
    )

    env = integration.make_reference_env(cfg)
    try:
        assert env.num_agents == 1
        assert env.is_multiagent is True
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert env.observation_space["obs"].shape == (4, 2, 2)
        assert env.action_space == _ScriptedAtariInfoEnv.action_space

        env_info = extract_env_info(env, cfg)

        assert env_info.num_agents == 1
        assert env_info.obs_space == env.observation_space
        assert env_info.action_space == env.action_space
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
    tmp_path, monkeypatch, env_name, seed, latency, expected_wandb_name
):
    if env_name == "flappy":
        _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

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
    if env_name == "flappy":
        assert "max_episode_steps" not in train_cfg["env"]


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
    expected_eval_latencies = list(range(16))
    assert train_cfg["training"]["train_for_env_steps"] == 1234
    assert train_cfg["latency"]["train_values"] == [4]
    assert train_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert train_cfg["logging"]["wandb_name"] == "demon-attack-l4-s0"
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["evaluation"]["eval_episodes"] == 10
    assert eval_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert eval_cfg["evaluation"]["eval_max_steps"] == 3600
    assert "max_episode_steps" not in eval_cfg["env"]
    assert eval_cfg["env"]["terminal_on_life_loss"] is True
    assert eval_cfg["env"]["clip_reward"] is True
    assert eval_cfg["logging"] == {
        "output_dir": "/workspace/latency_sweeps/eval/demon_attack/latency_4/seed_0"
    }


@pytest.mark.parametrize(
    ("env_name", "max_steps", "expected_policy", "expected_ms", "expected_eval_max_steps"),
    (
        ("flappy", 1800, "flappy_sf", 100.0, 3600),
        ("demon_attack", 25000, "demon_attack_sf", 50.0, 3600),
        ("deadly_corridor", 2100, "deadly_corridor_sf", 85.7142857143, 30000),
    ),
)
def test_same_latency_sweep_generator_emits_latency_bench_eval_configs(
    tmp_path, monkeypatch, env_name, max_steps, expected_policy, expected_ms, expected_eval_max_steps
):
    if env_name == "flappy":
        _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

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
    assert set(loaded["experiment"]) == {"name", "seed", "num_episodes"}
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
        "flappy": list(range(16)),
        "demon_attack": list(range(16)),
        "deadly_corridor": [*range(16), 40],
    }
    assert loaded["evaluation"]["eval_latency_values"] == expected_eval_latencies[env_name]
    assert loaded["evaluation"]["eval_max_steps"] == expected_eval_max_steps
    assert "max_episode_steps" not in loaded["env"]
    if env_name == "demon_attack":
        assert loaded["env"]["terminal_on_life_loss"] is True
        assert loaded["env"]["clip_reward"] is True
    if env_name == "deadly_corridor":
        assert loaded["env"]["fps_env"] == 35
        assert loaded["env"]["noop_action"] == [0, 0, 0, 0]


def test_same_latency_sweep_generator_maps_zero_latency_to_zero_source(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

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
    assert bench_cfg["evaluation"]["eval_latency_values"] == list(range(16))
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
    expected_eval_latencies = [*range(16), 40]

    assert full_cfg["latency"]["train_values"] == [0]
    assert eval_cfg["latency"]["train_values"] == [0]
    assert bench_cfg["latency"]["train_values"] == [0]
    assert eval_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert bench_cfg["evaluation"]["eval_latency_values"] == expected_eval_latencies
    assert "max_episode_steps" not in eval_cfg["env"]
    assert "max_episode_steps" not in bench_cfg["env"]
    assert eval_cfg["env"]["fps_env"] == 35
    assert bench_cfg["env"]["fps_env"] == 35
    assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]


def test_same_latency_sweep_generator_writes_three_configs_and_commands(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=1)

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
        "flappy": list(range(16)),
        "demon_attack": list(range(16)),
        "deadly_corridor": [*range(16), 40],
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
            assert set(loaded["experiment"]) == {"name", "seed", "num_episodes"}
            assert "max_episode_steps" not in loaded["env"]
            assert "max_episode_steps" not in eval_loaded["env"]
            if env_name == "flappy":
                assert "max_episode_steps" not in train_loaded["env"]
            if env_name == "demon_attack":
                assert train_loaded["env"]["max_episode_steps"] == 25000
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
