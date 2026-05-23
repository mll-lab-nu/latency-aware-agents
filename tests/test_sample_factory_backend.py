from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import yaml

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.core.latency_distribution import parse_latency_distribution
import scripts.generate_sample_factory_same_latency_sweep as same_latency_sweep
from scripts.generate_sample_factory_same_latency_sweep import (
    _build_full_run_config,
    _build_latency_bench_eval_config,
    _build_train_config,
    _latency_steps_to_ms,
    main as generate_same_latency_sweep,
)
from training.common.config_adapter import (
    apply_runtime_logging_config,
    apply_training_overrides,
    build_common_sf_args,
    load_training_config,
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


class _ActionTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        super().__init__()
        self.actions: list[int] = []
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.actions.clear()
        self.step_count = 0
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(int(action))
        self.step_count += 1
        obs = np.array([self.step_count], dtype=np.uint8)
        return obs, 1.0, False, False, {"raw_step": self.step_count}


class _RawRewardTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, rewards):
        super().__init__()
        self.rewards = list(rewards)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        reward = float(self.rewards[self.step_count])
        self.step_count += 1
        return np.array([self.step_count], dtype=np.uint8), reward, False, False, {}


def test_sample_factory_training_configs_load():
    cfg = load_training_config("configs/experiments/demon_attack/sf_fixed_latency_4.yaml")
    assert cfg["backend"]["type"] == "sample_factory"
    assert cfg["env"]["name"] == "demon_attack"
    assert cfg["latency"]["type"] == "fixed"
    assert cfg["latency"]["fixed_latency_ms"] == pytest.approx(4 * 1000.0 / 60.0)


def test_build_common_sf_args_emits_non_zero_latency_and_eval_settings():
    cfg = load_config("configs/experiments/deadly_corridor/sf_fixed_latency_4.yaml")
    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")
    assert "--latency-type" in args
    assert args[args.index("--latency-type") + 1] == "fixed"
    assert "--fixed-latency-ms" in args
    assert "--eval-latency-raw-frame-values" not in args
    assert "--eval-max-steps" in args
    assert "--experiment" in args
    assert "deadly_corridor_sf_fixed_latency_4" in args


def test_build_common_sf_args_maps_cuda_to_sample_factory_gpu():
    cfg = load_config("configs/experiments/deadly_corridor/sf_fixed_latency_4.yaml")
    cfg["backend"]["device"] = "cuda"

    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")

    assert args[args.index("--device") + 1] == "gpu"


def test_flappy_sf_cli_maps_explicit_env_max_episode_steps():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"]["max_episode_steps"] = 1800

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--max_episode_steps") + 1] == "1800"


def test_flappy_sf_cli_maps_env_rates_and_scheduler_policies():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"]["env_fps"] = 60
    cfg["env"]["obs_fps"] = 10
    cfg["scheduler"]["hold_policy"] = "hold"
    cfg["scheduler"]["ordering_policy"] = "issue_order_fifo"

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--env-fps") + 1] == "60"
    assert args[args.index("--obs-fps") + 1] == "10"
    assert args[args.index("--hold-policy") + 1] == "hold"
    assert args[args.index("--ordering-policy") + 1] == "issue_order_fifo"


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


def test_build_common_sf_args_emits_wandb_tags_as_multiple_values():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["logging"]["wandb_tags"] = ["factor_sweep", "flappy", "frame_stack"]

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    idx = args.index("--wandb_tags")
    assert args[idx : idx + 4] == ["--wandb_tags", "factor_sweep", "flappy", "frame_stack"]


def test_build_common_sf_args_splits_legacy_comma_wandb_tags():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["logging"]["wandb_tags"] = "factor_sweep,flappy,frame_stack"

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    idx = args.index("--wandb_tags")
    assert args[idx : idx + 4] == ["--wandb_tags", "factor_sweep", "flappy", "frame_stack"]


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


def test_flappy_image_obs_wrapper_reports_raw_rgb_before_resize():
    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.flappy_sf.integration import FlappyImageObsWrapper

    wrapped = FlappyImageObsWrapper(
        _RenderedRgbEnv(shape=(6, 5, 3)),
        obs_resize=(2, 2),
        export_raw_rgb_frames=True,
    )

    obs, info = wrapped.reset()

    assert obs.shape == (3, 2, 2)
    assert info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].shape == (1, 6, 5, 3)
    assert info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][0, 0, 0].tolist() == [17, 91, 203]


def test_flappy_image_obs_wrapper_keeps_training_info_clean_by_default():
    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.flappy_sf.integration import FlappyImageObsWrapper

    wrapped = FlappyImageObsWrapper(_RenderedRgbEnv(shape=(6, 5, 3)), obs_resize=(2, 2))

    _, info = wrapped.reset()

    assert ENV_RAW_RGB_FRAME_STACK_INFO_KEY not in info


def test_flappy_frame_stack_export_advances_raw_rgb_stack_oldest_to_newest():
    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.flappy_sf.integration import FlappyFrameStackWrapper, FlappyImageObsWrapper

    class StepRenderedEnv(_RenderedRgbEnv):
        def render(self):
            frame = super().render()
            frame[..., 0] = 17 + self.step_count
            return frame

    image_env = FlappyImageObsWrapper(
        StepRenderedEnv(shape=(6, 5, 3)),
        obs_resize=(2, 2),
        export_raw_rgb_frames=True,
    )
    wrapped = FlappyFrameStackWrapper(image_env, num_frames=3, export_raw_rgb_frames=True)

    _, reset_info = wrapped.reset()
    _, _, _, _, step_info = wrapped.step(0)

    assert reset_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0].tolist() == [
        [17, 91, 203],
        [17, 91, 203],
        [17, 91, 203],
    ]
    assert step_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0].tolist() == [
        [17, 91, 203],
        [17, 91, 203],
        [18, 91, 203],
    ]


def test_flappy_sf_resize_does_not_change_env_screen_size():
    from training.flappy_sf.integration import FlappyImageObsWrapper

    base_env = _RenderedRgbEnv()
    wrapped = FlappyImageObsWrapper(base_env, obs_resize=(84, 84))
    wrapped.reset()

    assert base_env.screen_size == (288, 512)


def test_flappy_sf_latency_bench_eval_uses_same_obs_resize(tmp_path, monkeypatch):
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
    eval_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
        env_name="flappy",
        seed=0,
        latency=0,
        max_steps=1800,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )

    assert train_cfg["env"]["obs_resize"] == [84, 84]
    assert eval_cfg["env"]["obs_resize"] == [84, 84]
    assert train_cfg["env"]["env_fps"] == 30
    assert train_cfg["env"]["obs_fps"] == 30
    assert eval_cfg["env"]["env_fps"] == 30
    assert eval_cfg["env"]["obs_fps"] == 30
    assert train_cfg["scheduler"] == full_cfg["scheduler"]
    assert eval_cfg["scheduler"] == full_cfg["scheduler"]
    assert "obs_mode" not in train_cfg["env"]
    assert "obs_mode" not in eval_cfg["env"]


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
    eval_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
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


def test_flappy_sf_sweep_config_preserves_gpu_worker_splits_without_validation(tmp_path, monkeypatch):
    _patch_flappy_base_config(tmp_path, monkeypatch, worker_num_splits=2)

    cfg = _build_full_run_config(
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

    assert cfg["training"]["worker_num_splits"] == 2


def test_flappy_sf_shape_mismatch_fails_fast():
    from latency_bench.policy.sample_factory_common import validate_sample_factory_obs_shape

    with pytest.raises(ValueError, match="observation shape mismatch"):
        validate_sample_factory_obs_shape(
            {"obs": np.zeros((1, 12, 512, 288), dtype=np.uint8)},
            expected_shape=(12, 84, 84),
            policy_type="flappy_sf",
        )


def test_flappy_sf_update_ratio_metadata():
    from training.common.config_adapter import estimate_optimizer_steps_per_1m_env_steps

    assert estimate_optimizer_steps_per_1m_env_steps(
        {"batch_size": 1024, "num_epochs": 4, "num_batches_per_epoch": 2}
    ) == pytest.approx(3906.25)


def test_common_sf_args_passes_latency_distribution_without_legacy_train_values():
    config = {
        "experiment": {"name": "dist_run", "seed": 3},
        "backend": {"type": "sample_factory", "train_dir": "runs", "run_mode": "train"},
        "env": {"name": "demon_attack", "env_fps": 60, "obs_fps": 15},
        "latency": {
            "type": "uniform_distribution",
            "min_latency_ms": 0.0,
            "max_latency_ms": 250.0,
            "add_latency_info": False,
        },
        "training": {"train_for_env_steps": 100, "num_workers": 1, "num_envs_per_worker": 1, "batch_size": 1},
        "evaluation": {},
        "logging": {},
    }

    args = build_common_sf_args(config, sf_env_name="latency_demon_attack")

    assert "--latency-raw-frame-values" not in args
    assert "--latency-mode" not in args
    assert args[args.index("--latency-type") + 1] == "uniform_distribution"
    assert args[args.index("--max-latency-ms") + 1] == "250.0"


def test_sample_factory_latency_params_use_bench_ceil_for_fixed_ms():
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="fixed",
        fixed_latency_ms=(1000.0 / 30.0) + 0.001,
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=30)

    assert params.fixed_latency_raw_frames == 2
    assert params.latency_distribution_spec is None


def test_sample_factory_latency_distribution_preserves_fractional_bounds():
    from latency_bench.core.latency_distribution import sample_latency_frame
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="uniform_distribution",
        min_latency_ms=50.0,
        max_latency_ms=(3.5 * 1000.0 / 30.0),
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=30)
    spec = params.latency_distribution_spec

    assert params.fixed_latency_raw_frames == 0
    assert spec is not None
    assert spec.min == pytest.approx(1.5)
    assert spec.max == pytest.approx(3.5)
    assert sample_latency_frame(spec, 0.0) == 2
    assert sample_latency_frame(spec, 99.0) == 4


def test_action_latency_wrapper_exposes_fixed_non_zero_latency():
    import gymnasium as gym
    from training.common import latency_wrapper
    from training.common.latency_wrapper import ActionLatencyWrapper

    assert not hasattr(latency_wrapper, "QueueLatencyWrapper")
    assert not hasattr(latency_wrapper, "RawFrameActionLatencyWrapper")
    env = gym.make("CartPole-v1")
    wrapped = ActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=4,
        env_fps=60,
        obs_fps=60,
    )
    _, info = wrapped.reset(seed=123)
    assert info["latency_raw_frames"] == 4
    _, _, _, _, step_info = wrapped.step(0)
    assert step_info["latency_raw_frames"] == 4
    wrapped.close()


def test_action_latency_wrapper_distribution_samples_deterministically():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.common.latency_wrapper import ActionLatencyWrapper

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

    from training.common.latency_wrapper import ActionLatencyWrapper

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

    from training.common.latency_wrapper import ActionLatencyWrapper

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


def test_action_latency_wrapper_advances_obs_stride_with_one_frame_noop():
    from training.common.latency_wrapper import ActionLatencyWrapper

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
    from training.common.latency_wrapper import ActionLatencyWrapper

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
    from training.common.latency_wrapper import ActionLatencyWrapper, RewardTraceWrapper

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

    from training.common.latency_wrapper import ActionLatencyWrapper

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
    wrapped = ActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        seed=11,
        env_reset_seed_base=500,
    )
    wrapped.reset()
    wrapped.reset()
    wrapped.reset(seed=999)
    assert env.seeds == [500, 501, 999]


def test_episode_metrics_wrapper_suppresses_life_loss_episode_event(tmp_path):
    from training.common.latency_wrapper import EpisodeMetricsWrapper

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

    from training.common.latency_wrapper import EpisodeMetricsWrapper

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


def test_episode_metrics_wrapper_keeps_learner_reward_clipped_and_episode_return_raw(tmp_path):
    import json

    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})]),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)

    assert reward == 1.0
    assert info["episodic_raw_return"] == 10.0
    assert info["episodic_return"] == 10.0
    assert info["episode_extra_stats"]["episode/raw_return"] == 10.0
    assert info["episode_extra_stats"]["episode/return"] == 10.0
    assert "episodic_clipped_return" not in info
    assert "episode/clipped_return" not in info["episode_extra_stats"]
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 10.0
    assert event["return"] == 10.0
    assert "clipped_return" not in event


def test_episode_metrics_wrapper_accumulates_raw_return_across_steps(tmp_path):
    import json

    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv(
                [
                    (10.0, False, False, {"lives": 3}),
                    (-5.0, False, False, {"lives": 3}),
                    (0.0, True, False, {"lives": 0}),
                ]
            ),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    wrapped.step(0)
    wrapped.step(0)
    _, _, _, _, info = wrapped.step(0)

    assert info["episodic_raw_return"] == 5.0
    assert info["episodic_return"] == 5.0
    assert "episodic_clipped_return" not in info
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 5.0
    assert event["return"] == 5.0
    assert "clipped_return" not in event


def test_episode_metrics_wrapper_does_not_emit_raw_return_on_life_loss(tmp_path):
    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 2})]),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
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
    from training.common.latency_wrapper import (
        EpisodeMetricsWrapper,
        RewardTraceWrapper,
    )

    metrics_path = tmp_path / "episodes.jsonl"
    wrapped = EpisodeMetricsWrapper(
        RewardTraceWrapper(
            _ScriptedAtariInfoEnv([(10.0, True, False, {"lives": 0})]),
            learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)),
        ),
        metrics_path=str(metrics_path),
    )

    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)

    assert info["is_true_episode_end"] is True
    assert info["episodic_raw_return"] == 10.0
    assert info["episodic_return"] == 10.0
    assert "episodic_clipped_return" not in info


def test_episode_metrics_wrapper_accumulates_return_across_life_losses(tmp_path):
    import json

    from training.common.latency_wrapper import EpisodeMetricsWrapper

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
    from training.common.latency_wrapper import EpisodeMetricsWrapper

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

    from training.common.latency_wrapper import EpisodeMetricsWrapper

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


def test_flappy_reward_trace_records_raw_return(tmp_path):
    import json

    from training.common.latency_wrapper import EpisodeMetricsWrapper, RewardTraceWrapper

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
    assert "episode/clipped_return" not in info["episode_extra_stats"]
    event = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert event["raw_return"] == 2.0
    assert "clipped_return" not in event


def test_episode_metrics_aggregator_logs_raw_visible_metrics(tmp_path):
    from training.common.episode_metrics import EpisodeMetricsAggregator

    aggregator = EpisodeMetricsAggregator(path=tmp_path / "episodes.jsonl", rolling_window=10)

    payload = aggregator.process_event(
        {
            "return": 30.0,
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
    assert [key for key in payload if "clipped" in key] == []


def test_episode_metrics_aggregator_logs_numeric_debug_metrics_and_omits_done_reason(tmp_path):
    from training.common.episode_metrics import EpisodeMetricsAggregator

    aggregator = EpisodeMetricsAggregator(path=tmp_path / "episodes.jsonl", rolling_window=10)

    payload = aggregator.process_event(
        {
            "return": 12.0,
            "raw_return": 12.0,
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
    assert "episode/terminated_rate" not in payload
    assert "episode/truncated_rate" not in payload
    assert "episode/timeout_rate" not in payload
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
        cfg=SimpleNamespace(
            env_fps=60,
            obs_fps=15,
            frame_stack=1,
            latency_type="zero",
            fixed_latency_ms=None,
            mean_latency_ms=None,
            std_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            latency_seed=None,
            add_latency_info=False,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_episode_steps=0,
            episode_metrics_path="",
        ),
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
        seed=0,
        env_fps=60,
        obs_fps=15,
        frame_stack=4,
        terminal_on_life_loss=True,
        clip_reward=True,
        latency_type="zero",
        fixed_latency_ms=None,
        mean_latency_ms=None,
        std_latency_ms=None,
        min_latency_ms=None,
        max_latency_ms=None,
        latency_seed=None,
        add_latency_info=False,
        hold_policy="hold",
        ordering_policy="latest_ready",
        max_pending_actions=None,
        max_episode_steps=0,
        episode_metrics_path="",
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


def test_flappy_cpu_render_reference_env_is_sample_factory_compatible(monkeypatch):
    from sample_factory.algo.utils.env_info import extract_env_info

    from training.flappy_sf import integration

    class FakeFlappyGymnasium:
        @staticmethod
        def make(gym_id, **kwargs):
            assert gym_id == "FlappyBird-v0"
            assert kwargs["render_mode"] == "rgb_array"
            return _RenderedRgbEnv()

    monkeypatch.setattr(integration, "_ENVS_REGISTERED", False)
    monkeypatch.setattr(integration.gym, "make", FakeFlappyGymnasium.make)

    cfg = SimpleNamespace(
        env=integration.SF_ENV_NAME,
        env_gpu_actions=False,
        env_gpu_observations=False,
        seed=0,
        gym_id="FlappyBird-v0",
        env_fps=30,
        obs_fps=30,
        use_lidar=False,
        normalize_obs=True,
        audio_on=False,
        pipe_gap=100,
        bird_color="yellow",
        pipe_color="green",
        background="day",
        debug=False,
        debug_timelimit_diagnostics=False,
        screen_size="",
        score_limit=-1,
        obs_resize="84,84",
        frame_stack=4,
        max_episode_steps=0,
        latency_type="zero",
        fixed_latency_ms=None,
        mean_latency_ms=None,
        std_latency_ms=None,
        min_latency_ms=None,
        max_latency_ms=None,
        latency_seed=None,
        add_latency_info=False,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        max_pending_actions=None,
        episode_metrics_path="",
        use_gpu_render=False,
    )

    env = integration.make_reference_env(cfg)
    try:
        assert env.num_agents == 1
        assert env.is_multiagent is True
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert env.observation_space["obs"].shape == (12, 84, 84)
        assert env.action_space == _RenderedRgbEnv.action_space

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


def test_doom_frame_stack_wrapper_does_not_concatenate_on_reset_or_step(monkeypatch):
    from gymnasium.spaces import Box, Discrete

    from training.deadly_corridor_sf import integration

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

    def fail_concatenate(*args, **kwargs):
        raise AssertionError("Doom frame stack should reuse a preallocated buffer")

    monkeypatch.setattr(integration.np, "concatenate", fail_concatenate)
    wrapped = integration.DoomFrameStackWrapper(ImageEnv(), num_frames=4)

    obs, _ = wrapped.reset()
    assert obs.shape == (12, 2, 2)
    assert np.all(obs == 1)
    obs, *_ = wrapped.step(0)
    assert np.all(obs[:9] == 1)
    assert np.all(obs[9:] == 2)


def test_doom_frame_stack_wrapper_returns_stable_observation_snapshots():
    from gymnasium.spaces import Box, Dict, Discrete

    from training.deadly_corridor_sf.integration import DoomFrameStackWrapper

    class DictImageEnv(gym.Env):
        observation_space = Dict(
            {
                "obs": Box(low=0, high=255, shape=(1, 2, 2), dtype=np.uint8),
                "state": Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            }
        )
        action_space = Discrete(2)

        def __init__(self):
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            self.step_count = 0
            return {
                "obs": np.ones((1, 2, 2), dtype=np.uint8),
                "state": np.array([7], dtype=np.uint8),
            }, {}

        def step(self, action):
            self.step_count += 1
            return {
                "obs": np.full((1, 2, 2), self.step_count + 1, dtype=np.uint8),
                "state": np.array([self.step_count], dtype=np.uint8),
            }, 0.0, False, False, {}

    wrapped = DoomFrameStackWrapper(DictImageEnv(), num_frames=3)

    reset_obs, _ = wrapped.reset()
    first_step_obs, *_ = wrapped.step(0)
    first_step_snapshot = first_step_obs["obs"].copy()
    second_step_obs, *_ = wrapped.step(0)

    assert wrapped.observation_space["obs"].shape == (3, 2, 2)
    assert np.all(reset_obs["obs"] == 1)
    assert np.array_equal(first_step_obs["obs"], first_step_snapshot)
    assert np.all(second_step_obs["obs"][:1] == 1)
    assert np.all(second_step_obs["obs"][1:2] == 2)
    assert np.all(second_step_obs["obs"][2:] == 3)
    assert np.array_equal(second_step_obs["state"], np.array([2], dtype=np.uint8))


def test_doom_raw_rgb_frame_stack_wrapper_reports_screen_buffer_without_changing_observation():
    from gymnasium.spaces import Box, Discrete

    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackInfoWrapper

    class FakeState:
        def __init__(self, value):
            self.screen_buffer = np.zeros((3, 6, 5), dtype=np.uint8)
            self.screen_buffer[0] = value
            self.screen_buffer[1] = value + 1
            self.screen_buffer[2] = value + 2

    class FakeGame:
        def __init__(self, env):
            self.env = env

        def get_state(self):
            return FakeState(self.env.step_count + 10)

    class ImageEnv(gym.Env):
        observation_space = Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        action_space = Discrete(2)

        def __init__(self):
            self.step_count = 0
            self.game = FakeGame(self)

        def reset(self, *, seed=None, options=None):
            self.step_count = 0
            return np.ones((3, 2, 2), dtype=np.uint8), {}

        def step(self, action):
            self.step_count += 1
            return np.full((3, 2, 2), 7, dtype=np.uint8), 0.0, False, False, {}

    def raw_frame(env):
        return np.transpose(env.unwrapped.game.get_state().screen_buffer, (1, 2, 0))

    wrapped = RawRgbFrameStackInfoWrapper(ImageEnv(), num_frames=2, raw_frame_fn=raw_frame)

    obs, info = wrapped.reset()
    step_obs, *_rest, step_info = wrapped.step(0)

    assert obs.shape == (3, 2, 2)
    assert step_obs.shape == (3, 2, 2)
    assert info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].shape == (2, 6, 5, 3)
    assert info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][0, 0, 0].tolist() == [10, 11, 12]
    assert step_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0].tolist() == [[10, 11, 12], [11, 12, 13]]


def test_raw_rgb_frame_stack_wrapper_keeps_last_stack_on_terminal_step():
    from gymnasium.spaces import Box, Discrete

    from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackInfoWrapper

    class FakeState:
        def __init__(self):
            self.screen_buffer = np.zeros((3, 6, 5), dtype=np.uint8)
            self.screen_buffer[0] = 21
            self.screen_buffer[1] = 22
            self.screen_buffer[2] = 23

    class FakeGame:
        def __init__(self, env):
            self.env = env

        def get_state(self):
            if self.env.terminal:
                return None
            return FakeState()

    class TerminalEnv(gym.Env):
        observation_space = Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8)
        action_space = Discrete(2)

        def __init__(self):
            self.terminal = False
            self.game = FakeGame(self)

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.terminal = False
            return np.ones((3, 2, 2), dtype=np.uint8), {}

        def step(self, action):
            del action
            self.terminal = True
            return np.full((3, 2, 2), 7, dtype=np.uint8), 0.0, True, False, {}

    def raw_frame(env):
        return np.transpose(env.unwrapped.game.get_state().screen_buffer, (1, 2, 0))

    wrapped = RawRgbFrameStackInfoWrapper(TerminalEnv(), num_frames=2, raw_frame_fn=raw_frame)

    _, reset_info = wrapped.reset()
    _, _, terminated, truncated, step_info = wrapped.step(0)

    assert terminated is True
    assert truncated is False
    assert step_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].tolist() == reset_info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].tolist()


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


def test_same_latency_sweep_generator_cli_overrides_train_wandb_group(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"

    generate_same_latency_sweep(
        [
            "--envs",
            "demon_attack",
            "--seeds",
            "0",
            "--demon-attack-latencies",
            "4",
            "--generated-root",
            str(generated_root),
            "--train-dir",
            "results/checkpoints/sample_factory_runs",
            "--output-root",
            str(output_root),
            "--device",
            "gpu",
            "--wandb-group",
            "dist-sweeps-v1",
        ]
    )

    train_path = generated_root / "demon_attack" / "latency_4" / "seed_0" / "train.yaml"
    train_loaded = yaml.safe_load(train_path.read_text(encoding="utf-8"))

    assert train_loaded["logging"]["wandb_group"] == "dist-sweeps-v1"
    assert train_loaded["logging"]["wandb_project"] == "latency-sensitive-bench"


def test_same_latency_sweep_generator_emits_sample_factory_train_configs():
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
    expected_eval_suites = full_cfg["evaluation"]["eval_suites"]
    assert train_cfg["training"]["train_for_env_steps"] == 1234
    assert train_cfg["latency"]["type"] == "fixed"
    assert train_cfg["latency"]["fixed_latency_ms"] == pytest.approx(4 * 1000.0 / 60.0)
    assert train_cfg["evaluation"]["eval_suites"] == expected_eval_suites
    assert train_cfg["logging"]["wandb_name"] == "demon-attack-l4-s0"


@pytest.mark.parametrize(
    ("env_name", "max_steps", "expected_policy", "expected_ms", "expected_eval_max_steps"),
    (
        ("flappy", 1800, "flappy_sf", 100.0, 3600),
        ("demon_attack", 25000, "demon_attack_sf", 50.0, 3600),
        ("deadly_corridor", 2100, "deadly_corridor_sf", 50.0, 30000),
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
    assert math.isclose(loaded["latency"]["fixed_latency_ms"], expected_ms, rel_tol=1e-9)
    assert loaded["logging"]["output_dir"] == str(
        tmp_path / "outputs" / "latency_bench" / env_name / "train_latency_3" / "seed_0"
    )
    assert loaded["logging"]["wandb_name"] == full_cfg["logging"]["wandb_name"]
    assert loaded["logging"]["wandb_project"] == full_cfg["logging"]["wandb_project"]
    assert loaded["logging"]["wandb_group"] == full_cfg["logging"]["wandb_group"]
    assert loaded["logging"]["wandb_tags"] == full_cfg["logging"]["wandb_tags"]
    assert loaded["evaluation"]["eval_suites"] == full_cfg["evaluation"]["eval_suites"]
    assert loaded["evaluation"]["eval_max_steps"] == expected_eval_max_steps
    assert "max_episode_steps" not in loaded["env"]
    if env_name == "demon_attack":
        assert loaded["env"]["terminal_on_life_loss"] is True
        assert loaded["env"]["clip_reward"] is False
    if env_name == "deadly_corridor":
        assert loaded["env"]["env_fps"] == 60
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
    assert bench_cfg["evaluation"]["eval_suites"] == full_cfg["evaluation"]["eval_suites"]
    assert bench_cfg["logging"]["output_dir"] == str(
        tmp_path / "outputs" / "latency_bench" / "flappy" / "train_latency_0" / "seed_0"
    )
    assert _latency_steps_to_ms(full_cfg, 3) == 100.0


def test_same_latency_sweep_generator_preserves_deadly_corridor_latency_bench_eval_latencies_for_latency_zero(tmp_path):
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
    bench_cfg = _build_latency_bench_eval_config(
        full_config=full_cfg,
        env_name="deadly_corridor",
        seed=0,
        latency=0,
        max_steps=2100,
        output_root=str(tmp_path / "outputs"),
        eval_episodes=100,
    )
    expected_eval_suites = full_cfg["evaluation"]["eval_suites"]

    assert full_cfg["latency"]["type"] == "zero"
    assert bench_cfg["latency"]["type"] == "zero"
    assert bench_cfg["evaluation"]["eval_suites"] == expected_eval_suites
    assert "max_episode_steps" not in bench_cfg["env"]
    assert bench_cfg["env"]["env_fps"] == 60
    assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]


def test_same_latency_sweep_generator_writes_train_and_latency_bench_eval_configs_and_commands(tmp_path, monkeypatch):
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
        "deadly_corridor": 50.0,
    }
    expected_eval_suites = {
        env_name: load_config(same_latency_sweep.BASE_CONFIGS[env_name])["evaluation"]["eval_suites"]
        for env_name in expected_policy_types
    }
    for env_name, policy_type in expected_policy_types.items():
        for latency in (0, 3):
            run_dir = generated_root / env_name / f"latency_{latency}" / "seed_0"
            train_path = run_dir / "train.yaml"
            assert train_path.is_file()
            bench_path = run_dir / "latency_bench_eval.yaml"
            assert bench_path.is_file()

            train_loaded = yaml.safe_load(train_path.read_text(encoding="utf-8"))
            loaded = load_config(bench_path)
            assert train_loaded["evaluation"]["eval_suites"] == expected_eval_suites[env_name]
            assert train_loaded["latency"]["type"] == ("zero" if latency == 0 else "fixed")
            assert loaded["experiment"]["name"] == f"{env_name}_sf_same_latency_l{latency}_seed0"
            assert loaded["backend"]["train_dir"] == "results/checkpoints/sample_factory_runs"
            assert loaded["policy"]["type"] == policy_type
            assert loaded["policy"]["checkpoint_path"] == ""
            assert loaded["evaluation"]["eval_suites"] == expected_eval_suites[env_name]
            assert set(loaded["experiment"]) == {"name", "seed", "num_episodes"}
            assert "max_episode_steps" not in loaded["env"]
            if env_name == "flappy":
                assert "max_episode_steps" not in train_loaded["env"]
            if env_name == "demon_attack":
                assert train_loaded["env"]["max_episode_steps"] == 25000
                assert loaded["env"]["terminal_on_life_loss"] is True
                assert loaded["env"]["clip_reward"] is False
            assert loaded["logging"]["output_dir"] == str(
                output_root / "latency_bench" / env_name / f"train_latency_{latency}" / "seed_0"
            )
            if env_name == "deadly_corridor":
                assert train_loaded["env"]["env_fps"] == 60
                assert loaded["env"]["env_fps"] == 60
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


def test_same_latency_generator_writes_single_normal_distribution_run(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"

    generate_same_latency_sweep(
        [
            "--envs",
            "demon_attack",
            "--seeds",
            "0",
            "--latency-distribution",
            "normal",
            "--latency-mean-frames",
            "5.0",
            "--latency-std-frames",
            "2",
            "--latency-min-frames",
            "0",
            "--latency-max-frames",
            "15",
            "--generated-root",
            str(generated_root),
            "--train-dir",
            "results/checkpoints/sample_factory_runs",
            "--output-root",
            str(output_root),
            "--device",
            "gpu",
        ]
    )

    label = "normal_mean_5_std_2_min_0_max_15"
    run_dir = generated_root / "demon_attack" / f"latency_{label}" / "seed_0"
    train_loaded = yaml.safe_load((run_dir / "train.yaml").read_text(encoding="utf-8"))
    bench_loaded = load_config(run_dir / "latency_bench_eval.yaml")
    manifest = (generated_root / "manifest.tsv").read_text(encoding="utf-8")

    assert train_loaded["latency"]["type"] == "normal_distribution"
    assert train_loaded["latency"]["mean_latency_ms"] == pytest.approx(5.0 * 1000.0 / 60.0)
    assert train_loaded["latency"]["std_latency_ms"] == pytest.approx(2.0 * 1000.0 / 60.0)
    assert train_loaded["latency"]["min_latency_ms"] == pytest.approx(0.0)
    assert train_loaded["latency"]["max_latency_ms"] == pytest.approx(15.0 * 1000.0 / 60.0)
    assert "train_values" not in train_loaded["latency"]
    assert "train_distribution" not in train_loaded["latency"]
    assert "train_mode" not in train_loaded["latency"]
    assert bench_loaded["latency"]["type"] == "normal_distribution"
    assert bench_loaded["latency"]["mean_latency_ms"] == pytest.approx(5.0 * 1000.0 / 60.0)
    assert bench_loaded["latency"]["std_latency_ms"] == pytest.approx(2.0 * 1000.0 / 60.0)
    assert bench_loaded["latency"]["min_latency_ms"] == pytest.approx(0.0)
    assert bench_loaded["latency"]["max_latency_ms"] == pytest.approx(15.0 * 1000.0 / 60.0)
    assert f"demon_attack\t{label}\t0" in manifest


def test_same_latency_generator_writes_single_uniform_distribution_bench_config(tmp_path):
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"

    generate_same_latency_sweep(
        [
            "--envs",
            "deadly_corridor",
            "--seeds",
            "0",
            "--latency-distribution",
            "uniform",
            "--latency-min-frames",
            "0",
            "--latency-max-frames",
            "15",
            "--generated-root",
            str(generated_root),
            "--train-dir",
            "results/checkpoints/sample_factory_runs",
            "--output-root",
            str(output_root),
            "--device",
            "gpu",
        ]
    )

    label = "uniform_min_0_max_15"
    run_dir = generated_root / "deadly_corridor" / f"latency_{label}" / "seed_0"
    loaded = load_config(run_dir / "latency_bench_eval.yaml")

    assert loaded["latency"]["type"] == "uniform_distribution"
    assert loaded["latency"]["min_latency_ms"] == pytest.approx(0.0)
    assert loaded["latency"]["max_latency_ms"] == pytest.approx(15.0 * 1000.0 / float(loaded["env"]["env_fps"]))
    assert loaded["logging"]["output_dir"] == str(
        output_root / "latency_bench" / "deadly_corridor" / f"train_latency_{label}" / "seed_0"
    )
