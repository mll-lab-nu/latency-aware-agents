from __future__ import annotations

import importlib
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

import gymnasium as gym
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.core.latency_distribution import parse_latency_distribution
from training.common.config_adapter import (
    apply_runtime_logging_config,
    apply_training_overrides,
    build_common_sf_args,
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
        info = dict(info)
        info["command_admitted"] = True
        info["observation_dropped"] = False
        return obs, float(reward), bool(terminated), bool(truncated), info


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


@pytest.mark.parametrize(
    "module_name",
    [
        "training.flappy_sf.integration",
        "training.demon_attack_sf.integration",
        "training.deadly_corridor_sf.integration",
    ],
)
def test_sample_factory_integrations_split_train_and_eval_parser_contract(module_name):
    module = importlib.import_module(module_name)
    argv = [
        "--algo",
        "APPO",
        "--env",
        module.SF_ENV_NAME,
        "--experiment",
        "parser_contract",
    ]
    parser = module.SAMPLE_FACTORY_CONFIG_PARSER

    train_cfg = parser.parse_train(argv)
    eval_cfg = parser.parse_eval([*argv, "--policy_index", "2"])

    assert not hasattr(train_cfg, "policy_index")
    assert eval_cfg.policy_index == 2


def test_deadly_corridor_parser_records_raw_frame_checkpoint_overrides():
    from training.deadly_corridor_sf import integration

    cfg = integration.SAMPLE_FACTORY_CONFIG_PARSER.parse_train(
        [
            "--algo",
            "APPO",
            "--env",
            integration.SF_ENV_NAME,
            "--experiment",
            "raw_frame_overrides",
        ]
    )

    assert cfg.cli_args["env_frameskip"] == 1
    assert cfg.cli_args["eval_env_frameskip"] == 1


@pytest.mark.parametrize(
    "module_name",
    [
        "training.flappy_sf.integration",
        "training.demon_attack_sf.integration",
        "training.deadly_corridor_sf.integration",
    ],
)
def test_sample_factory_eval_parser_adds_policy_index_to_legacy_checkpoint(module_name, tmp_path):
    from sample_factory.cfg.arguments import load_from_checkpoint

    module = importlib.import_module(module_name)
    experiment = module_name.split(".")[-2]
    experiment_dir = tmp_path / experiment
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "algo": "APPO",
                "env": module.SF_ENV_NAME,
                "experiment": experiment,
                "train_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    cfg = module.SAMPLE_FACTORY_CONFIG_PARSER.parse_eval(
        [
            "--algo",
            "APPO",
            "--env",
            module.SF_ENV_NAME,
            "--experiment",
            experiment,
            "--train_dir",
            str(tmp_path),
        ]
    )

    loaded_cfg = load_from_checkpoint(cfg)

    assert loaded_cfg.policy_index == 0


def test_build_common_sf_args_emits_non_zero_latency_and_eval_settings():
    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["experiment"]["name"] = "deadly_corridor_sf_fixed_latency_4"
    cfg["latency"] = {"method": "fixed", "fixed_latency_ms": 4 * 1000.0 / 35.0}
    cfg["training"]["fasttd3_actor_action_l2"] = 1.0
    cfg["training"]["fasttd3_train_for_optimizer_steps"] = 500_000
    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")
    assert "--latency-type" in args
    assert args[args.index("--latency-type") + 1] == "fixed"
    assert "--fixed-latency-ms" in args
    assert "--eval-latency-raw-frame-values" not in args
    assert "--eval-max-steps" in args
    assert "--experiment" in args
    assert "deadly_corridor_sf_fixed_latency_4" in args
    assert args[args.index("--fasttd3_actor_action_l2") + 1] == "1.0"
    assert args[args.index("--fasttd3_train_for_optimizer_steps") + 1] == "500000"


def test_build_common_sf_args_emits_initial_model_path():
    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["backend"]["initial_model_path"] = "teacher.pth"

    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")

    assert args[args.index("--initial_model_path") + 1] == "teacher.pth"


def test_build_common_sf_args_maps_cuda_to_sample_factory_gpu():
    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["latency"] = {"method": "fixed", "fixed_latency_ms": 4 * 1000.0 / 35.0}
    cfg["backend"]["device"] = "cuda"

    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")

    assert args[args.index("--device") + 1] == "gpu"


def test_build_common_sf_args_emits_iid_profile_settings(tmp_path):
    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["latency"] = {
        "method": "iid",
        "profile_path": str(tmp_path / "profile.json"),
        "profile_worker_slot": 0,
        "seed": 42,
    }

    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")

    assert args[args.index("--latency-type") + 1] == "iid"
    assert args[args.index("--latency-profile-path") + 1] == str(tmp_path / "profile.json")
    assert args[args.index("--latency-profile-worker-slot") + 1] == "0"


def test_flappy_sf_cli_maps_explicit_env_max_episode_steps():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["env"]["max_episode_steps"] = 1800

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--max_episode_steps") + 1] == "1800"


def test_flappy_sf_cli_actor_mask_is_opt_in():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    args = build_cli_args_from_config(cfg)
    assert args[args.index("--mask-unadmitted-actor") + 1] == "False"

    cfg["training"]["mask_unadmitted_actor"] = True
    args = build_cli_args_from_config(cfg)
    assert args[args.index("--mask-unadmitted-actor") + 1] == "True"


def test_flappy_sf_cli_passes_learning_rate_schedule():
    from training.flappy_sf.integration import SAMPLE_FACTORY_CONFIG_PARSER, build_cli_args_from_config

    cfg = load_config("configs/base/flappy.yaml")
    cfg["training"]["lr_schedule"] = "linear_decay_floor"
    cfg["training"]["lr_schedule_min_lr"] = 5e-5
    args = build_cli_args_from_config(cfg)

    assert args[args.index("--lr_schedule") + 1] == "linear_decay_floor"
    assert args[args.index("--lr_schedule_min_lr") + 1] == "5e-05"
    parsed = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(args)
    assert parsed.lr_schedule == "linear_decay_floor"
    assert parsed.lr_schedule_min_lr == 5e-5


def test_flappy_linear_lr_resume_uses_checkpoint_optimizer_step():
    from sample_factory.algo.learning.learner import get_lr_scheduler
    from sample_factory.utils.attr_dict import AttrDict

    cfg = AttrDict(
        lr_schedule="linear_decay",
        train_for_env_steps=25_000_000,
        batch_size=4096,
        num_epochs=8,
        learning_rate=2.5e-4,
    )
    updates = cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs
    saved_step = 19_584
    saved_lr = cfg.learning_rate * (1 - saved_step / updates)

    assert get_lr_scheduler(cfg, saved_step).update(saved_lr, []) == pytest.approx(
        cfg.learning_rate * (1 - (saved_step + 1) / updates)
    )


def test_flappy_linear_lr_floor_preserves_early_schedule():
    from sample_factory.algo.learning.learner import get_lr_scheduler
    from sample_factory.utils.attr_dict import AttrDict

    cfg = AttrDict(
        lr_schedule="linear_decay_floor",
        lr_schedule_min_lr=5e-5,
        train_for_env_steps=25_000_000,
        batch_size=4096,
        num_epochs=8,
        learning_rate=2.5e-4,
    )
    regular = get_lr_scheduler(AttrDict({**cfg, "lr_schedule": "linear_decay"}), 19_584)
    floored = get_lr_scheduler(cfg, 19_584)
    assert floored.update(0, []) == pytest.approx(regular.update(0, []))

    updates = cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs
    assert get_lr_scheduler(cfg, updates - 1).update(0, []) == cfg.lr_schedule_min_lr


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
    cfg = load_config("configs/base/flappy.yaml")
    cfg["logging"]["wandb_name"] = "flappy-bird-l0-s0"

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    assert "--wandb_unique_id" not in args


def test_build_common_sf_args_emits_wandb_tags_as_multiple_values():
    cfg = load_config("configs/base/flappy.yaml")
    cfg["logging"]["wandb_tags"] = ["observation_stride", "flappy", "frame_stack"]
    cfg["logging"]["wandb_project"] = "test-project"
    cfg["logging"]["wandb_entity"] = "test-team"

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    idx = args.index("--wandb_tags")
    assert args[idx : idx + 4] == ["--wandb_tags", "observation_stride", "flappy", "frame_stack"]
    assert args[args.index("--wandb_user") + 1] == "test-team"


def test_build_common_sf_args_splits_legacy_comma_wandb_tags():
    cfg = load_config("configs/base/flappy.yaml")
    cfg["logging"]["wandb_tags"] = "observation_stride,flappy,frame_stack"
    cfg["logging"]["wandb_project"] = "test-project"

    args = build_common_sf_args(cfg, sf_env_name="flappy_bird")

    idx = args.index("--wandb_tags")
    assert args[idx : idx + 4] == ["--wandb_tags", "observation_stride", "flappy", "frame_stack"]


def test_runtime_logging_config_survives_checkpoint_load(tmp_path):
    from sample_factory.cfg.arguments import load_from_checkpoint

    experiment_dir = tmp_path / "curriculum"
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": "curriculum",
                "train_dir": str(tmp_path),
                "output_dir": "stage00",
                "wandb_unique_id": "stage00",
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(experiment="curriculum", train_dir=str(tmp_path))

    apply_runtime_logging_config(
        cfg,
        {
            "output_dir": "outputs/latency_sweeps/train/flappy/latency_0/seed_0",
            "wandb_name": "flappy-bird-l0-s0",
        },
    )

    loaded_cfg = load_from_checkpoint(cfg)

    assert (
        loaded_cfg.output_dir
        == "outputs/latency_sweeps/train/flappy/latency_0/seed_0"
    )
    assert loaded_cfg.wandb_unique_id == "flappy-bird-l0-s0"


def test_apply_training_overrides_updates_run_name_train_dir_and_mode():
    cfg = load_config("configs/base/flappy.yaml")
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


@pytest.mark.parametrize("missing_section", ["backend", "experiment", "logging"])
def test_apply_training_overrides_requires_top_level_sections(missing_section):
    config = load_config("configs/base/flappy.yaml")
    del config[missing_section]

    with pytest.raises(KeyError, match=missing_section):
        apply_training_overrides(config)


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
    from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
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
    from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.flappy_sf.integration import FlappyImageObsWrapper

    wrapped = FlappyImageObsWrapper(_RenderedRgbEnv(shape=(6, 5, 3)), obs_resize=(2, 2))

    _, info = wrapped.reset()

    assert ENV_RAW_RGB_FRAME_STACK_INFO_KEY not in info


def test_flappy_frame_stack_export_advances_raw_rgb_stack_oldest_to_newest():
    from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
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


def test_flappy_sf_shape_mismatch_fails_fast():
    from latency_bench.policy.sample_factory import validate_sample_factory_obs_shape

    with pytest.raises(ValueError, match="observation shape mismatch"):
        validate_sample_factory_obs_shape(
            {"obs": np.zeros((1, 12, 512, 288), dtype=np.uint8)},
            expected_shape=(12, 84, 84),
        )


def test_flappy_sf_update_ratio_metadata():
    from training.common.config_adapter import estimate_optimizer_steps_per_1m_env_steps

    assert estimate_optimizer_steps_per_1m_env_steps(
        {"batch_size": 1024, "num_epochs": 4, "num_batches_per_epoch": 2}
    ) == pytest.approx(3906.25)


def test_common_sf_args_passes_latency_distribution_without_legacy_train_values():
    config = {
        "executor": {"mode": "simulated"},
        "policy": {"type": "sample_factory"},
        "experiment": {"name": "dist_run", "seed": 3},
        "backend": {"type": "sample_factory", "train_dir": "runs", "run_mode": "train"},
        "env": {"name": "demon_attack", "env_fps": 60, "obs_fps": 15},
        "executor": {"mode": "simulated", "simulated_worker_capacity": 1},
        "latency": {
            "method": "uniform",
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
    assert args[args.index("--simulated-worker-capacity") + 1] == "1"


@pytest.mark.parametrize(
    ("module_name", "register_args_name"),
    (
        ("training.flappy_sf.integration", "add_flappy_env_args"),
        ("training.demon_attack_sf.integration", "add_demon_attack_env_args"),
        ("training.deadly_corridor_sf.integration", "add_latency_args"),
        ("training.gymnasium_sf.integration", "add_gymnasium_env_args"),
    ),
)
def test_all_sample_factory_integrations_register_worker_capacity(module_name, register_args_name):
    import argparse

    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser()
    getattr(module, register_args_name)(parser)

    capacity_actions = [
        action for action in parser._actions if action.dest == "simulated_worker_capacity"
    ]
    assert len(capacity_actions) == 1
    assert capacity_actions[0].type is int


def test_sample_factory_latency_params_use_bench_ceil_for_fixed_ms():
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="fixed",
        fixed_latency_ms=(1000.0 / 30.0) + 0.001,
        simulated_worker_capacity=None,
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=30)

    assert params.fixed_latency_raw_frames == 2
    assert params.latency_distribution_spec is None


def test_sample_factory_latency_distribution_preserves_fractional_bounds():
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="uniform_distribution",
        min_latency_ms=50.0,
        max_latency_ms=(3.5 * 1000.0 / 30.0),
        simulated_worker_capacity=None,
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=30)
    spec = params.latency_distribution_spec

    assert params.fixed_latency_raw_frames == 0
    assert spec is not None
    assert spec.min == pytest.approx(1.5)
    assert spec.max == pytest.approx(3.5)


def test_sample_factory_latency_params_support_temporal_profile_and_capacity(tmp_path):
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="profile_sample",
        latency_profile_path=str(tmp_path / "profile.json"),
        latency_profile_worker_slot=0,
        simulated_worker_capacity=1,
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=50)

    assert params.profile_method == "temporal"
    assert params.profile_path == str(tmp_path / "profile.json")
    assert params.profile_worker_slot == 0
    assert params.simulated_worker_capacity == 1


@pytest.mark.parametrize("env_name", ("doom_deadly_corridor", "doom_defend_the_line"))
def test_vizdoom_scalar_integrations_forward_worker_capacity(monkeypatch, env_name):
    from sf_examples.vizdoom.doom import doom_utils
    from training.deadly_corridor_sf import integration

    base_env = _ActionTraceEnv()
    monkeypatch.setattr(doom_utils, "make_doom_env_from_spec", lambda *args, **kwargs: base_env)
    monkeypatch.setattr(integration, "ensure_vizdoom_init_guard_installed", lambda *args, **kwargs: None)
    cfg = SimpleNamespace(
        disable_reward_scaling=False,
        env_fps=30,
        obs_fps=30,
        frame_stack=1,
        export_env_raw_rgb_frames=False,
        max_episode_steps=0,
        seed=0,
        latency_seed=123,
        latency_type="fixed",
        fixed_latency_ms=100.0,
        simulated_worker_capacity=1,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        add_latency_info=True,
        max_pending_actions=None,
        episode_metrics_path="",
    )
    env = integration.make_latency_doom_env_from_spec(
        SimpleNamespace(name=env_name, reward_scaling=1.0),
        env_name,
        cfg,
        SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )
    try:
        env.reset(seed=123)
        first_info = env.step(1)[4]
        second_info = env.step(1)[4]
    finally:
        env.close()

    assert first_info["command_admitted"] is True
    assert second_info["command_admitted"] is False
    assert second_info["observation_dropped"] is True
    assert base_env.step_count == 2


def test_action_latency_wrapper_queues_absolute_g1_requests_with_configured_noop():
    import gymnasium as gym
    from gymnasium.spaces import Box
    from training.gymnasium_sf.integration import G1RawRequestWrapper
    from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

    class ActionTargetEnv(gym.Env):
        observation_space = Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        action_space = Box(low=-10, high=10, shape=(43,), dtype=np.float32)
        _raw_learn_hands = True

        def __init__(self):
            self.actions = []
            self._control_mode = "direct_absolute_position"

        def reset(self, *, seed=None, options=None):
            self.actions.clear()
            return np.zeros(1, dtype=np.float32), {"action_target": np.full(43, 7.0)}

        def g1_absolute_request_from_raw_action(self, action):
            return 2.0 + 2.0 * action

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    base_env = ActionTargetEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        fixed_latency_raw_frames=2,
        env_fps=30,
        obs_fps=30,
        add_latency_info=True,
        noop_action=base_env.g1_absolute_request_from_raw_action(np.zeros(43, dtype=np.float32)),
    )
    wrapped = G1RawRequestWrapper(wrapped)
    try:
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.ones(43, dtype=np.float32))
        np.testing.assert_allclose(info["issued_action"], np.full(43, 4.0))
        np.testing.assert_array_equal(wrapped.action_space.low, np.full(43, -1.0))
        np.testing.assert_array_equal(wrapped.action_space.high, np.full(43, 1.0))
        wrapped.step(np.ones(43, dtype=np.float32))
        wrapped.step(np.ones(43, dtype=np.float32))
    finally:
        wrapped.close()

    np.testing.assert_allclose(base_env.actions, np.repeat([[2.0], [2.0], [4.0]], 43, axis=1))


def test_sample_factory_latency_params_support_mixed_distribution():
    from training.common.latency_config import sample_factory_latency_params_from_flat_config

    cfg = SimpleNamespace(
        latency_type="mixed_uniform_distribution",
        min_latency_ms=50.0,
        max_latency_ms=(3.5 * 1000.0 / 30.0),
        simulated_worker_capacity=None,
    )

    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=30)

    assert params.fixed_latency_raw_frames == 0
    assert params.latency_distribution_spec is not None
    assert params.latency_distribution_spec.type == "uniform"
    assert params.sample_distribution_per_episode is True


def test_action_latency_wrapper_sums_raw_reward_across_observation_stride():
    from latency_bench.envs.latency_wrapper import ActionLatencyWrapper
    from training.common.metrics_wrappers import RewardTraceWrapper

    wrapped = ActionLatencyWrapper(
        RewardTraceWrapper(_RawRewardTraceEnv([1.0, 2.0, 3.0])),
        noop_action=0,
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


def test_episode_metrics_wrapper_suppresses_life_loss_episode_event(tmp_path):
    from training.common.metrics_wrappers import EpisodeMetricsWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper, RewardTraceWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper, RewardTraceWrapper

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
    from training.common.metrics_wrappers import EpisodeMetricsWrapper, RewardTraceWrapper

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
    from training.common.metrics_wrappers import EpisodeMetricsWrapper, RewardTraceWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper

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
    from training.common.metrics_wrappers import EpisodeMetricsWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper

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

    from training.common.metrics_wrappers import EpisodeMetricsWrapper, RewardTraceWrapper

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
            "submitted_observation_count": 2,
            "dropped_observation_count": 1,
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
    assert payload["episode_train/latest_observation_drop_ratio"] == pytest.approx(1.0 / 3.0)
    assert payload["episode_train/rolling_observation_drop_ratio"] == pytest.approx(1.0 / 3.0)
    assert payload["episode_train/cumulative_observation_drop_ratio"] == pytest.approx(1.0 / 3.0)
    assert [key for key in payload if "clipped" in key] == []


def test_episode_metrics_resume_reads_only_new_events(tmp_path):
    from training.common.episode_metrics import (
        EpisodeMetricsAggregator,
        append_episode_metrics_event,
    )

    path = tmp_path / "episodes.jsonl"
    old_event = {
        "return": 1.0,
        "length": 1,
        "raw_frames": 1,
        "submitted_observation_count": 1,
        "dropped_observation_count": 0,
    }
    new_event = {
        "return": 2.0,
        "length": 2,
        "raw_frames": 2,
        "submitted_observation_count": 1,
        "dropped_observation_count": 1,
    }
    append_episode_metrics_event(path, old_event)
    aggregator = EpisodeMetricsAggregator(path=path, start_at_end=True)

    append_episode_metrics_event(path, new_event)

    assert aggregator.poll_once() == 1
    assert aggregator.completed_episodes == 1
    assert list(aggregator.return_window) == [2.0]


def test_episode_metrics_reader_waits_for_complete_locked_event(tmp_path):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor

    from training.common.episode_metrics import EpisodeMetricsAggregator

    path = tmp_path / "episodes.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write('{"return": 1.0,')
        handle.flush()
        aggregator = EpisodeMetricsAggregator(path=path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(aggregator.poll_once)
            assert not result.done()
            handle.write(
                '"length": 1, "raw_frames": 1, '
                '"submitted_observation_count": 1, "dropped_observation_count": 0}\n'
            )
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            assert result.result() == 1
    assert list(aggregator.return_window) == [1.0]


def test_episode_metrics_aggregator_logs_numeric_debug_metrics_and_omits_done_reason(tmp_path):
    from training.common.episode_metrics import EpisodeMetricsAggregator

    aggregator = EpisodeMetricsAggregator(path=tmp_path / "episodes.jsonl", rolling_window=10)

    payload = aggregator.process_event(
        {
            "return": 12.0,
            "raw_return": 12.0,
            "length": 4,
            "raw_frames": 4,
            "submitted_observation_count": 3,
            "dropped_observation_count": 1,
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
            seed=0,
            env_id="ALE/DemonAttack-v5",
            env_fps=60,
            obs_fps=15,
            simulator="cpu",
            screen_size=84,
            noop_max=30,
            reset_mode="fire",
            frame_stack=1,
            terminal_on_life_loss=True,
            clip_reward=True,
            repeat_action_probability=0.0,
            full_action_space=False,
            export_env_raw_rgb_frames=False,
            latency_type="zero",
            fixed_latency_ms=None,
            mean_latency_ms=None,
            std_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            latency_seed=None,
            simulated_worker_capacity=None,
            add_latency_info=False,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
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

    cfg = Namespace(
        env=integration.SF_ENV_NAME,
        env_gpu_actions=False,
        env_gpu_observations=False,
        seed=0,
        env_id="ALE/DemonAttack-v5",
        simulator="cpu",
        env_fps=60,
        obs_fps=15,
        screen_size=84,
        noop_max=30,
        reset_mode="fire",
        frame_stack=4,
        terminal_on_life_loss=True,
        clip_reward=True,
        repeat_action_probability=0.0,
        full_action_space=False,
        export_env_raw_rgb_frames=False,
        latency_type="zero",
        fixed_latency_ms=None,
        mean_latency_ms=None,
        std_latency_ms=None,
        min_latency_ms=None,
        max_latency_ms=None,
        latency_seed=None,
        simulated_worker_capacity=None,
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

    cfg = Namespace(
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
        simulated_worker_capacity=None,
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

    from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.common.raw_rgb_export import RawRgbFrameStackInfoWrapper

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

    from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
    from training.common.raw_rgb_export import RawRgbFrameStackInfoWrapper

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
