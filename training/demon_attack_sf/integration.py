from __future__ import annotations

import argparse
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from latency_bench.envs.demon_attack import make_demon_attack_benchmark_env
from training.sample_factory_common.common import env_instance_seed, str2bool
from training.sample_factory_common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
)
from training.sample_factory_common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.sample_factory_common.latency_wrapper import (
    ActionLatencyWrapper,
    EpisodeMetricsWrapper,
    RewardTraceWrapper,
)
from training.sample_factory_common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.sample_factory_common.latency_bench_periodic_eval import run_sample_factory_training
from training.sample_factory_common.sf_raw_reward_stats_compat import (
    ensure_sample_factory_raw_reward_console_stats_installed,
)
from training.sample_factory_common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "latency_demon_attack"
_ENVS_REGISTERED = False


def _force_sample_factory_raw_frame_step_cfg(cfg) -> None:
    setattr(cfg, "env_frameskip", 1)
    setattr(cfg, "eval_env_frameskip", 1)


def _clock_from_cfg(cfg) -> EnvClock:
    env_fps = getattr(cfg, "env_fps", None)
    obs_fps = getattr(cfg, "obs_fps", None)
    if env_fps is None or obs_fps is None:
        raise ValueError("DemonAttack Sample Factory requires env_fps and obs_fps")
    return EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps))


class AtariObsFormatWrapper(gym.ObservationWrapper):
    """Ensure Atari observations are CHW uint8 images for Sample Factory."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = self.observation_space
        if not isinstance(space, Box):
            raise TypeError("DemonAttack Sample Factory integration expects Box observations")
        shape = tuple(int(dim) for dim in space.shape)
        if len(shape) == 2:
            shape = (1,) + shape
        elif len(shape) == 3:
            shape = (shape[2], shape[0], shape[1])
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        array = np.asarray(observation, dtype=np.uint8)
        if array.ndim == 2:
            array = array[None, ...]
        elif array.ndim == 3:
            array = np.transpose(array, (2, 0, 1))
        return array


class AtariFrameStackWrapper(gym.Wrapper):
    observation_space: Box

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if not isinstance(env.observation_space, Box):
            raise TypeError("AtariFrameStackWrapper expects a Box observation space")
        base_shape = tuple(int(dim) for dim in env.observation_space.shape)
        if len(base_shape) != 3:
            raise ValueError(f"Expected channel-first image observation, got shape {base_shape}")
        self.num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        stacked_shape = (base_shape[0] * self.num_frames, base_shape[1], base_shape[2])
        self.observation_space = Box(low=0, high=255, shape=stacked_shape, dtype=np.uint8)

    def _stack_frames(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(np.array(obs, copy=True))
        return self._stack_frames(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.array(obs, copy=True))
        return self._stack_frames(), reward, terminated, truncated, info


def make_demon_attack_sf_env(
    _full_env_name: str,
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del render_mode
    ensure_sample_factory_raw_reward_console_stats_installed(env_name=SF_ENV_NAME)
    env_config = env_config or {}
    _force_sample_factory_raw_frame_step_cfg(cfg)
    clock = _clock_from_cfg(cfg)
    env = make_demon_attack_benchmark_env(
        env_id=getattr(cfg, "env_id", "ALE/DemonAttack-v5"),
        screen_size=int(getattr(cfg, "screen_size", 84)),
        noop_max=int(getattr(cfg, "noop_max", 30)),
        terminal_on_life_loss=bool(getattr(cfg, "terminal_on_life_loss", False)),
        clip_reward=False,
        repeat_action_probability=float(getattr(cfg, "repeat_action_probability", 0.0)),
        full_action_space=bool(getattr(cfg, "full_action_space", False)),
    )
    env = AtariObsFormatWrapper(env)
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    if frame_stack > 1:
        env = AtariFrameStackWrapper(env, num_frames=frame_stack)
    max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if bool(getattr(cfg, "clip_reward", True)):
        env = RewardTraceWrapper(env, learner_reward_fn=lambda raw_reward, info: float(np.sign(raw_reward)))
    else:
        env = RewardTraceWrapper(env)
    wrapper_seed = env_instance_seed(
        base_seed=int(getattr(cfg, "seed", 0) or 0),
        worker_index=int(getattr(env_config, "worker_index", 0) or 0),
        vector_index=int(getattr(env_config, "vector_index", 0) or 0),
        env_id=int(getattr(env_config, "env_id", 0) or 0),
    )
    latency_seed = sample_factory_latency_seed_from_flat_config(
        cfg,
        fallback_seed=wrapper_seed,
        worker_index=int(getattr(env_config, "worker_index", 0) or 0),
        vector_index=int(getattr(env_config, "vector_index", 0) or 0),
        env_id=int(getattr(env_config, "env_id", 0) or 0),
    )
    latency_params = sample_factory_latency_params_from_flat_config(cfg, env_fps=clock.env_fps)
    env = ActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=latency_params.fixed_latency_raw_frames,
        env_fps=clock.env_fps,
        obs_fps=clock.obs_fps,
        hold_policy=getattr(cfg, "hold_policy", ActionHoldPolicy.HOLD.value),
        ordering_policy=getattr(cfg, "ordering_policy", OrderingPolicy.LATEST_READY.value),
        latency_distribution_spec=latency_params.latency_distribution_spec,
        seed=latency_seed,
        env_reset_seed_base=wrapper_seed + 104729,
        add_latency_info=bool(getattr(cfg, "add_latency_info", False)),
        max_pending_actions=getattr(cfg, "max_pending_actions", None),
    )
    env = EpisodeMetricsWrapper(
        env,
        frames_per_decision=clock.obs_stride_raw_frames,
        metrics_path=str(getattr(cfg, "episode_metrics_path", "") or ""),
        worker_index=int(getattr(env_config, "worker_index", 0) or 0),
        vector_index=int(getattr(env_config, "vector_index", 0) or 0),
        env_id=int(getattr(env_config, "env_id", 0) or 0),
    )
    return env


def make_reference_env(cfg):
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.utils.attr_dict import AttrDict

    _force_sample_factory_raw_frame_step_cfg(cfg)
    register_demon_attack_envs()
    return make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )


def register_demon_attack_envs() -> None:
    global _ENVS_REGISTERED
    if _ENVS_REGISTERED:
        return
    from sample_factory.envs.env_utils import register_env

    register_env(SF_ENV_NAME, make_demon_attack_sf_env)
    _ENVS_REGISTERED = True


def add_demon_attack_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env_id", type=str, default="ALE/DemonAttack-v5")
    parser.add_argument("--env-fps", dest="env_fps", type=float, default=60)
    parser.add_argument("--obs-fps", dest="obs_fps", type=float, default=15)
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--terminal_on_life_loss", type=str2bool, default=False)
    parser.add_argument("--clip_reward", type=str2bool, default=True)
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--full_action_space", type=str2bool, default=False)
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--latency-type", dest="latency_type", type=str, default="zero")
    parser.add_argument("--fixed-latency-ms", dest="fixed_latency_ms", type=float)
    parser.add_argument("--mean-latency-ms", dest="mean_latency_ms", type=float)
    parser.add_argument("--std-latency-ms", dest="std_latency_ms", type=float)
    parser.add_argument("--min-latency-ms", dest="min_latency_ms", type=float)
    parser.add_argument("--max-latency-ms", dest="max_latency_ms", type=float)
    parser.add_argument("--latency-seed", dest="latency_seed", type=int)
    parser.add_argument("--add-latency-info", type=str2bool, default=False)
    parser.add_argument("--max-pending-actions", dest="max_pending_actions", type=int)
    parser.add_argument(
        "--hold-policy",
        dest="hold_policy",
        type=str,
        default=ActionHoldPolicy.HOLD.value,
        choices=[item.value for item in ActionHoldPolicy],
    )
    parser.add_argument(
        "--ordering-policy",
        dest="ordering_policy",
        type=str,
        default=OrderingPolicy.LATEST_READY.value,
        choices=[item.value for item in OrderingPolicy],
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-parallel-envs", type=int, default=1)
    parser.add_argument(
        "--eval-latency-raw-frame-values",
        dest="eval_latency_raw_frame_values",
        type=str,
        default="0,1,2,3,4,5",
    )
    parser.add_argument("--eval-max-steps", type=int, default=30000)
    parser.add_argument("--eval-deterministic", type=str2bool, default=True)
    parser.add_argument("--eval-raw-reward", action="store_true")
    parser.add_argument("--episode_metrics_path", type=str, default="")


def demon_attack_override_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        encoder_conv_architecture="convnet_atari",
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=2.5e-4,
        lr_schedule="linear_decay",
        adam_eps=1e-5,
        use_rnn=False,
        normalize_input=True,
        normalize_returns=True,
    )


def parse_cfg(argv=None):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

    parser, _ = parse_sf_args(argv=argv, evaluation=False)
    add_demon_attack_env_args(parser)
    demon_attack_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    _force_sample_factory_raw_frame_step_cfg(cfg)
    return cfg


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    scheduler_cfg = config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        raise ValueError("scheduler must be a mapping for DemonAttack Sample Factory")
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    append_cli_arg(args, "--env_id", env_cfg.get("env_id", "ALE/DemonAttack-v5"))
    append_cli_arg(args, "--env-fps", env_cfg["env_fps"])
    append_cli_arg(args, "--obs-fps", env_cfg["obs_fps"])
    append_cli_arg(args, "--screen_size", env_cfg.get("screen_size", 84))
    append_cli_arg(args, "--noop_max", env_cfg.get("noop_max", 30))
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--terminal_on_life_loss", env_cfg.get("terminal_on_life_loss", False))
    append_cli_arg(args, "--clip_reward", env_cfg.get("clip_reward", True))
    append_cli_arg(args, "--repeat_action_probability", env_cfg.get("repeat_action_probability", 0.0))
    append_cli_arg(args, "--full_action_space", env_cfg.get("full_action_space", False))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    append_cli_arg(args, "--hold-policy", scheduler_cfg["hold_policy"])
    append_cli_arg(args, "--ordering-policy", scheduler_cfg["ordering_policy"])
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    from training.sample_factory_common.common import seed_everything

    register_demon_attack_envs()
    cfg = parse_cfg(build_cli_args_from_config(config))
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    apply_runtime_logging_config(cfg, logging_cfg)
    seed_everything(int(cfg.seed or 0))
    patch_sample_factory_wandb()
    ensure_sample_factory_raw_reward_console_stats_installed(env_name=SF_ENV_NAME)

    metrics_stop_event = None
    metrics_thread = None
    if cfg.mode == "train":
        metrics_stop_event, metrics_thread = start_episode_metrics_aggregator(cfg)

    status = ExperimentStatus.SUCCESS
    try:
        if cfg.mode == "train":
            status = run_sample_factory_training(cfg, config)
    finally:
        stop_episode_metrics_aggregator(metrics_stop_event, metrics_thread)
    return int(status) if isinstance(status, int) else 0
