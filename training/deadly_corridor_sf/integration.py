from __future__ import annotations

import argparse
import copy
import functools
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.common.utils import (
    env_instance_seed,
    maybe_set_cli_override,
    seed_everything,
    str2bool,
)
from training.common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
)
from training.common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.latency_bench_periodic_eval import run_sample_factory_training
from training.common.latency_wrapper import ActionLatencyWrapper, EpisodeMetricsWrapper, RewardTraceWrapper
from training.common.raw_rgb_export import RawRgbFrameStackInfoWrapper
from training.common.sample_factory_config import SampleFactoryConfigParser
from training.common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed
from training.common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "doom_deadly_corridor"
_ENVS_REGISTERED = False
_MODELS_REGISTERED = False
_SAMPLE_FACTORY_PROBE_ENV_CONFIG = SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def _force_doom_raw_frame_step_cfg(cfg) -> None:
    maybe_set_cli_override(cfg, "env_frameskip", 1)
    maybe_set_cli_override(cfg, "eval_env_frameskip", 1)


def _doom_raw_reward_from_scaled(env_reward: float, info: dict[str, Any], reward_scale: float) -> float:
    if "raw_reward" in info:
        return float(info["raw_reward"])
    if reward_scale == 0.0:
        return float(env_reward)
    return float(env_reward) / float(reward_scale)


def _sample_factory_env_config(env_config):
    # Sample Factory probes env_info before worker/vector ids exist.
    if env_config is None:
        return _SAMPLE_FACTORY_PROBE_ENV_CONFIG
    return env_config


class DoomFrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        self._dict_observation = isinstance(env.observation_space, gym.spaces.Dict)
        if self._dict_observation:
            obs_space = env.observation_space.spaces.get("obs")
        else:
            obs_space = env.observation_space
        base_shape = tuple(int(dim) for dim in obs_space.shape)

        self.num_frames = int(num_frames)
        self._base_channels = int(base_shape[0])
        stacked_shape = (base_shape[0] * self.num_frames, base_shape[1], base_shape[2])
        self._stacked_obs = np.empty(stacked_shape, dtype=obs_space.dtype)
        stacked_obs_space = gym.spaces.Box(low=0, high=255, shape=stacked_shape, dtype=obs_space.dtype)
        if self._dict_observation:
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["obs"] = stacked_obs_space
            self.observation_space = gym.spaces.Dict(new_spaces)
        else:
            self.observation_space = stacked_obs_space

    def _stack_obs(self, observation):
        stacked_frames = self._stacked_obs.copy()
        if self._dict_observation:
            stacked = dict(observation)
            stacked["obs"] = stacked_frames
            return stacked
        return stacked_frames

    def _extract_frame(self, observation):
        if self._dict_observation:
            return observation["obs"]
        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = self._extract_frame(observation)
        for frame_idx in range(self.num_frames):
            start = frame_idx * self._base_channels
            self._stacked_obs[start : start + self._base_channels] = frame
        return self._stack_obs(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        frame = self._extract_frame(observation)
        self._stacked_obs[:-self._base_channels] = self._stacked_obs[self._base_channels :]
        self._stacked_obs[-self._base_channels :] = frame
        return self._stack_obs(observation), reward, terminated, truncated, info


def _vizdoom_raw_rgb_frame(env: gym.Env) -> np.ndarray:
    return np.transpose(env.unwrapped.game.get_state().screen_buffer, (1, 2, 0))


def make_latency_doom_env_from_spec(
    spec,
    env_name: str,
    cfg,
    env_config,
    render_mode: Optional[str] = None,
    **kwargs,
):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec as sf_make_doom_env_from_spec

    env_config = _sample_factory_env_config(env_config)
    spec_to_use = copy.copy(spec)
    if bool(getattr(cfg, "disable_reward_scaling", False)):
        spec_to_use.reward_scaling = 1.0
    _force_doom_raw_frame_step_cfg(cfg)
    clock = EnvClock(
        env_fps=cfg.env_fps,
        obs_fps=cfg.obs_fps,
    )
    env = sf_make_doom_env_from_spec(
        spec_to_use,
        env_name,
        cfg,
        env_config,
        render_mode=render_mode,
        **kwargs,
    )
    ensure_vizdoom_init_guard_installed(env, env_name=env_name, cfg=cfg, env_config=env_config)
    env = RewardTraceWrapper(
        env,
        raw_reward_fn=lambda env_reward, info, scale=float(spec_to_use.reward_scaling): _doom_raw_reward_from_scaled(
            env_reward,
            info,
            scale,
        ),
    )
    frame_stack = getattr(cfg, "frame_stack", 1)
    if bool(getattr(cfg, "export_env_raw_rgb_frames", False)):
        env = RawRgbFrameStackInfoWrapper(env, num_frames=frame_stack, raw_frame_fn=_vizdoom_raw_rgb_frame)
    if frame_stack > 1:
        env = DoomFrameStackWrapper(env, num_frames=frame_stack)
    max_episode_steps = getattr(cfg, "max_episode_steps", 0) or 0
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

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
        sample_distribution_per_episode=latency_params.sample_distribution_per_episode,
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


def register_deadly_corridor_envs() -> None:
    global _ENVS_REGISTERED
    if _ENVS_REGISTERED:
        return
    from sample_factory.envs.env_utils import register_env
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS

    for env_spec in DOOM_ENVS:
        register_env(env_spec.name, functools.partial(make_latency_doom_env_from_spec, env_spec))
    _ENVS_REGISTERED = True


def register_deadly_corridor_models() -> None:
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return
    from sample_factory.algo.utils.context import global_model_factory
    from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder

    global_model_factory().register_encoder_factory(make_vizdoom_encoder)
    _MODELS_REGISTERED = True


def register_deadly_corridor_components() -> None:
    register_deadly_corridor_envs()
    register_deadly_corridor_models()


def make_reference_env(cfg):
    from sample_factory.utils.attr_dict import AttrDict
    from sample_factory.algo.utils.make_env import make_env_func_batched

    return make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )


def add_latency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--env-fps", dest="env_fps", type=float, default=35)
    parser.add_argument("--obs-fps", dest="obs_fps", type=float, default=8.75)
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
        default="0,1,2,3,4,5,10,15,40",
    )
    parser.add_argument("--eval-max-steps", type=int, default=30000)
    parser.add_argument("--eval-raw-reward", action="store_true")
    parser.add_argument("--eval-deterministic", type=str2bool, default=True)
    parser.add_argument("--disable-reward-scaling", type=str2bool, default=False)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--simulator", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--episode_metrics_path", type=str, default="")
    parser.add_argument("--vizdoom_init_max_parallel", type=int, default=10)
    parser.add_argument("--vizdoom_init_lock_timeout_sec", type=float, default=5.0)
    parser.add_argument("--vizdoom_init_lock_max_wait_sec", type=float, default=60.0)
    parser.add_argument("--vizdoom_init_retry_jitter_sec", type=float, default=0.25)


def configure_sample_factory_parser(parser: argparse.ArgumentParser) -> None:
    from sf_examples.vizdoom.doom.doom_params import (
        add_doom_env_args,
        add_doom_env_eval_args,
        doom_override_defaults,
    )

    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    add_latency_args(parser)
    doom_override_defaults(parser)
    parser.set_defaults(env_frameskip=1, eval_env_frameskip=1)


SAMPLE_FACTORY_CONFIG_PARSER = SampleFactoryConfigParser(
    configure_parser=configure_sample_factory_parser,
    finalize_cfg=_force_doom_raw_frame_step_cfg,
)


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    evaluation_cfg = config.get("evaluation", {})
    scheduler_cfg = config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        raise ValueError("scheduler must be a mapping for DeadlyCorridor Sample Factory")
    args = build_common_sf_args(config, sf_env_name=env_cfg.get("env_id", SF_ENV_NAME))
    append_cli_arg(args, "--env-fps", env_cfg["env_fps"])
    append_cli_arg(args, "--obs-fps", env_cfg["obs_fps"])
    append_cli_arg(args, "--res_w", env_cfg.get("res_w", 128))
    append_cli_arg(args, "--res_h", env_cfg.get("res_h", 72))
    append_cli_arg(args, "--wide_aspect_ratio", env_cfg.get("wide_aspect_ratio", False))
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    append_cli_arg(args, "--simulator", env_cfg["simulator"])
    append_cli_arg(args, "--vizdoom_init_max_parallel", env_cfg.get("vizdoom_init_max_parallel"))
    append_cli_arg(args, "--vizdoom_init_lock_timeout_sec", env_cfg.get("vizdoom_init_lock_timeout_sec"))
    append_cli_arg(args, "--vizdoom_init_lock_max_wait_sec", env_cfg.get("vizdoom_init_lock_max_wait_sec"))
    append_cli_arg(args, "--vizdoom_init_retry_jitter_sec", env_cfg.get("vizdoom_init_retry_jitter_sec"))
    append_cli_arg(args, "--disable-reward-scaling", evaluation_cfg.get("eval_raw_reward", False))
    append_cli_arg(args, "--hold-policy", scheduler_cfg["hold_policy"])
    append_cli_arg(args, "--ordering-policy", scheduler_cfg["ordering_policy"])
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    register_deadly_corridor_components()
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    apply_runtime_logging_config(cfg, logging_cfg)
    seed_everything(int(cfg.seed or 0))
    patch_sample_factory_wandb()

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
