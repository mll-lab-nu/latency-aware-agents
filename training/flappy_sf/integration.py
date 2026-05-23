from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict
from PIL import Image

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from latency_bench.envs.flappy import resize_frame_pil
from training.common.utils import (
    env_instance_seed,
    str2bool,
    seed_everything,
)
from training.common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
    sample_factory_training_summary,
)
from training.common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.latency_bench_periodic_eval import run_sample_factory_training
from training.common.latency_wrapper import (
    ActionLatencyWrapper,
    EpisodeMetricsWrapper,
    RewardTraceWrapper,
)
from training.common.sf_episode_extra_stats_compat import (
    ensure_sample_factory_episode_extra_stats_installed,
)
from training.common.wandb_compat import patch_sample_factory_wandb

from sample_factory.algo.utils.misc import ExperimentStatus


SF_ENV_NAME = "latency_flappy"
ENV_RAW_RGB_FRAME_STACK_INFO_KEY = "env_raw_rgb_frame_stack"
_ENVS_REGISTERED = False
_TIMELIMIT_DIAGNOSTICS_PRINTED: set[tuple[int, int, int]] = set()


def _force_sample_factory_raw_frame_step_cfg(cfg: Any) -> None:
    setattr(cfg, "env_frameskip", 1)
    setattr(cfg, "eval_env_frameskip", 1)


def parse_obs_resize(value: Any) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("obs_resize must be [height, width] or a 'height,width' string")
    if len(parts) != 2:
        raise ValueError("obs_resize must contain exactly two positive integers: height,width")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError("obs_resize height and width must be positive")
    return height, width


def _format_shape(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(dim)) for dim in shape) + ")"


def log_flappy_sf_resolved_config(cfg: Any, final_obs_shape: tuple[int, int, int]) -> None:
    if not bool(getattr(cfg, "debug_timelimit_diagnostics", False)):
        return
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    obs_resize = parse_obs_resize(getattr(cfg, "obs_resize", None))
    single_frame_shape = (3, int(final_obs_shape[1]), int(final_obs_shape[2]))
    stacked_shape = (3 * frame_stack, int(final_obs_shape[1]), int(final_obs_shape[2]))
    print(
        "[flappy_sf] observation "
        f"single_frame_shape={_format_shape(single_frame_shape)} "
        f"stacked_rgb_shape={_format_shape(stacked_shape)} "
        f"final_model_input_shape={_format_shape(final_obs_shape)} "
        "dtype=uint8 "
        f"obs_resize={obs_resize}"
    )


def _wrapper_chain(env: gym.Env) -> list[str]:
    chain: list[str] = []
    current: Any = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = getattr(current, "env", None)
    return chain


def _time_limit_max_episode_steps(env: gym.Env) -> int | None:
    current: Any = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, "_max_episode_steps", None)
        if value is not None:
            return int(value)
        current = getattr(current, "env", None)
    return None


def _spec_max_episode_steps(env: gym.Env) -> int | None:
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None)
    return None if value is None else int(value)


def _maybe_log_flappy_timelimit_diagnostics(
    cfg: Any,
    *,
    env: gym.Env,
    representative_env: gym.Env,
    final_obs_shape: tuple[int, int, int],
    env_config: Any,
) -> None:
    if not bool(getattr(cfg, "debug_timelimit_diagnostics", False)):
        return
    worker_index = int(getattr(env_config, "worker_index", 0) or 0)
    vector_index = int(getattr(env_config, "vector_index", 0) or 0)
    env_id = int(getattr(env_config, "env_id", 0) or 0)
    key = (worker_index, vector_index, env_id)
    if key in _TIMELIMIT_DIAGNOSTICS_PRINTED:
        return
    _TIMELIMIT_DIAGNOSTICS_PRINTED.add(key)
    print(
        "[flappy_sf][timelimit] "
        f"worker_index={worker_index} "
        f"vector_index={vector_index} "
        f"env_id={env_id} "
        f"cfg.max_episode_steps={int(getattr(cfg, 'max_episode_steps', 0) or 0)} "
        f"env.spec.max_episode_steps={_spec_max_episode_steps(representative_env)} "
        f"TimeLimit._max_episode_steps={_time_limit_max_episode_steps(representative_env)} "
        f"wrapper_chain={'->'.join(_wrapper_chain(representative_env))} "
        f"resolved_max_episode_steps={_time_limit_max_episode_steps(representative_env)} "
        f"use_gpu_render={bool(getattr(cfg, 'use_gpu_render', False))} "
        f"gpu_render_batch_size={int(getattr(cfg, 'gpu_render_batch_size', 1) or 1)} "
        f"num_agents={int(getattr(env, 'num_agents', 1) or 1)} "
        f"obs_shape={_format_shape(final_obs_shape)}"
    )


def _flappy_final_obs_shape(space: gym.Space) -> tuple[int, int, int]:
    if isinstance(space, GymDict):
        obs_space = space.spaces.get("obs")
        if not isinstance(obs_space, Box):
            raise TypeError("Flappy Sample Factory Dict observation space must contain Box space at key 'obs'")
        shape = obs_space.shape
    elif isinstance(space, Box):
        shape = space.shape
    else:
        raise TypeError(f"Unsupported Flappy observation space: {space!r}")
    if shape is None or len(shape) != 3:
        raise ValueError(f"Expected Flappy CHW observation shape, got {shape!r}")
    return tuple(int(dim) for dim in shape)


def log_flappy_sf_training_summary(config: dict) -> None:
    summary = sample_factory_training_summary(config)
    print(
        "[flappy_sf] training "
        f"effective_envs={summary['effective_envs']} "
        f"batch_size={summary['batch_size']} "
        f"num_epochs={summary['num_epochs']} "
        f"num_batches_per_epoch={summary['num_batches_per_epoch']} "
        f"estimated_optimizer_steps_per_1m_env_steps={summary['optimizer_steps_per_1m_env_steps']:.2f} "
        f"async_rl={summary['async_rl']}"
    )


class FlappyImageObsWrapper(gym.Wrapper):
    """Expose rendered RGB frames as the environment observation."""

    observation_space: Box

    def __init__(self, env: gym.Env, obs_resize: tuple[int, int] | None = None):
        super().__init__(env)
        self.obs_resize = obs_resize
        obs, _ = self.env.reset(seed=0)
        frame = self._observation_frame(self._render_raw_rgb_frame())
        self._last_numeric_obs = obs
        self.observation_space = Box(
            low=0,
            high=255,
            shape=tuple(int(dim) for dim in frame.shape),
            dtype=np.uint8,
        )

    def _render_raw_rgb_frame(self) -> np.ndarray:
        frame = self.env.render()
        return np.asarray(frame, dtype=np.uint8)

    def _observation_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(raw_frame, dtype=np.uint8)
        if self.obs_resize is not None:
            height, width = self.obs_resize
            if height == width:
                frame = resize_frame_pil(frame, image_size=height, grayscale=False)
            else:
                image = Image.fromarray(frame).convert("RGB")
                image = image.resize((width, height), Image.BILINEAR)
                frame = np.asarray(image, dtype=np.uint8)
            if frame.shape[:2] != (height, width):
                frame = np.asarray(frame, dtype=np.uint8)
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"Flappy obs_resize expected HWC frame {(height, width, 3)}, got {frame.shape!r}"
                    )
        return np.transpose(frame, (2, 0, 1))

    def reset(self, **kwargs):
        numeric_obs, info = self.env.reset(**kwargs)
        self._last_numeric_obs = numeric_obs
        raw_frame = self._render_raw_rgb_frame()
        info = dict(info or {})
        info["numeric_obs"] = numeric_obs
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = np.stack([raw_frame], axis=0)
        return self._observation_frame(raw_frame), info

    def step(self, action):
        numeric_obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_numeric_obs = numeric_obs
        raw_frame = self._render_raw_rgb_frame()
        info = dict(info or {})
        info["numeric_obs"] = numeric_obs
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = np.stack([raw_frame], axis=0)
        return self._observation_frame(raw_frame), reward, terminated, truncated, info


class FlappyFrameStackWrapper(gym.Wrapper):
    observation_space: Box

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        base_shape = tuple(int(dim) for dim in env.observation_space.shape)
        self.num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        self._raw_rgb_frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        stacked_shape = (base_shape[0] * self.num_frames, base_shape[1], base_shape[2])
        self.observation_space = Box(low=0, high=255, shape=stacked_shape, dtype=np.uint8)

    def _stack_frames(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames), axis=0)

    def _stack_raw_rgb_frames(self) -> np.ndarray:
        return np.stack(tuple(self._raw_rgb_frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        self._raw_rgb_frames.clear()
        raw_frame = np.asarray(info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1], dtype=np.uint8)
        for _ in range(self.num_frames):
            self._frames.append(np.array(obs, copy=True))
            self._raw_rgb_frames.append(raw_frame.copy())
        info = dict(info or {})
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._stack_raw_rgb_frames()
        return self._stack_frames(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_frame = np.asarray(info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1], dtype=np.uint8)
        self._frames.append(np.array(obs, copy=True))
        self._raw_rgb_frames.append(raw_frame.copy())
        info = dict(info or {})
        info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._stack_raw_rgb_frames()
        return self._stack_frames(), reward, terminated, truncated, info


def make_flappy_sf_env(
    _full_env_name: str,
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del render_mode
    ensure_sample_factory_episode_extra_stats_installed(env_name=SF_ENV_NAME)
    env_config = env_config or {}
    _force_sample_factory_raw_frame_step_cfg(cfg)
    use_gpu_render = bool(getattr(cfg, "use_gpu_render", False))
    obs_resize = parse_obs_resize(getattr(cfg, "obs_resize", None))
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    clock = EnvClock(
        env_fps=cfg.env_fps,
        obs_fps=cfg.obs_fps,
    )

    if use_gpu_render:
        from training.flappy_sf.batched_gpu_env import FlappyBatchedGpuRenderedEnv

        env = FlappyBatchedGpuRenderedEnv(
            cfg=cfg,
            env_config=env_config,
            obs_resize=obs_resize,
            frame_stack=frame_stack,
            gpu_render_device=getattr(cfg, "gpu_render_device", "auto"),
            gpu_render_batch_size=int(getattr(cfg, "gpu_render_batch_size", 1) or 1),
            gpu_render_profile=bool(getattr(cfg, "gpu_render_profile", False)),
            gpu_render_profile_interval=int(getattr(cfg, "gpu_render_profile_interval", 1000) or 1000),
        )
        final_obs_shape = _flappy_final_obs_shape(env.observation_space)
        log_flappy_sf_resolved_config(cfg, final_obs_shape)
        representative_env = env.sub_envs[0] if getattr(env, "sub_envs", None) else env
        _maybe_log_flappy_timelimit_diagnostics(
            cfg,
            env=env,
            representative_env=representative_env,
            final_obs_shape=final_obs_shape,
            env_config=env_config,
        )
        return env

    kwargs: dict[str, Any] = {
        "use_lidar": bool(getattr(cfg, "use_lidar", False)),
        "normalize_obs": bool(getattr(cfg, "normalize_obs", True)),
        "audio_on": bool(getattr(cfg, "audio_on", False)),
        "pipe_gap": int(getattr(cfg, "pipe_gap", 100)),
        "bird_color": getattr(cfg, "bird_color", "yellow"),
        "pipe_color": getattr(cfg, "pipe_color", "green"),
        "background": getattr(cfg, "background", "day"),
        "debug": bool(getattr(cfg, "debug", False)),
        "render_mode": None if use_gpu_render else "rgb_array",
    }
    screen_size = getattr(cfg, "screen_size", None)
    if screen_size not in (None, ""):
        raise ValueError(
            "screen_size is not supported for Flappy Sample Factory; "
            "use env.obs_resize to resize observations without changing env geometry"
        )
    score_limit = getattr(cfg, "score_limit", None)
    if score_limit not in (None, "", 0, -1):
        kwargs["score_limit"] = int(score_limit)
    env = gym.make(getattr(cfg, "gym_id", "FlappyBird-v0"), **kwargs)
    if not use_gpu_render:
        env = FlappyImageObsWrapper(env, obs_resize=obs_resize)
        if frame_stack > 1:
            env = FlappyFrameStackWrapper(env, num_frames=frame_stack)
    max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
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
        hold_policy=getattr(cfg, "hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value),
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
        issued_action_labels={0: "noop", 1: "flap"},
    )
    final_obs_shape = _flappy_final_obs_shape(env.observation_space)
    log_flappy_sf_resolved_config(cfg, final_obs_shape)
    _maybe_log_flappy_timelimit_diagnostics(
        cfg,
        env=env,
        representative_env=env,
        final_obs_shape=final_obs_shape,
        env_config=env_config,
    )
    return env


def make_reference_env(cfg):
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.utils.attr_dict import AttrDict

    _force_sample_factory_raw_frame_step_cfg(cfg)
    register_flappy_envs()
    return make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )


def register_flappy_envs() -> None:
    global _ENVS_REGISTERED
    if _ENVS_REGISTERED:
        return
    from sample_factory.envs.env_utils import register_env

    register_env(SF_ENV_NAME, make_flappy_sf_env)
    _ENVS_REGISTERED = True


def add_flappy_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gym_id", type=str, default="FlappyBird-v0")
    parser.add_argument("--env-fps", dest="env_fps", type=float, default=30)
    parser.add_argument("--obs-fps", dest="obs_fps", type=float, default=30)
    parser.add_argument("--use_lidar", type=str2bool, default=False)
    parser.add_argument("--normalize_obs", type=str2bool, default=True)
    parser.add_argument("--audio_on", type=str2bool, default=False)
    parser.add_argument("--screen_size", type=str, default="")
    parser.add_argument("--obs_resize", type=str, default="")
    parser.add_argument("--use-gpu-render", dest="use_gpu_render", type=str2bool, default=False)
    parser.add_argument("--gpu-render-device", dest="gpu_render_device", type=str, default="auto")
    parser.add_argument("--gpu-render-batch-size", dest="gpu_render_batch_size", type=int, default=1)
    parser.add_argument("--gpu-render-profile", dest="gpu_render_profile", type=str2bool, default=False)
    parser.add_argument("--gpu-render-profile-interval", dest="gpu_render_profile_interval", type=int, default=1000)
    parser.add_argument("--pipe_gap", type=int, default=100)
    parser.add_argument("--bird_color", type=str, default="yellow")
    parser.add_argument("--pipe_color", type=str, default="green")
    parser.add_argument("--background", type=str, default="day")
    parser.add_argument("--score_limit", type=int, default=-1)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--debug-timelimit-diagnostics", dest="debug_timelimit_diagnostics", type=str2bool, default=False)
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
        default=ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value,
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


def flappy_override_defaults(parser: argparse.ArgumentParser) -> None:
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
    add_flappy_env_args(parser)
    flappy_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    _force_sample_factory_raw_frame_step_cfg(cfg)
    return cfg


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    scheduler_cfg = config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        raise ValueError("scheduler must be a mapping for Flappy Sample Factory")
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    append_cli_arg(args, "--gym_id", env_cfg.get("gym_id", "FlappyBird-v0"))
    append_cli_arg(args, "--env-fps", env_cfg["env_fps"])
    append_cli_arg(args, "--obs-fps", env_cfg["obs_fps"])
    append_cli_arg(args, "--use_lidar", env_cfg.get("use_lidar", False))
    append_cli_arg(args, "--normalize_obs", env_cfg.get("normalize_obs", True))
    append_cli_arg(args, "--audio_on", env_cfg.get("audio_on", False))
    if env_cfg.get("screen_size") is not None:
        raise ValueError(
            "env.screen_size is not supported for Flappy Sample Factory; "
            "use env.obs_resize to resize observations without changing env geometry"
        )
    append_cli_arg(args, "--obs_resize", env_cfg.get("obs_resize"))
    append_cli_arg(args, "--use-gpu-render", env_cfg.get("use_gpu_render", False))
    append_cli_arg(args, "--gpu-render-device", env_cfg.get("gpu_render_device", "auto"))
    append_cli_arg(args, "--gpu-render-profile", env_cfg.get("gpu_render_profile", False))
    append_cli_arg(args, "--gpu-render-profile-interval", env_cfg.get("gpu_render_profile_interval", 1000))
    append_cli_arg(args, "--pipe_gap", env_cfg.get("pipe_gap", 100))
    append_cli_arg(args, "--bird_color", env_cfg.get("bird_color", "yellow"))
    append_cli_arg(args, "--pipe_color", env_cfg.get("pipe_color", "green"))
    append_cli_arg(args, "--background", env_cfg.get("background", "day"))
    if env_cfg.get("score_limit") is not None:
        append_cli_arg(args, "--score_limit", env_cfg.get("score_limit"))
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--debug", env_cfg.get("debug", False))
    append_cli_arg(args, "--debug-timelimit-diagnostics", env_cfg.get("debug_timelimit_diagnostics", False))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    append_cli_arg(args, "--hold-policy", scheduler_cfg["hold_policy"])
    append_cli_arg(args, "--ordering-policy", scheduler_cfg["ordering_policy"])
    return args


def run_from_config(config: dict) -> int:
    register_flappy_envs()
    cfg = parse_cfg(build_cli_args_from_config(config))
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    apply_runtime_logging_config(cfg, logging_cfg)
    seed_everything(int(cfg.seed or 0))
    patch_sample_factory_wandb()
    ensure_sample_factory_episode_extra_stats_installed(env_name=SF_ENV_NAME)
    log_flappy_sf_training_summary(config)

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
