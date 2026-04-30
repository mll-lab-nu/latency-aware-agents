from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from training.sample_factory_common.common import env_instance_seed, parse_int_list, str2bool
from training.sample_factory_common.config_adapter import append_cli_arg, build_common_sf_args
from training.sample_factory_common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper, QueueLatencyWrapper, RewardTraceWrapper
from training.sample_factory_common.queue_eval import (
    evaluate_queue_latency,
    finish_queue_eval_wandb,
    init_queue_eval_wandb,
    log_queue_eval_result,
    write_queue_eval_result,
)
from training.sample_factory_common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "latency_flappy"
_ENVS_REGISTERED = False


def _import_flappy_gymnasium():
    try:
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        flappy_root = repo_root / "flappy-bird-gymnasium"
        if flappy_root.exists() and str(flappy_root) not in sys.path:
            sys.path.insert(0, str(flappy_root))
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium


class FlappyImageObsWrapper(gym.Wrapper):
    """Expose rendered RGB frames as the environment observation."""

    observation_space: Box

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs, _ = self.env.reset(seed=0)
        frame = self._render_frame()
        self._last_numeric_obs = obs
        self.observation_space = Box(
            low=0,
            high=255,
            shape=tuple(int(dim) for dim in frame.shape),
            dtype=np.uint8,
        )

    def _render_frame(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("FlappyBird-v0 render() returned None; use render_mode='rgb_array'")
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"Expected Flappy RGB frame with shape (H, W, 3), got {frame.shape!r}")
        return np.transpose(frame, (2, 0, 1))

    def reset(self, **kwargs):
        numeric_obs, info = self.env.reset(**kwargs)
        self._last_numeric_obs = numeric_obs
        info = dict(info or {})
        info["numeric_obs"] = numeric_obs
        return self._render_frame(), info

    def step(self, action):
        numeric_obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_numeric_obs = numeric_obs
        info = dict(info or {})
        info["numeric_obs"] = numeric_obs
        return self._render_frame(), reward, terminated, truncated, info


class FlappyFrameStackWrapper(gym.Wrapper):
    observation_space: Box

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if not isinstance(env.observation_space, Box):
            raise TypeError("FlappyFrameStackWrapper expects a Box observation space")
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


def make_flappy_sf_env(
    _full_env_name: str,
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del render_mode
    env_config = env_config or {}
    gymnasium = _import_flappy_gymnasium()
    kwargs: dict[str, Any] = {
        "use_lidar": bool(getattr(cfg, "use_lidar", False)),
        "normalize_obs": bool(getattr(cfg, "normalize_obs", True)),
        "audio_on": bool(getattr(cfg, "audio_on", False)),
        "pipe_gap": int(getattr(cfg, "pipe_gap", 100)),
        "bird_color": getattr(cfg, "bird_color", "yellow"),
        "pipe_color": getattr(cfg, "pipe_color", "green"),
        "background": getattr(cfg, "background", "day"),
        "debug": bool(getattr(cfg, "debug", False)),
        "render_mode": "rgb_array",
    }
    screen_size = getattr(cfg, "screen_size", None)
    if screen_size not in (None, ""):
        if isinstance(screen_size, str):
            parts = [int(item.strip()) for item in screen_size.split(",") if item.strip()]
            if parts:
                kwargs["screen_size"] = tuple(parts)
        else:
            kwargs["screen_size"] = tuple(int(item) for item in screen_size)
    score_limit = getattr(cfg, "score_limit", None)
    if score_limit not in (None, "", 0, -1):
        kwargs["score_limit"] = int(score_limit)
    env = gymnasium.make(getattr(cfg, "gym_id", "FlappyBird-v0"), **kwargs)
    env = FlappyImageObsWrapper(env)
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    if frame_stack > 1:
        env = FlappyFrameStackWrapper(env, num_frames=frame_stack)
    max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RewardTraceWrapper(env)
    latency_values = parse_int_list(getattr(cfg, "queue_latency_values", "0"))
    wrapper_seed = env_instance_seed(
        base_seed=int(getattr(cfg, "seed", 0) or 0),
        worker_index=int(getattr(env_config, "worker_index", 0) or 0),
        vector_index=int(getattr(env_config, "vector_index", 0) or 0),
        env_id=int(getattr(env_config, "env_id", 0) or 0),
    )
    env = QueueLatencyWrapper(
        env,
        queue_latency_values=latency_values or [0],
        latency_mode=getattr(cfg, "latency_mode", "episode_fixed"),
        latency_sampling_mode=getattr(cfg, "latency_sampling_mode", "uniform_all"),
        episodes_per_stage=int(getattr(cfg, "episodes_per_stage", 10)),
        randomize_within_stage=bool(getattr(cfg, "randomize_within_stage", False)),
        within_episode_change_interval=int(getattr(cfg, "within_episode_change_interval", 1)),
        within_episode_change_prob=float(getattr(cfg, "within_episode_change_prob", 1.0)),
        seed=wrapper_seed,
        env_reset_seed_base=wrapper_seed + 104729,
        latency_info_keys=("queue_latency", "frame_skip", "latency_frame_skip"),
        add_latency_info=bool(getattr(cfg, "add_latency_info", False)),
    )
    env = EpisodeMetricsWrapper(
        env,
        frames_per_decision=1,
        metrics_path=str(getattr(cfg, "episode_metrics_path", "") or ""),
        worker_index=int(getattr(env_config, "worker_index", 0) or 0),
        vector_index=int(getattr(env_config, "vector_index", 0) or 0),
        env_id=int(getattr(env_config, "env_id", 0) or 0),
    )
    return env


def make_reference_env(cfg):
    from sample_factory.utils.attr_dict import AttrDict

    return make_flappy_sf_env(
        SF_ENV_NAME,
        cfg=cfg,
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
    parser.add_argument("--use_lidar", type=str2bool, default=False)
    parser.add_argument("--normalize_obs", type=str2bool, default=True)
    parser.add_argument("--audio_on", type=str2bool, default=False)
    parser.add_argument("--screen_size", type=str, default="")
    parser.add_argument("--pipe_gap", type=int, default=100)
    parser.add_argument("--bird_color", type=str, default="yellow")
    parser.add_argument("--pipe_color", type=str, default="green")
    parser.add_argument("--background", type=str, default="day")
    parser.add_argument("--score_limit", type=int, default=-1)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "train_eval"])
    parser.add_argument("--queue-latency-values", type=str, default="0")
    parser.add_argument("--latency-mode", type=str, default="episode_fixed")
    parser.add_argument("--latency-sampling-mode", type=str, default="uniform_all")
    parser.add_argument("--episodes-per-stage", type=int, default=10)
    parser.add_argument("--randomize-within-stage", type=str2bool, default=False)
    parser.add_argument("--within-episode-change-interval", type=int, default=1)
    parser.add_argument("--within-episode-change-prob", type=float, default=1.0)
    parser.add_argument("--add-latency-info", type=str2bool, default=False)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-queue-latency-values", type=str, default="0,1,2,3,4,5")
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


def parse_cfg(argv=None, evaluation: bool = False):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_flappy_env_args(parser)
    flappy_override_defaults(parser)
    return parse_full_cfg(parser, argv)


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    append_cli_arg(args, "--gym_id", env_cfg.get("gym_id", "FlappyBird-v0"))
    append_cli_arg(args, "--use_lidar", env_cfg.get("use_lidar", False))
    append_cli_arg(args, "--normalize_obs", env_cfg.get("normalize_obs", True))
    append_cli_arg(args, "--audio_on", env_cfg.get("audio_on", False))
    append_cli_arg(args, "--screen_size", env_cfg.get("screen_size"))
    append_cli_arg(args, "--pipe_gap", env_cfg.get("pipe_gap", 100))
    append_cli_arg(args, "--bird_color", env_cfg.get("bird_color", "yellow"))
    append_cli_arg(args, "--pipe_color", env_cfg.get("pipe_color", "green"))
    append_cli_arg(args, "--background", env_cfg.get("background", "day"))
    if env_cfg.get("score_limit") is not None:
        append_cli_arg(args, "--score_limit", env_cfg.get("score_limit"))
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--debug", env_cfg.get("debug", False))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    from training.sample_factory_common.common import seed_everything

    register_flappy_envs()
    requested_mode = str(config.get("backend", {}).get("run_mode", "train"))
    cfg = parse_cfg(build_cli_args_from_config(config), evaluation=requested_mode in {"eval", "train_eval"})
    seed_everything(int(cfg.seed or 0))
    patch_sample_factory_wandb()
    from sample_factory.train import run_rl

    metrics_stop_event = None
    metrics_thread = None
    if cfg.mode in {"train", "train_eval"}:
        metrics_stop_event, metrics_thread = start_episode_metrics_aggregator(cfg)

    status = ExperimentStatus.SUCCESS
    try:
        if cfg.mode in {"train", "train_eval"}:
            status = run_rl(cfg)
        if cfg.mode in {"eval", "train_eval"}:
            latencies = parse_int_list(cfg.eval_queue_latency_values)
            wandb_run, owned_wandb_run = init_queue_eval_wandb(cfg)
            try:
                for latency in latencies:
                    result = evaluate_queue_latency(cfg, int(latency))
                    write_queue_eval_result(cfg, result)
                    log_queue_eval_result(wandb_run, result)
                    print(
                        f"[eval] latency={latency} mean_return={result['mean_return']:.3f} "
                        f"std_return={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
                    )
            finally:
                finish_queue_eval_wandb(wandb_run, owned_wandb_run)
    finally:
        stop_episode_metrics_aggregator(metrics_stop_event, metrics_thread)
    return int(status) if isinstance(status, int) else 0
