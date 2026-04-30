from __future__ import annotations

import argparse
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from latency_bench.envs.demon_attack import make_demon_attack_benchmark_env
from training.sample_factory_common.common import env_instance_seed, parse_int_list, str2bool
from training.sample_factory_common.config_adapter import append_cli_arg, build_common_sf_args
from training.sample_factory_common.latency_wrapper import QueueLatencyWrapper
from training.sample_factory_common.queue_eval import evaluate_queue_latency


SF_ENV_NAME = "latency_demon_attack"
_ENVS_REGISTERED = False


class AtariObsFormatWrapper(gym.ObservationWrapper):
    """Ensure Atari observations are HWC uint8 images for Sample Factory."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = self.observation_space
        if not isinstance(space, Box):
            raise TypeError("DemonAttack Sample Factory integration expects Box observations")
        shape = tuple(int(dim) for dim in space.shape)
        if len(shape) == 2:
            shape = shape + (1,)
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        array = np.asarray(observation, dtype=np.uint8)
        if array.ndim == 2:
            array = array[..., None]
        return array


def make_demon_attack_sf_env(
    _full_env_name: str,
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del render_mode
    env_config = env_config or {}
    env = make_demon_attack_benchmark_env(
        env_id=getattr(cfg, "env_id", "ALE/DemonAttack-v5"),
        screen_size=int(getattr(cfg, "screen_size", 84)),
        noop_max=int(getattr(cfg, "noop_max", 30)),
        action_repeat=int(getattr(cfg, "action_repeat", 1)),
        terminal_on_life_loss=bool(getattr(cfg, "terminal_on_life_loss", False)),
        clip_reward=bool(getattr(cfg, "clip_reward", True)),
        repeat_action_probability=float(getattr(cfg, "repeat_action_probability", 0.0)),
        full_action_space=bool(getattr(cfg, "full_action_space", False)),
    )
    env = AtariObsFormatWrapper(env)
    max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
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
    return env


def make_reference_env(cfg):
    from sample_factory.utils.attr_dict import AttrDict

    return make_demon_attack_sf_env(
        SF_ENV_NAME,
        cfg=cfg,
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
    parser.add_argument("--screen_size", type=int, default=84)
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--terminal_on_life_loss", type=str2bool, default=False)
    parser.add_argument("--clip_reward", type=str2bool, default=True)
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--full_action_space", type=str2bool, default=False)
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


def parse_cfg(argv=None, evaluation: bool = False):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    add_demon_attack_env_args(parser)
    demon_attack_override_defaults(parser)
    return parse_full_cfg(parser, argv)


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    append_cli_arg(args, "--env_id", env_cfg.get("env_id", "ALE/DemonAttack-v5"))
    append_cli_arg(args, "--screen_size", env_cfg.get("screen_size", 84))
    append_cli_arg(args, "--noop_max", env_cfg.get("noop_max", 30))
    append_cli_arg(args, "--action_repeat", env_cfg.get("action_repeat", 1))
    append_cli_arg(args, "--terminal_on_life_loss", env_cfg.get("terminal_on_life_loss", False))
    append_cli_arg(args, "--clip_reward", env_cfg.get("clip_reward", True))
    append_cli_arg(args, "--repeat_action_probability", env_cfg.get("repeat_action_probability", 0.0))
    append_cli_arg(args, "--full_action_space", env_cfg.get("full_action_space", False))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus
    from sample_factory.train import run_rl

    from training.sample_factory_common.common import seed_everything

    register_demon_attack_envs()
    requested_mode = str(config.get("backend", {}).get("run_mode", "train"))
    cfg = parse_cfg(build_cli_args_from_config(config), evaluation=requested_mode in {"eval", "train_eval"})
    seed_everything(int(cfg.seed or 0))

    status = ExperimentStatus.SUCCESS
    if cfg.mode in {"train", "train_eval"}:
        status = run_rl(cfg)
    if cfg.mode in {"eval", "train_eval"}:
        latencies = parse_int_list(cfg.eval_queue_latency_values)
        for latency in latencies:
            result = evaluate_queue_latency(cfg, int(latency))
            print(
                f"[eval] latency={latency} mean_return={result['mean_return']:.3f} "
                f"std_return={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
            )
    return int(status) if isinstance(status, int) else 0
