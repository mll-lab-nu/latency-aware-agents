from __future__ import annotations

import argparse
import copy
import functools
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from training.sample_factory_common.common import (
    env_instance_seed,
    maybe_set_cli_override,
    parse_int_list,
    seed_everything,
    str2bool,
)
from training.sample_factory_common.config_adapter import append_cli_arg, build_common_sf_args
from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper, QueueLatencyWrapper
from training.sample_factory_common.queue_eval import (
    finish_queue_eval_wandb,
    init_queue_eval_wandb,
    log_queue_eval_result,
)


SF_ENV_NAME = "doom_deadly_corridor"
_ENVS_REGISTERED = False
_MODELS_REGISTERED = False


def make_latency_doom_env_from_spec(
    spec,
    env_name: str,
    cfg,
    env_config,
    render_mode: Optional[str] = None,
    **kwargs,
):
    from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec as sf_make_doom_env_from_spec

    spec_to_use = copy.copy(spec)
    if bool(getattr(cfg, "disable_reward_scaling", False)):
        spec_to_use.reward_scaling = 1.0

    env = sf_make_doom_env_from_spec(
        spec_to_use,
        env_name,
        cfg,
        env_config,
        render_mode=render_mode,
        **kwargs,
    )
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
        latency_info_keys=("queue_latency",),
        add_latency_info=bool(getattr(cfg, "add_latency_info", False)),
    )
    env = EpisodeMetricsWrapper(env, frames_per_decision=int(getattr(cfg, "env_frameskip", 4) or 1))
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
    parser.add_argument("--eval-queue-latency-values", type=str, default="0,1,2,3,4,5,10,15,40")
    parser.add_argument("--eval-max-steps", type=int, default=30000)
    parser.add_argument("--eval-raw-reward", action="store_true")
    parser.add_argument("--eval-deterministic", type=str2bool, default=True)
    parser.add_argument("--disable-reward-scaling", type=str2bool, default=False)
    parser.add_argument("--max_episode_steps", type=int, default=0)


def parse_cfg(argv=None):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sf_examples.vizdoom.doom.doom_params import (
        add_doom_env_args,
        add_doom_env_eval_args,
        doom_override_defaults,
    )

    parser, _ = parse_sf_args(argv=argv, evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    add_latency_args(parser)
    doom_override_defaults(parser)
    return parse_full_cfg(parser, argv)


def _load_policy_state_dict(cfg, actor_critic, device: torch.device) -> None:
    from sample_factory.algo.learning.learner import Learner

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if not checkpoint_dict:
        raise RuntimeError("Could not load checkpoint")
    actor_critic.load_state_dict(checkpoint_dict["model"])


def _prepare_eval_cfg(base_cfg, latency_value: int):
    cfg = copy.deepcopy(base_cfg)
    maybe_set_cli_override(cfg, "queue_latency_values", str(int(latency_value)))
    maybe_set_cli_override(cfg, "latency_mode", "episode_fixed")
    maybe_set_cli_override(cfg, "latency_sampling_mode", "uniform_all")
    maybe_set_cli_override(cfg, "episodes_per_stage", int(10**9))
    maybe_set_cli_override(cfg, "randomize_within_stage", False)
    maybe_set_cli_override(cfg, "eval_env_frameskip", int(cfg.env_frameskip))
    maybe_set_cli_override(cfg, "num_envs", 1)
    maybe_set_cli_override(cfg, "no_render", True)
    maybe_set_cli_override(cfg, "save_video", False)
    maybe_set_cli_override(cfg, "fps", 0)
    maybe_set_cli_override(cfg, "eval_deterministic", bool(getattr(base_cfg, "eval_deterministic", True)))
    maybe_set_cli_override(cfg, "disable_reward_scaling", bool(getattr(base_cfg, "eval_raw_reward", False)))
    return cfg


@torch.no_grad()
def evaluate_latency(base_cfg, latency_value: int) -> dict[str, Any]:
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    cfg = _prepare_eval_cfg(base_cfg, latency_value)
    cfg = load_from_checkpoint(cfg)
    cfg.queue_latency_values = str(int(latency_value))
    cfg.latency_mode = "episode_fixed"
    cfg.latency_sampling_mode = "uniform_all"
    cfg.eval_env_frameskip = cfg.env_frameskip

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)
    _load_policy_state_dict(cfg, actor_critic, device)

    returns: list[float] = []
    lengths: list[int] = []
    eval_episodes = int(getattr(base_cfg, "eval_episodes", 20))
    eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 30000))
    max_total_frames = int(1e9 if eval_max_steps <= 0 else eval_max_steps * max(1, eval_episodes))

    obs, _ = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros([env.num_agents], dtype=torch.float32, device=device)
    episode_length = np.zeros([env.num_agents], dtype=np.int64)
    total_frames = 0

    while len(returns) < eval_episodes and total_frames < max_total_frames:
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
        actions = policy_outputs["actions"]
        if bool(getattr(base_cfg, "eval_deterministic", True)):
            actions = argmax_actions(actor_critic.action_distribution())
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        rnn_states = policy_outputs["new_rnn_states"]
        obs, rew, terminated, truncated, _ = env.step(actions)
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        dones = make_dones(terminated, truncated)
        episode_reward += rew.float()
        episode_length += 1
        total_frames += 1

        for agent_i, done_flag in enumerate(dones.cpu().numpy()):
            if done_flag and len(returns) < eval_episodes:
                returns.append(float(episode_reward[agent_i].item()))
                lengths.append(int(episode_length[agent_i]))
                episode_reward[agent_i] = 0.0
                episode_length[agent_i] = 0
                rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])

    env.close()
    if not returns:
        returns = [float("nan")]
        lengths = [0]
    return {
        "latency": int(latency_value),
        "mean_return": float(np.nanmean(returns)),
        "std_return": float(np.nanstd(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
    }


def build_cli_args_from_config(config: dict) -> list[str]:
    env_cfg = config["env"]
    evaluation_cfg = config.get("evaluation", {})
    args = build_common_sf_args(config, sf_env_name=env_cfg.get("env_id", SF_ENV_NAME))
    append_cli_arg(args, "--env_frameskip", env_cfg.get("env_frameskip", 4))
    append_cli_arg(args, "--eval_env_frameskip", env_cfg.get("env_frameskip", 4))
    append_cli_arg(args, "--res_w", env_cfg.get("res_w", 128))
    append_cli_arg(args, "--res_h", env_cfg.get("res_h", 72))
    append_cli_arg(args, "--wide_aspect_ratio", env_cfg.get("wide_aspect_ratio", False))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    append_cli_arg(args, "--disable-reward-scaling", evaluation_cfg.get("eval_raw_reward", False))
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus
    from sample_factory.train import run_rl

    register_deadly_corridor_components()
    cfg = parse_cfg(build_cli_args_from_config(config))
    seed_everything(int(cfg.seed or 0))

    status = ExperimentStatus.SUCCESS
    if cfg.mode in {"train", "train_eval"}:
        status = run_rl(cfg)
    if cfg.mode in {"eval", "train_eval"}:
        latencies = parse_int_list(cfg.eval_queue_latency_values)
        wandb_run, owned_wandb_run = init_queue_eval_wandb(cfg)
        try:
            for latency in latencies:
                result = evaluate_latency(cfg, int(latency))
                log_queue_eval_result(wandb_run, result)
                print(
                    f"[eval] latency={latency} mean_return={result['mean_return']:.3f} "
                    f"std_return={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
                )
        finally:
            finish_queue_eval_wandb(wandb_run, owned_wandb_run)
    return int(status) if isinstance(status, int) else 0
