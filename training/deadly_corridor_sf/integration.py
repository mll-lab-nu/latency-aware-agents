from __future__ import annotations

import argparse
import copy
from collections import deque
import functools
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from training.sample_factory_common.common import (
    env_instance_seed,
    maybe_set_cli_override,
    parse_int_list,
    seed_everything,
    str2bool,
)
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths
from training.sample_factory_common.config_adapter import append_cli_arg, build_common_sf_args
from training.sample_factory_common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper, QueueLatencyWrapper, RewardTraceWrapper
from training.sample_factory_common.queue_eval import (
    finish_queue_eval_wandb,
    init_queue_eval_wandb,
    log_queue_eval_result,
    write_queue_eval_result,
)
from training.sample_factory_common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "doom_deadly_corridor"
SF_ENV_NAME_DISCRETE7 = "doom_deadly_corridor_discrete7"
_ENVS_REGISTERED = False
_MODELS_REGISTERED = False

DEADLY_CORRIDOR_DISCRETE7_ACTION_LABELS = {
    0: "MOVE_FORWARD",
    1: "MOVE_BACKWARD",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "TURN_LEFT",
    5: "TURN_RIGHT",
    6: "ATTACK",
}


def build_deadly_corridor_discrete7_spec():
    from sf_examples.vizdoom.doom.doom_utils import DoomSpec

    return DoomSpec(
        SF_ENV_NAME_DISCRETE7,
        "deadly_corridor.cfg",
        Discrete(8),  # internal idle + 7 primitive buttons
        0.01,
    )


def get_deadly_corridor_spec(env_id: str):
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS

    if str(env_id) == SF_ENV_NAME_DISCRETE7:
        return build_deadly_corridor_discrete7_spec()

    spec = next((item for item in DOOM_ENVS if item.name == str(env_id)), None)
    if spec is None:
        raise ValueError(f"Unknown ViZDoom env spec: {env_id}")
    return spec


def _doom_raw_reward_from_scaled(transformed_reward: float, info: dict[str, Any], reward_scale: float) -> float:
    if "raw_reward" in info:
        return float(info["raw_reward"])
    if reward_scale == 0.0:
        return float(transformed_reward)
    return float(transformed_reward) / float(reward_scale)


class DoomFrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        self._dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        if self._dict_obs:
            obs_space = env.observation_space.spaces.get("obs")
            if not isinstance(obs_space, gym.spaces.Box):
                raise TypeError("DoomFrameStackWrapper expects observation_space['obs'] to be a Box")
        elif isinstance(env.observation_space, gym.spaces.Box):
            obs_space = env.observation_space
        else:
            raise TypeError("DoomFrameStackWrapper expects a Box or Dict observation space")
        base_shape = tuple(int(dim) for dim in obs_space.shape)
        if len(base_shape) != 3:
            raise ValueError(f"Expected channel-first Doom observation, got shape {base_shape}")

        self.num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        stacked_shape = (base_shape[0] * self.num_frames, base_shape[1], base_shape[2])
        stacked_obs_space = gym.spaces.Box(
            low=obs_space.low.min() if np.ndim(obs_space.low) > 0 else obs_space.low,
            high=obs_space.high.max() if np.ndim(obs_space.high) > 0 else obs_space.high,
            shape=stacked_shape,
            dtype=obs_space.dtype,
        )
        if self._dict_obs:
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["obs"] = stacked_obs_space
            self.observation_space = gym.spaces.Dict(new_spaces)
        else:
            self.observation_space = stacked_obs_space

    def _extract_frame(self, observation) -> np.ndarray:
        if self._dict_obs:
            if not isinstance(observation, dict):
                raise TypeError(f"Expected dict observation, got {type(observation)!r}")
            return np.array(observation["obs"], copy=True)
        return np.array(observation, copy=True)

    def _stack_obs(self, observation):
        stacked_frames = np.concatenate(tuple(self._frames), axis=0)
        if self._dict_obs:
            if not isinstance(observation, dict):
                raise TypeError(f"Expected dict observation, got {type(observation)!r}")
            stacked = dict(observation)
            stacked["obs"] = stacked_frames
            return stacked
        return stacked_frames

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = self._extract_frame(observation)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(frame.copy())
        return self._stack_obs(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._extract_frame(observation))
        return self._stack_obs(observation), reward, terminated, truncated, info


class DoomDiscrete7ActionWrapper(gym.ActionWrapper):
    """Expose Deadly Corridor as 7 primitive policy actions, with an internal idle action for latency handling."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Discrete(7)
        self.noop_action = -1
        self.action_labels = dict(DEADLY_CORRIDOR_DISCRETE7_ACTION_LABELS)

    def action(self, action):
        action_value = int(np.asarray(action).item())
        if action_value == self.noop_action:
            return 0  # internal Doom idle action
        if action_value < 0 or action_value >= int(self.action_space.n):
            raise ValueError(f"Invalid discrete7 Deadly Corridor action: {action_value}")
        return action_value + 1

    def reverse_action(self, action):
        action_value = int(np.asarray(action).item())
        if action_value <= 0:
            return self.noop_action
        return action_value - 1


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
    policy_noop_action = None
    if env_name == SF_ENV_NAME_DISCRETE7:
        env = DoomDiscrete7ActionWrapper(env)
        policy_noop_action = env.noop_action
    env = RewardTraceWrapper(
        env,
        raw_reward_fn=lambda transformed_reward, info, scale=float(spec_to_use.reward_scaling): _doom_raw_reward_from_scaled(
            transformed_reward,
            info,
            scale,
        ),
    )
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    if frame_stack > 1:
        env = DoomFrameStackWrapper(env, num_frames=frame_stack)
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
        noop_action=policy_noop_action,
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
    env = EpisodeMetricsWrapper(
        env,
        frames_per_decision=int(getattr(cfg, "env_frameskip", 4) or 1),
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

    discrete7_spec = build_deadly_corridor_discrete7_spec()
    register_env(discrete7_spec.name, functools.partial(make_latency_doom_env_from_spec, discrete7_spec))
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
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--episode_metrics_path", type=str, default="")


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
    checkpoint_dict = load_sample_factory_checkpoint_from_paths(checkpoints, device)
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
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = torch.zeros([env.num_agents], dtype=torch.float32, device=device)
    episode_length = np.zeros([env.num_agents], dtype=np.int64)
    total_frames = 0

    while len(returns) < eval_episodes and total_frames < max_total_frames:
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states)
        actions = policy_outputs["actions"]
        if bool(getattr(base_cfg, "eval_deterministic", True)):
            actions = argmax_actions(actor_critic.action_distribution())
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        rnn_states = policy_outputs["new_rnn_states"]
        obs, rew, terminated, truncated, _ = env.step(actions)
        dones = make_dones(terminated, truncated)
        reward_tensor = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32, device=device)
        episode_reward += reward_tensor.to(device=device, dtype=torch.float32)
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
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
    append_cli_arg(args, "--disable-reward-scaling", evaluation_cfg.get("eval_raw_reward", False))
    return args


def run_from_config(config: dict) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    register_deadly_corridor_components()
    cfg = parse_cfg(build_cli_args_from_config(config))
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    output_dir = logging_cfg.get("output_dir")
    if output_dir:
        cfg.output_dir = str(output_dir)
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
                    result = evaluate_latency(cfg, int(latency))
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
