from __future__ import annotations

import argparse
import copy
from collections import deque
import functools
import json
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.sample_factory_common.common import (
    env_instance_seed,
    maybe_set_cli_override,
    parse_int_list,
    seed_everything,
    str2bool,
)
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths
from training.sample_factory_common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
)
from training.sample_factory_common.episode_metrics import start_episode_metrics_aggregator, stop_episode_metrics_aggregator
from training.sample_factory_common.latency_config import (
    raw_frames_to_latency_ms,
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.sample_factory_common.latency_bench_periodic_eval import run_sample_factory_training
from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper, EpisodeMetricsWrapper, RewardTraceWrapper
from training.sample_factory_common.queue_eval import (
    evaluate_queue_latency_distribution,
    finish_queue_eval_wandb,
    has_queue_latency_distribution,
    init_queue_eval_wandb,
    log_queue_eval_result,
    queue_eval_output_dir,
    write_queue_eval_result,
)
from training.sample_factory_common.vizdoom_init_guard import ensure_vizdoom_init_guard_installed
from training.sample_factory_common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "doom_deadly_corridor"
_ENVS_REGISTERED = False
_MODELS_REGISTERED = False


def _clock_from_cfg(cfg) -> EnvClock:
    env_fps = getattr(cfg, "env_fps", None)
    obs_fps = getattr(cfg, "obs_fps", None)
    if env_fps is None or obs_fps is None:
        raise ValueError("DeadlyCorridor Sample Factory requires env_fps and obs_fps")
    return EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps))


def _force_doom_raw_frame_step_cfg(cfg) -> None:
    maybe_set_cli_override(cfg, "env_frameskip", 1)
    maybe_set_cli_override(cfg, "eval_env_frameskip", 1)


def _doom_raw_reward_from_scaled(env_reward: float, info: dict[str, Any], reward_scale: float) -> float:
    if "raw_reward" in info:
        return float(info["raw_reward"])
    if reward_scale == 0.0:
        return float(env_reward)
    return float(env_reward) / float(reward_scale)


class DoomFrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        self._dict_observation = isinstance(env.observation_space, gym.spaces.Dict)
        if self._dict_observation:
            obs_space = env.observation_space.spaces.get("obs")
        else:
            obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("DoomFrameStackWrapper expects a Box observation space")
        base_shape = tuple(int(dim) for dim in obs_space.shape)
        if len(base_shape) != 3:
            raise ValueError(f"Expected channel-first Doom observation, got shape {base_shape}")

        self.num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        stacked_shape = (base_shape[0] * self.num_frames, base_shape[1], base_shape[2])
        stacked_obs_space = gym.spaces.Box(low=0, high=255, shape=stacked_shape, dtype=obs_space.dtype)
        if self._dict_observation:
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["obs"] = stacked_obs_space
            self.observation_space = gym.spaces.Dict(new_spaces)
        else:
            self.observation_space = stacked_obs_space

    def _stack_obs(self, observation):
        stacked_frames = np.concatenate(tuple(self._frames), axis=0)
        if self._dict_observation:
            if not isinstance(observation, dict):
                raise TypeError(f"Expected dict observation, got {type(observation)!r}")
            stacked = dict(observation)
            stacked["obs"] = stacked_frames
            return stacked
        return stacked_frames

    def _extract_frame(self, observation):
        if self._dict_observation:
            if not isinstance(observation, dict):
                raise TypeError(f"Expected dict observation, got {type(observation)!r}")
            return observation["obs"]
        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        frame = np.array(self._extract_frame(observation), copy=True)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(frame.copy())
        return self._stack_obs(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.array(self._extract_frame(observation), copy=True))
        return self._stack_obs(observation), reward, terminated, truncated, info


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
    _force_doom_raw_frame_step_cfg(cfg)
    clock = _clock_from_cfg(cfg)

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
    frame_stack = int(getattr(cfg, "frame_stack", 1) or 1)
    if frame_stack > 1:
        env = DoomFrameStackWrapper(env, num_frames=frame_stack)
    max_episode_steps = int(getattr(cfg, "max_episode_steps", 0) or 0)
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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "train_eval"])
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
    parser.add_argument("--episode_metrics_path", type=str, default="")
    parser.add_argument("--vizdoom_init_max_parallel", type=int, default=10)
    parser.add_argument("--vizdoom_init_lock_timeout_sec", type=float, default=5.0)
    parser.add_argument("--vizdoom_init_lock_max_wait_sec", type=float, default=60.0)
    parser.add_argument("--vizdoom_init_retry_jitter_sec", type=float, default=0.25)


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
    parser.set_defaults(env_frameskip=1, eval_env_frameskip=1)
    cfg = parse_full_cfg(parser, argv)
    _force_doom_raw_frame_step_cfg(cfg)
    return cfg


def _load_policy_state_dict(cfg, actor_critic, device: torch.device) -> None:
    from sample_factory.algo.learning.learner import Learner

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = load_sample_factory_checkpoint_from_paths(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _single_agent_action(value: Any) -> Any:
    action = _jsonable(value)
    if isinstance(action, list) and len(action) == 1 and isinstance(action[0], list):
        return action[0]
    return action


def _synchronous_eval_fps(base_cfg, default: int = 35) -> int:
    fallback = int(default)
    if fallback <= 0:
        fallback = 35
    try:
        fps = int(float(getattr(base_cfg, "fps", fallback)))
    except (TypeError, ValueError):
        fps = fallback
    return fps if fps > 0 else fallback


def _prepare_eval_cfg(base_cfg, latency_value: int):
    cfg = copy.deepcopy(base_cfg)
    maybe_set_cli_override(cfg, "latency_type", "zero" if int(latency_value) == 0 else "fixed")
    maybe_set_cli_override(
        cfg,
        "fixed_latency_ms",
        raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps"))),
    )
    _force_doom_raw_frame_step_cfg(cfg)
    maybe_set_cli_override(cfg, "num_envs", 1)
    maybe_set_cli_override(cfg, "no_render", True)
    maybe_set_cli_override(cfg, "save_video", False)
    maybe_set_cli_override(cfg, "fps", _synchronous_eval_fps(base_cfg))
    maybe_set_cli_override(cfg, "eval_deterministic", bool(getattr(base_cfg, "eval_deterministic", True)))
    maybe_set_cli_override(cfg, "disable_reward_scaling", bool(getattr(base_cfg, "eval_raw_reward", False)))
    if int(getattr(cfg, "max_episode_steps", 0) or 0) <= 0:
        eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
        if eval_max_steps > 0:
            maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)
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
    cfg.latency_type = "zero" if int(latency_value) == 0 else "fixed"
    cfg.fixed_latency_ms = raw_frames_to_latency_ms(latency_value, env_fps=float(getattr(base_cfg, "env_fps")))
    _force_doom_raw_frame_step_cfg(cfg)
    if int(getattr(cfg, "max_episode_steps", 0) or 0) <= 0:
        eval_max_steps = int(getattr(base_cfg, "eval_max_steps", 0) or 0)
        if eval_max_steps > 0:
            maybe_set_cli_override(cfg, "max_episode_steps", eval_max_steps)

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
    episode_indices = np.zeros([env.num_agents], dtype=np.int64)
    total_frames = 0
    trace_handle = None
    output_dir = queue_eval_output_dir(base_cfg)
    if output_dir is not None:
        trace_handle = (output_dir / f"queue_eval_latency_{int(latency_value)}_actions.jsonl").open(
            "w",
            encoding="utf-8",
        )

    try:
        while len(returns) < eval_episodes and total_frames < max_total_frames:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if bool(getattr(base_cfg, "eval_deterministic", True)):
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            action_for_trace = _single_agent_action(actions)
            rnn_states = policy_outputs["new_rnn_states"]
            obs, rew, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated)
            reward_tensor = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32, device=device)
            reward_tensor = reward_tensor.to(device=device, dtype=torch.float32)
            episode_reward += reward_tensor
            if trace_handle is not None:
                done_values = dones.detach().cpu().numpy()
                reward_values = reward_tensor.detach().cpu().numpy()
                for agent_i, done_flag in enumerate(done_values):
                    trace_handle.write(
                        json.dumps(
                            {
                                "latency": int(latency_value),
                                "episode_id": int(episode_indices[agent_i]),
                                "env_step": int(episode_length[agent_i]),
                                "total_frame": int(total_frames),
                                "action": action_for_trace,
                                "reward": float(reward_values[agent_i]),
                                "episode_return": float(episode_reward[agent_i].item()),
                                "done": bool(done_flag),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            episode_length += 1
            total_frames += 1

            for agent_i, done_flag in enumerate(dones.cpu().numpy()):
                if done_flag and len(returns) < eval_episodes:
                    returns.append(float(episode_reward[agent_i].item()))
                    lengths.append(int(episode_length[agent_i]))
                    episode_reward[agent_i] = 0.0
                    episode_length[agent_i] = 0
                    episode_indices[agent_i] += 1
                    rnn_states[agent_i] = torch.zeros_like(rnn_states[agent_i])
    finally:
        if trace_handle is not None:
            trace_handle.close()
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
    backend_cfg = config.get("backend", {})
    evaluation_cfg = config.get("evaluation", {})
    scheduler_cfg = config.get("scheduler")
    if not isinstance(scheduler_cfg, dict):
        raise ValueError("scheduler must be a mapping for DeadlyCorridor Sample Factory")
    args = build_common_sf_args(config, sf_env_name=env_cfg.get("env_id", SF_ENV_NAME))
    if str(backend_cfg.get("run_mode", "train")).strip().lower() in {"eval", "train_eval"}:
        append_cli_arg(args, "--fps", env_cfg.get("env_fps", 35))
    append_cli_arg(args, "--env-fps", env_cfg["env_fps"])
    append_cli_arg(args, "--obs-fps", env_cfg["obs_fps"])
    append_cli_arg(args, "--res_w", env_cfg.get("res_w", 128))
    append_cli_arg(args, "--res_h", env_cfg.get("res_h", 72))
    append_cli_arg(args, "--wide_aspect_ratio", env_cfg.get("wide_aspect_ratio", False))
    append_cli_arg(args, "--frame_stack", env_cfg.get("frame_stack", 1))
    append_cli_arg(args, "--max_episode_steps", env_cfg.get("max_episode_steps"))
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
    cfg = parse_cfg(build_cli_args_from_config(config))
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    apply_runtime_logging_config(cfg, logging_cfg)
    seed_everything(int(cfg.seed or 0))
    patch_sample_factory_wandb()

    metrics_stop_event = None
    metrics_thread = None
    if cfg.mode in {"train", "train_eval"}:
        metrics_stop_event, metrics_thread = start_episode_metrics_aggregator(cfg)

    status = ExperimentStatus.SUCCESS
    try:
        if cfg.mode in {"train", "train_eval"}:
            status = run_sample_factory_training(cfg, config)
        if cfg.mode in {"eval", "train_eval"}:
            wandb_run, owned_wandb_run = init_queue_eval_wandb(cfg)
            try:
                if has_queue_latency_distribution(cfg):
                    result = evaluate_queue_latency_distribution(cfg)
                    write_queue_eval_result(cfg, result)
                    log_queue_eval_result(wandb_run, result)
                    print(
                        f"[eval] distribution={result['latency_distribution_label']} "
                        f"mean_return={result['mean_return']:.3f} "
                        f"std_return={result['std_return']:.3f} mean_len={result['mean_length']:.1f}"
                    )
                else:
                    latencies = parse_int_list(cfg.eval_latency_raw_frame_values)
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
