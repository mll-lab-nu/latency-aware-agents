from __future__ import annotations

import argparse
import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionHoldPolicy, OrderingPolicy
from latency_bench.envs.gymnasium_task import (
    gymnasium_action_space_contract,
    gymnasium_discrete_action_labels,
    gymnasium_discrete_action_values,
    gymnasium_noop_action_id,
)
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.factory import build_persistence_policy
from training.common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
)
from training.common.episode_metrics import (
    start_episode_metrics_aggregator,
    stop_episode_metrics_aggregator,
)
from training.common.latency_bench_periodic_eval import run_sample_factory_training
from training.common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.latency_wrapper import (
    ActionLatencyWrapper,
    EpisodeMetricsWrapper,
    RewardTraceWrapper,
    clone_action,
)
from training.common.raw_rgb_export import RawRgbFrameStackInfoWrapper
from training.common.sample_factory_config import (
    SampleFactoryConfigParser,
    force_sample_factory_raw_frame_step_cfg,
)
from training.common.utils import env_instance_seed, seed_everything, str2bool
from training.common.wandb_compat import patch_sample_factory_wandb


SF_ENV_NAME = "latency_gymnasium"
_ENVS_REGISTERED = False
_SAMPLE_FACTORY_PROBE_ENV_CONFIG = SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def _sample_factory_env_config(env_config: Any) -> Any:
    return _SAMPLE_FACTORY_PROBE_ENV_CONFIG if env_config is None else env_config


def _gymnasium_make_kwargs(cfg: Any) -> dict[str, Any]:
    return json.loads(cfg.gym_make_kwargs_json)


def _gymnasium_action_labels(cfg: Any) -> list[str]:
    return json.loads(cfg.gym_action_labels_json)


def _gymnasium_action_values(cfg: Any) -> list[Any]:
    return json.loads(cfg.gym_action_values_json)


def _gymnasium_action_space(cfg: Any) -> dict[str, Any]:
    return json.loads(cfg.gym_action_space_json)


def _gymnasium_noop_action(cfg: Any) -> Any:
    return json.loads(cfg.gym_noop_action_json)


def _render_rgb_frame(env: gym.Env):
    return env.render()


class GymnasiumActionCatalogWrapper(gym.Wrapper):
    """Expose a stable discrete policy catalog over native Gymnasium actions."""

    def __init__(self, env: gym.Env, native_actions: list[Any]):
        super().__init__(env)
        self.native_actions = native_actions
        self.action_space = Discrete(len(native_actions))

    def step(self, action):
        return self.env.step(self.native_actions[action])


class ChannelFirstFrameStackWrapper(gym.Wrapper):
    """Stack channel-first image observations along the channel axis."""

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        self.num_frames = num_frames
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low] * num_frames, axis=0),
            high=np.concatenate([env.observation_space.high] * num_frames, axis=0),
            dtype=env.observation_space.dtype,
        )
        self.frames = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames = [np.asarray(observation).copy() for _ in range(self.num_frames)]
        return np.concatenate(self.frames, axis=0), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames = [*self.frames[1:], np.asarray(observation).copy()]
        return np.concatenate(self.frames, axis=0), reward, terminated, truncated, info


class GymnasiumStateInfoWrapper(gym.Wrapper):
    """Keep each raw Gymnasium observation beside the policy observation."""

    @staticmethod
    def _state(observation):
        if isinstance(observation, dict):
            observation = observation["obs"]
        return np.asarray(observation).copy()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, {**info, "gymnasium_state": self._state(observation)}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, {
            **info,
            "gymnasium_state": self._state(observation),
        }


class GymnasiumActionLatencyWrapper(ActionLatencyWrapper):
    def __init__(self, *args, noop_action: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self.noop_action = noop_action
        self.action_scheduler = ActionScheduler(
            clock=self.clock,
            noop_action=Action(
                value=clone_action(self.noop_action),
                name="noop",
                is_noop=True,
            ),
            persistence_policy=build_persistence_policy(self.hold_policy),
            ordering_policy=self.ordering_policy,
            hold_last_chunk_action=self.action_scheduler.hold_last_chunk_action,
        )
        self.current_applied_action = clone_action(self.noop_action)


class G1RawRequestWrapper(gym.ActionWrapper):
    """Encode normalized G1 actions and attach chunk execution clocks for replay."""

    def __init__(self, env, action_chunk_horizon: int = 1, latency_env=None):
        super().__init__(env)
        self.action_dim = 29 + (14 if self.unwrapped._raw_learn_hands else 0)
        self.action_chunk_horizon = action_chunk_horizon
        self.latency_env = latency_env
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(self.action_dim * action_chunk_horizon,),
            dtype=np.float32,
        )
        if action_chunk_horizon > 1:
            spaces = (
                dict(env.observation_space.spaces)
                if isinstance(env.observation_space, gym.spaces.Dict)
                else {"obs": env.observation_space}
            )
            spaces.update(
                replay_clock=gym.spaces.Box(
                    -1, np.iinfo(np.int64).max, (7,), dtype=np.int64
                ),
            )
            self.observation_space = gym.spaces.Dict(spaces)

    def action(self, action):
        if self.action_chunk_horizon == 1:
            return self.unwrapped.g1_absolute_request_from_raw_action(action)
        chunk = np.asarray(action).reshape(self.action_chunk_horizon, self.action_dim)
        return np.stack(
            [self.unwrapped.g1_absolute_request_from_raw_action(row) for row in chunk]
        )

    def _chunk_observation(self, observation, applied_source, applied_index, admitted, applied_window):
        result = observation if isinstance(observation, dict) else {"obs": observation}
        return {
            **result,
            "replay_clock": np.asarray(
                (
                    self.latency_env.episode_idx,
                    self.latency_env.env_step_idx,
                    applied_source,
                    applied_index,
                    admitted,
                    *applied_window,
                ),
                dtype=np.int64,
            ),
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.action_chunk_horizon == 1:
            return observation, info
        return self._chunk_observation(observation, -1, -1, False, (-1, -1)), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        if self.action_chunk_horizon == 1:
            return observation, reward, terminated, truncated, info
        return (
            self._chunk_observation(
                observation,
                info["applied_source_raw_frame"],
                info["applied_chunk_index"],
                info["command_admitted"],
                info["applied_chunk_window"],
            ),
            reward,
            terminated,
            truncated,
            info,
        )


def make_gymnasium_sf_env(
    _full_env_name: str,
    cfg: Any = None,
    env_config: Any = None,
    render_mode: Optional[str] = None,
):
    del render_mode
    force_sample_factory_raw_frame_step_cfg(cfg)
    env_config = _sample_factory_env_config(env_config)
    for module_name in json.loads(cfg.gym_registration_imports_json):
        import_module(module_name)

    make_kwargs = _gymnasium_make_kwargs(cfg)
    raw_g1_requests = (
        cfg.gym_task_name == "g1_sonic_balance_simple"
        and make_kwargs["control_mode"] == "raw_body_target"
    )
    if raw_g1_requests:
        make_kwargs["control_mode"] = "direct_absolute_position"
    env = gym.make(cfg.gym_env_id, **make_kwargs)
    if cfg.gym_state_labels_json:
        env = GymnasiumStateInfoWrapper(env)
    action_space_contract = _gymnasium_action_space(cfg)
    if action_space_contract["type"] == "discrete":
        env = GymnasiumActionCatalogWrapper(env, _gymnasium_action_values(cfg))
    frame_stack = cfg.frame_stack
    if frame_stack > 1:
        env = ChannelFirstFrameStackWrapper(env, frame_stack)
    if bool(cfg.export_env_raw_rgb_frames):
        env = RawRgbFrameStackInfoWrapper(
            env,
            num_frames=frame_stack,
            raw_frame_fn=_render_rgb_frame,
        )
    env = RewardTraceWrapper(env)

    clock = EnvClock(env_fps=cfg.env_fps, obs_fps=cfg.obs_fps)
    wrapper_seed = env_instance_seed(
        base_seed=int(cfg.seed),
        worker_index=int(env_config.worker_index),
        vector_index=int(env_config.vector_index),
        env_id=int(env_config.env_id),
    )
    latency_seed = sample_factory_latency_seed_from_flat_config(
        cfg,
        fallback_seed=wrapper_seed,
        worker_index=int(env_config.worker_index),
        vector_index=int(env_config.vector_index),
        env_id=int(env_config.env_id),
    )
    latency_params = sample_factory_latency_params_from_flat_config(cfg, env_fps=clock.env_fps)
    if latency_params.profile_path is not None and cfg.latency_seed is not None:
        latency_seed = int(cfg.latency_seed)
    env = GymnasiumActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=latency_params.fixed_latency_raw_frames,
        env_fps=clock.env_fps,
        obs_fps=clock.obs_fps,
        hold_policy=cfg.hold_policy,
        ordering_policy=cfg.ordering_policy,
        latency_distribution_spec=latency_params.latency_distribution_spec,
        sample_distribution_per_episode=latency_params.sample_distribution_per_episode,
        profile_path=latency_params.profile_path,
        profile_worker_slot=latency_params.profile_worker_slot,
        profile_method=latency_params.profile_method,
        simulated_worker_capacity=latency_params.simulated_worker_capacity,
        seed=latency_seed,
        env_reset_seed_base=wrapper_seed + 104729,
        add_latency_info=bool(cfg.add_latency_info),
        max_pending_actions=cfg.max_pending_actions,
        hold_last_chunk_action=cfg.hold_last_chunk_action,
        noop_action=(
            cfg.gym_noop_action_id
            if action_space_contract["type"] == "discrete"
            else _gymnasium_noop_action(cfg)
        ),
    )
    if raw_g1_requests:
        env = G1RawRequestWrapper(
            env, action_chunk_horizon=cfg.fasttd3_action_chunk_horizon, latency_env=env
        )
    action_labels = (
        {
            action_id: name
            for action_id, name in enumerate(_gymnasium_action_labels(cfg))
        }
        if action_space_contract["type"] == "discrete"
        else {}
    )
    return EpisodeMetricsWrapper(
        env,
        frames_per_decision=clock.obs_stride_raw_frames,
        metrics_path=str(cfg.episode_metrics_path),
        worker_index=int(env_config.worker_index),
        vector_index=int(env_config.vector_index),
        env_id=int(env_config.env_id),
        issued_action_labels=action_labels,
    )


def register_gymnasium_envs() -> None:
    global _ENVS_REGISTERED
    if _ENVS_REGISTERED:
        return
    from sample_factory.envs.env_utils import register_env

    register_env(SF_ENV_NAME, make_gymnasium_sf_env)
    _ENVS_REGISTERED = True


def make_reference_env(cfg: Any):
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.utils.attr_dict import AttrDict

    force_sample_factory_raw_frame_step_cfg(cfg)
    register_gymnasium_envs()
    return make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )


def add_gymnasium_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gym-task-name", type=str, required=True)
    parser.add_argument("--gym-env-id", type=str, required=True)
    parser.add_argument("--gym-make-kwargs-json", type=str, required=True)
    parser.add_argument("--gym-registration-imports-json", type=str, required=True)
    parser.add_argument("--gym-action-space-json", type=str, required=True)
    parser.add_argument("--gym-action-labels-json", type=str)
    parser.add_argument("--gym-action-values-json", type=str)
    parser.add_argument("--gym-noop-action-id", type=int)
    parser.add_argument("--gym-noop-action-json", type=str, required=True)
    parser.add_argument("--gym-base-prompt", type=str, required=True)
    parser.add_argument("--gym-state-labels-json", type=str, default="")
    parser.add_argument("--env-fps", dest="env_fps", type=float, required=True)
    parser.add_argument("--obs-fps", dest="obs_fps", type=float, required=True)
    parser.add_argument("--frame-stack", dest="frame_stack", type=int, required=True)
    parser.add_argument(
        "--export-env-raw-rgb-frames",
        dest="export_env_raw_rgb_frames",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--latency-type", dest="latency_type", type=str, default="zero")
    parser.add_argument("--fixed-latency-ms", dest="fixed_latency_ms", type=float)
    parser.add_argument("--mean-latency-ms", dest="mean_latency_ms", type=float)
    parser.add_argument("--std-latency-ms", dest="std_latency_ms", type=float)
    parser.add_argument("--min-latency-ms", dest="min_latency_ms", type=float)
    parser.add_argument("--max-latency-ms", dest="max_latency_ms", type=float)
    parser.add_argument("--latency-seed", dest="latency_seed", type=int)
    parser.add_argument("--latency-profile-path", dest="latency_profile_path", type=str)
    parser.add_argument("--latency-profile-worker-slot", dest="latency_profile_worker_slot", type=int)
    parser.add_argument("--simulated-worker-capacity", dest="simulated_worker_capacity", type=int)
    parser.add_argument("--add-latency-info", type=str2bool, default=False)
    parser.add_argument("--max-pending-actions", dest="max_pending_actions", type=int)
    parser.add_argument("--hold-last-chunk-action", type=str2bool, default=False)
    parser.add_argument(
        "--hold-policy",
        dest="hold_policy",
        type=str,
        required=True,
        choices=[item.value for item in ActionHoldPolicy],
    )
    parser.add_argument(
        "--ordering-policy",
        dest="ordering_policy",
        type=str,
        required=True,
        choices=[item.value for item in OrderingPolicy],
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-parallel-envs", type=int, default=1)
    parser.add_argument("--eval-latency-raw-frame-values", dest="eval_latency_raw_frame_values", type=str)
    parser.add_argument("--eval-max-steps", type=int, default=30000)
    parser.add_argument("--eval-deterministic", type=str2bool, default=True)
    parser.add_argument("--eval-raw-reward", action="store_true")
    parser.add_argument("--episode_metrics_path", type=str, default="")


def configure_sample_factory_parser(parser: argparse.ArgumentParser) -> None:
    add_gymnasium_env_args(parser)


SAMPLE_FACTORY_CONFIG_PARSER = SampleFactoryConfigParser(
    configure_parser=configure_sample_factory_parser,
    finalize_cfg=force_sample_factory_raw_frame_step_cfg,
)


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    env_cfg = config["env"]
    scheduler_cfg = config["scheduler"]
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    append_cli_arg(args, "--gym-task-name", env_cfg["task_name"])
    append_cli_arg(args, "--gym-env-id", env_cfg["env_id"])
    append_cli_arg(args, "--gym-make-kwargs-json", json.dumps(env_cfg["make_kwargs"]))
    append_cli_arg(
        args,
        "--gym-registration-imports-json",
        json.dumps(env_cfg["registration_imports"]),
    )
    action_space_contract = gymnasium_action_space_contract(env_cfg)
    append_cli_arg(args, "--gym-action-space-json", json.dumps(action_space_contract))
    append_cli_arg(args, "--gym-noop-action-json", json.dumps(env_cfg["noop_action"]))
    if action_space_contract["type"] == "discrete":
        append_cli_arg(
            args,
            "--gym-action-labels-json",
            json.dumps(gymnasium_discrete_action_labels(env_cfg)),
        )
        append_cli_arg(
            args,
            "--gym-action-values-json",
            json.dumps(gymnasium_discrete_action_values(env_cfg)),
        )
        append_cli_arg(args, "--gym-noop-action-id", gymnasium_noop_action_id(env_cfg))
    append_cli_arg(args, "--gym-base-prompt", env_cfg["base_prompt"])
    if "state_space" in env_cfg:
        append_cli_arg(
            args,
            "--gym-state-labels-json",
            json.dumps(env_cfg["state_space"]["labels"]),
        )
    append_cli_arg(args, "--env-fps", env_cfg["env_fps"])
    append_cli_arg(args, "--obs-fps", env_cfg["obs_fps"])
    append_cli_arg(args, "--frame-stack", env_cfg["frame_stack"])
    append_cli_arg(args, "--hold-policy", scheduler_cfg["hold_policy"])
    append_cli_arg(args, "--ordering-policy", scheduler_cfg["ordering_policy"])
    if "hold_last_chunk_action" in scheduler_cfg:
        append_cli_arg(args, "--hold-last-chunk-action", scheduler_cfg["hold_last_chunk_action"])
    append_cli_arg(args, "--use_env_info_cache", False)
    return args


def build_train_cfg_from_config(config: dict[str, Any]) -> Any:
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))
    if "lr_schedule" in config["training"]:
        cfg.lr_schedule = config["training"]["lr_schedule"]
    return cfg


def run_from_config(config: dict[str, Any]) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    register_gymnasium_envs()
    cfg = build_train_cfg_from_config(config)
    apply_runtime_logging_config(cfg, config["logging"])
    seed_everything(int(cfg.seed))
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
