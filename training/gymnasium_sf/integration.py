from __future__ import annotations

import argparse
import json
import os
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionHoldPolicy, OrderingPolicy
from latency_bench.envs.gymnasium_task import (
    gymnasium_action_space_contract,
    gymnasium_discrete_action_labels,
    gymnasium_discrete_action_values,
    gymnasium_noop_action_id,
)
from latency_bench.integrations.sample_factory.actions import scale_box_action
from latency_bench.scheduler.action_queue import ActionScheduler, chunk_context_window
from latency_bench.envs.humanoidbench_sonic_contract import POLICY_INDEX_FOR_MUJOCO
from latency_bench.scheduler.factory import build_persistence_policy
from training.asterix_sf.batched_gpu_env import AsterixBatchedGpuEnv
from training.atlantis_sf.batched_gpu_env import AtlantisBatchedGpuEnv
from training.common.config_adapter import (
    append_cli_arg,
    apply_runtime_logging_config,
    build_common_sf_args,
)
from training.common.episode_metrics import (
    start_episode_metrics_aggregator,
    stop_episode_metrics_aggregator,
    unlink_stale_episode_metrics_file,
)
from training.common.latency_bench_periodic_eval import run_sample_factory_training
from training.common.latency_config import (
    add_sample_factory_latency_args,
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
from training.common.sf_episode_extra_stats_compat import (
    ensure_sample_factory_episode_extra_stats_installed,
)
from training.common.sf_gpu_sampling_compat import (
    ensure_sample_factory_gpu_sampling_compat_installed,
)
from training.common.sf_raw_reward_stats_compat import (
    ensure_sample_factory_raw_reward_console_stats_installed,
)
from training.common.utils import env_instance_seed, seed_everything, str2bool
from training.common.wandb_compat import patch_sample_factory_wandb
from sample_factory.algo.mc_ppo.env import EpisodeEnv


SF_ENV_NAME = "latency_gymnasium"
_ENVS_REGISTERED = False
_SAMPLE_FACTORY_PROBE_ENV_CONFIG = SimpleNamespace(worker_index=0, vector_index=0, env_id=0)
_GPU_ATARI_ENVS = {
    "asterix": AsterixBatchedGpuEnv,
    "atlantis": AtlantisBatchedGpuEnv,
}


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
    """Convert normalized G1 joint requests to the native absolute joint order."""

    def __init__(self, env):
        super().__init__(env)
        action_dim = 29 + (14 if self.unwrapped._raw_learn_hands else 0)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

    def action(self, action):
        return self.unwrapped.g1_absolute_request_from_raw_action(action)

    def reverse_action(self, action):
        action = np.asarray(action)
        ordered = np.concatenate((action[:29][POLICY_INDEX_FOR_MUJOCO], action[29:]))
        ranges = self.unwrapped._joint_ranges
        return (2 * (ordered - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1).astype(np.float32)


class ActionChunkWrapper(gym.Wrapper):
    """Expose normalized chunks, raw-frame traces, and returned action context."""

    def __init__(self, env, horizon, obs_stride, *, latency_env=None, context_delay=None):
        super().__init__(env)
        self.horizon = horizon
        self.native_action_space = env.action_space
        self.latency_env = latency_env
        self.context_delay = context_delay
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (horizon * env.action_space.shape[0],), dtype=np.float32,
        )
        if horizon > 1:
            spaces = (
                dict(env.observation_space.spaces)
                if isinstance(env.observation_space, gym.spaces.Dict) else {"obs": env.observation_space}
            )
            spaces.update(
                replay_obs=gym.spaces.Box(-np.inf, np.inf, (obs_stride + 1, *spaces["obs"].shape), dtype=np.float32),
                replay_rewards=gym.spaces.Box(-np.inf, np.inf, (obs_stride,), dtype=np.float32),
                replay_clock=gym.spaces.Box(-1, np.iinfo(np.int64).max, (obs_stride, 7), dtype=np.int64),
                replay_length=gym.spaces.Box(0, obs_stride, (), dtype=np.int64),
            )
            if context_delay is not None:
                spaces.update(
                    scheduled_chunk=gym.spaces.Box(-1.0, 1.0, (horizon, env.action_space.shape[0]), dtype=np.float32),
                    scheduled_index=gym.spaces.Box(0.0, 1.0, (horizon,), dtype=np.float32),
                )
            self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.horizon == 1:
            return observation, info
        trace = {
            name: np.zeros(self.observation_space[name].shape, dtype=self.observation_space[name].dtype)
            for name in ("replay_obs", "replay_rewards", "replay_clock", "replay_length")
        }
        return self._observation(observation, trace), info

    def _observation(self, observation, trace):
        physical = observation if isinstance(observation, dict) else {"obs": observation}
        result = {**physical, **trace}
        if self.context_delay is not None:
            chunk = np.zeros((self.horizon, self.native_action_space.shape[0]), dtype=np.float32)
            indices = np.zeros(self.horizon, dtype=np.float32)
            latency = self.latency_env
            event, index = latency.action_scheduler.scheduled_chunk_event(
                latency.env_step_idx, latency.clock.step_to_time_ms(latency.env_step_idx),
            )
            if event is not None:
                if latency.profile_sampler is not None:
                    # Returned commands are known, but their replacement time is not.
                    start, end = index, self.horizon
                else:
                    start, end = chunk_context_window(
                        latency.env_step_idx, event.obs_env_step, self.context_delay,
                        [pending.obs_env_step for pending in latency.result_timeline.get_pending_events()],
                    )
                for head in np.unique(np.minimum(np.arange(start, end), self.horizon - 1)):
                    chunk[head] = self.env.reverse_action(event.action_chunk[head])
                indices[index] = 1.0
            result.update(scheduled_chunk=chunk, scheduled_index=indices)
        return result

    def step(self, action):
        action = np.asarray(action).reshape(self.horizon, self.native_action_space.shape[0])
        action = scale_box_action(action, self.native_action_space.low, self.native_action_space.high)
        observation, reward, terminated, truncated, info = self.env.step(action[0] if self.horizon == 1 else action)
        if self.horizon > 1:
            observation = self._observation(observation, info["replay_trace"])
        return observation, reward, terminated, truncated, info


def make_gymnasium_sf_env(
    _full_env_name: str,
    cfg: Any = None,
    env_config: Any = None,
    render_mode: Optional[str] = None,
):
    del render_mode
    force_sample_factory_raw_frame_step_cfg(cfg)
    env_config = _sample_factory_env_config(env_config)
    if cfg.simulator == "gpu":
        ensure_sample_factory_episode_extra_stats_installed(env_name=SF_ENV_NAME)
        ensure_sample_factory_raw_reward_console_stats_installed(env_name=SF_ENV_NAME)
        ensure_sample_factory_gpu_sampling_compat_installed(env_name=SF_ENV_NAME)
        env_type = _GPU_ATARI_ENVS[cfg.gym_task_name]
        return env_type(
            cfg=cfg,
            env_config=env_config,
            gpu_env_device=cfg.gpu_env_device,
            gpu_env_batch_size=cfg.gpu_env_batch_size,
        )
    for module_name in json.loads(cfg.gym_registration_imports_json):
        import_module(module_name)

    make_kwargs = _gymnasium_make_kwargs(cfg)
    raw_g1_requests = (
        "control_mode" in make_kwargs and make_kwargs["control_mode"] == "raw_body_target"
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
        env_id=int(env_config.env_id),
    )
    latency_seed = sample_factory_latency_seed_from_flat_config(
        cfg,
        fallback_seed=wrapper_seed,
        env_id=int(env_config.env_id),
    )
    latency_params = sample_factory_latency_params_from_flat_config(cfg, env_fps=clock.env_fps)
    noop_action = (
        cfg.gym_noop_action_id
        if action_space_contract["type"] == "discrete"
        else _gymnasium_noop_action(cfg)
    )
    if raw_g1_requests:
        noop_action = env.unwrapped.g1_absolute_request_from_raw_action(
            np.asarray(noop_action, dtype=np.float32)
        )
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
        action_history_decisions=cfg.action_history_decisions,
        profile_seed_by_reset=cfg.rollout_profile_seed_by_reset,
        hold_last_chunk_action=cfg.hold_last_chunk_action,
        release_results_at_observation_boundary=cfg.algo == "PPO",
        noop_action=noop_action,
    )
    latency_env = env
    if raw_g1_requests:
        env = G1RawRequestWrapper(env)
    if cfg.algo == "FAST_TD3":
        env = ActionChunkWrapper(env, cfg.fasttd3_action_chunk_horizon, clock.obs_stride_raw_frames)
    elif cfg.algo == "PPO":
        env = ActionChunkWrapper(env, cfg.ppo["chunk_horizon"], clock.obs_stride_raw_frames,
                                 latency_env=latency_env, context_delay=latency_params.fixed_latency_raw_frames)
    action_labels = (
        {
            action_id: name
            for action_id, name in enumerate(_gymnasium_action_labels(cfg))
        }
        if action_space_contract["type"] == "discrete"
        else {}
    )
    env = EpisodeMetricsWrapper(
        env,
        frames_per_decision=clock.obs_stride_raw_frames,
        metrics_path=str(cfg.episode_metrics_path),
        worker_index=int(env_config.worker_index),
        vector_index=int(env_config.vector_index),
        env_id=int(env_config.env_id),
        issued_action_labels=action_labels,
    )
    return EpisodeEnv(env, cfg.ppo) if cfg.algo == "PPO" else env


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
    parser.add_argument("--action-history-decisions", dest="action_history_decisions", type=int, default=None)
    parser.add_argument("--rollout-profile-seed-by-reset", type=str2bool, default=False,
                        help="Use per-reset temporal streams for reproducible offline probe/replay.")
    parser.add_argument("--simulator", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpu-env-device", dest="gpu_env_device", type=str, default="auto")
    parser.add_argument("--gpu-env-batch-size", dest="gpu_env_batch_size", type=int, default=1)
    parser.add_argument("--noop_max", type=int)
    parser.add_argument("--terminal_on_life_loss", type=str2bool)
    parser.add_argument("--clip_reward", type=str2bool)
    parser.add_argument("--max_episode_steps", type=int)
    parser.add_argument(
        "--export-env-raw-rgb-frames",
        dest="export_env_raw_rgb_frames",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    add_sample_factory_latency_args(parser)
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
    parser.add_argument("--mcppo", dest="ppo", type=json.loads)


SAMPLE_FACTORY_CONFIG_PARSER = SampleFactoryConfigParser(
    configure_parser=configure_sample_factory_parser,
    finalize_cfg=force_sample_factory_raw_frame_step_cfg,
)


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    env_cfg = config["env"]
    scheduler_cfg = config["scheduler"]
    args = build_common_sf_args(config, sf_env_name=SF_ENV_NAME)
    if "algo" in config["backend"] and config["backend"]["algo"] == "PPO":
        append_cli_arg(args, "--mcppo", json.dumps(config["training"]["ppo"]))
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
    if "action_history_decisions" in env_cfg:
        append_cli_arg(args, "--action-history-decisions", env_cfg["action_history_decisions"])
    if "simulator" in env_cfg:
        append_cli_arg(args, "--simulator", env_cfg["simulator"])
    if "simulator" in env_cfg and env_cfg["simulator"] == "gpu":
        make_kwargs = env_cfg["make_kwargs"]
        append_cli_arg(args, "--gpu-env-device", env_cfg["gpu_env_device"])
        append_cli_arg(args, "--noop_max", make_kwargs["noop_max"])
        append_cli_arg(args, "--terminal_on_life_loss", env_cfg["terminal_on_life_loss"])
        append_cli_arg(args, "--clip_reward", env_cfg["clip_reward"])
        append_cli_arg(
            args,
            "--max_episode_steps",
            make_kwargs["base_make_kwargs"]["max_num_frames_per_episode"],
        )
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
    if cfg.algo == "PPO":
        settings = cfg.ppo
        cfg.num_workers = settings["workers"]
        cfg.num_envs_per_worker = cfg.worker_num_splits = cfg.policy_workers_per_policy = 1
        cfg.batched_sampling = cfg.async_rl = cfg.use_rnn = False
        cfg.normalize_input = cfg.normalize_returns = False
        cfg.rollout = (settings["episode_frames"] + settings["decision_frames"] - 1) // settings["decision_frames"]
        cfg.batch_size = settings["episodes_per_round"] * cfg.rollout
        cfg.num_batches_per_epoch = cfg.recurrence = 1
        cfg.save_every_sec = cfg.train_for_env_steps = 10**12
        cfg.keep_checkpoints = settings["rounds"] + 2
        cfg.decorrelate_envs_on_one_worker = False
        cfg.decorrelate_experience_max_seconds = 0
        cfg.set_workers_cpu_affinity = False
        cfg.episode_metrics_path = ""
        cfg.wandb_user = config["logging"]["wandb_entity"]
    return cfg


def run_from_config(config: dict[str, Any]) -> int:
    from sample_factory.algo.utils.misc import ExperimentStatus

    register_gymnasium_envs()
    cfg = build_train_cfg_from_config(config)
    apply_runtime_logging_config(cfg, config["logging"])
    seed_everything(int(cfg.seed))
    patch_sample_factory_wandb()

    if cfg.algo == "PPO":
        # Evaluation imports the learner only when this optional SF backend runs.
        from sample_factory.algo.mc_ppo.learner import evaluate_checkpoint

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        torch.set_num_threads(1)
        if cfg.mode == "eval":
            evaluate_checkpoint(cfg)
            return 0
        status = run_sample_factory_training(cfg, config)
        if status == ExperimentStatus.SUCCESS and not cfg.benchmark:
            result = evaluate_checkpoint(cfg, final=True)
            if cfg.with_wandb:
                # W&B is optional; the training runner owns this run.
                import wandb

                wandb.run.summary["heldout"] = {
                    kind: {key: value for key, value in result[kind].items() if key != "episodes"}
                    for kind in ("baseline", "best", "latest")
                }
        return status

    metrics_stop_event = None
    metrics_thread = None
    if cfg.mode == "train":
        if cfg.simulator == "gpu":
            if cfg.episode_metrics_path:
                unlink_stale_episode_metrics_file(cfg.episode_metrics_path, cfg.restart_behavior)
        else:
            metrics_stop_event, metrics_thread = start_episode_metrics_aggregator(cfg)

    status = ExperimentStatus.SUCCESS
    try:
        if cfg.mode == "train":
            status = run_sample_factory_training(cfg, config)
    finally:
        stop_episode_metrics_aggregator(metrics_stop_event, metrics_thread)
    return int(status) if isinstance(status, int) else 0
