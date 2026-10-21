"""Sample Factory runtime construction for teacher rollout export."""

from __future__ import annotations

import contextlib
from importlib import import_module
from training.common.config_adapter import ENV_INTEGRATIONS
from copy import deepcopy
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from gymnasium.spaces import Discrete

from latency_bench.integrations.sample_factory.teacher_drivers import (
    BatchedTeacherRolloutDriver,
    RolloutStepResult,
    SerialTeacherRolloutDriver,
    batch_observations,
)
from latency_bench.envs.atari import is_true_atari_episode_end
from latency_bench.integrations.sample_factory.actions import (
    deterministic_actions_for_space,
    normalize_actions_for_space,
    split_batch_actions,
)
from latency_bench.integrations.sample_factory.runtime import (
    load_sample_factory_policy_state_dict,
    resolve_sample_factory_device,
    resolve_torch_device,
)
from latency_bench.latency.config import latency_type_from_config
from training.asterix_sf.batched_gpu_env import AsterixBatchedGpuEnv
from training.atlantis_sf.batched_gpu_env import AtlantisBatchedGpuEnv
from training.common.utils import maybe_set_cli_override


_GYMNASIUM_ATARI_GPU_ENVS = {
    "asterix": AsterixBatchedGpuEnv,
    "atlantis": AtlantisBatchedGpuEnv,
}

REGISTRATION_METHOD_BY_INTEGRATION = {
    "deadly_corridor": "register_deadly_corridor_components",
    "demon_attack": "register_demon_attack_envs",
    "flappy": "register_flappy_envs",
    "gymnasium": "register_gymnasium_envs",
}

GYMNASIUM_CHECKPOINT_COMMON_ARG_FIELDS = (
    ("gym_task_name", "--gym-task-name"),
    ("gym_env_id", "--gym-env-id"),
    ("gym_make_kwargs_json", "--gym-make-kwargs-json"),
    ("gym_registration_imports_json", "--gym-registration-imports-json"),
    ("gym_base_prompt", "--gym-base-prompt"),
    ("env_fps", "--env-fps"),
    ("obs_fps", "--obs-fps"),
    ("frame_stack", "--frame-stack"),
    ("hold_policy", "--hold-policy"),
    ("ordering_policy", "--ordering-policy"),
)

GYMNASIUM_CHECKPOINT_DISCRETE_ARG_FIELDS = (
    ("gym_action_labels_json", "--gym-action-labels-json"),
    ("gym_action_values_json", "--gym-action-values-json"),
    ("gym_noop_action_id", "--gym-noop-action-id"),
)


def _gymnasium_checkpoint_args(flat_cfg: Mapping[str, Any]) -> list[str]:
    action_space_json = flat_cfg["gym_action_space_json"]
    noop_action_json = flat_cfg["gym_noop_action_json"]

    argv: list[str] = []
    for field_name, flag in GYMNASIUM_CHECKPOINT_COMMON_ARG_FIELDS:
        argv.extend([flag, str(flat_cfg[field_name])])
    argv.extend(["--gym-action-space-json", str(action_space_json)])
    argv.extend(["--gym-noop-action-json", str(noop_action_json)])
    if json.loads(action_space_json)["type"] == "discrete":
        for field_name, flag in GYMNASIUM_CHECKPOINT_DISCRETE_ARG_FIELDS:
            argv.extend([flag, str(flat_cfg[field_name])])
    return argv


def _register_components(module: Any, integration_name: str) -> None:
    """Register Sample Factory integration components for an environment.

    Args:
        module: Imported training integration module.
        integration_name: Training integration selected by the checkpoint.
    """
    method_name = REGISTRATION_METHOD_BY_INTEGRATION[integration_name]
    getattr(module, method_name)()


def _apply_latency_to_flat_cfg(cfg: Any, latency_cfg: Mapping[str, Any]) -> None:
    """Apply nested latency config values to a Sample Factory cfg object.

    Args:
        cfg: Mutable Sample Factory config object.
        latency_cfg: Nested latency config mapping.
    """
    maybe_set_cli_override(
        cfg,
        "latency_type",
        "iid" if latency_cfg["method"] == "iid" else latency_type_from_config(latency_cfg),
    )
    field_map = {
        "fixed_latency_ms": "fixed_latency_ms",
        "mean_latency_ms": "mean_latency_ms",
        "std_latency_ms": "std_latency_ms",
        "min_latency_ms": "min_latency_ms",
        "max_latency_ms": "max_latency_ms",
        "seed": "latency_seed",
        "add_latency_info": "add_latency_info",
        "max_pending_actions": "max_pending_actions",
        "profile_path": "latency_profile_path",
        "profile_worker_slot": "latency_profile_worker_slot",
    }
    for source_key, target_key in field_map.items():
        if source_key in latency_cfg:
            maybe_set_cli_override(cfg, target_key, latency_cfg[source_key])
    maybe_set_cli_override(cfg, "add_latency_info", True)


def build_sample_factory_rollout_cfg(config: dict[str, Any]):
    """Build the Sample Factory config used for teacher rollout.

    Args:
        config: Rollout descriptor loaded from a checkpoint.

    Returns:
        Imported integration module and initialized Sample Factory cfg.
    """
    from sample_factory.cfg.arguments import load_from_checkpoint

    integration_name = config["integration_name"]

    module = import_module(ENV_INTEGRATIONS[integration_name])
    _register_components(module, integration_name)
    flat_cfg = config["flat_cfg"]
    argv = [
        "--mode",
        "eval",
        "--algo",
        str(flat_cfg["algo"]),
        "--env",
        str(flat_cfg["env"]),
        "--experiment",
        Path(str(config["checkpoint_experiment_dir"])).name,
        "--train_dir",
        str(Path(str(config["checkpoint_experiment_dir"])).parent),
    ]
    if integration_name == "gymnasium":
        argv.extend(_gymnasium_checkpoint_args(flat_cfg))
    if config["device_override"] is not None:
        argv.extend(["--device", resolve_sample_factory_device(str(config["device_override"]))])
    if "load_checkpoint_kind" in flat_cfg:
        argv.extend(["--load_checkpoint_kind", str(flat_cfg["load_checkpoint_kind"])])
    cfg = module.SAMPLE_FACTORY_CONFIG_PARSER.parse_eval(argv)
    cfg = load_from_checkpoint(cfg)
    for key, value in config.get("runtime_overrides", {}).items():
        maybe_set_cli_override(cfg, key, value)
    if config["latency_override"] is not None:
        _apply_latency_to_flat_cfg(cfg, config["latency_override"])
    maybe_set_cli_override(cfg, "add_latency_info", True)
    return module, cfg


def build_env_and_policy(module: Any, cfg: Any, config: Mapping[str, Any], num_envs: int):
    """Create rollout environments and load the teacher policy.

    Args:
        module: Imported training integration module.
        cfg: Sample Factory cfg.
        config: Rollout descriptor loaded from a checkpoint.
        num_envs: Number of rollout environments to create.

    Returns:
        Runtime mapping with envs, policy, device, and Sample Factory helpers.
    """
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    device = torch.device(resolve_torch_device(
        cfg.device if config["device_override"] is None else config["device_override"]
    ))
    envs = [
        make_env_func_batched(
            cfg,
            env_config=AttrDict(worker_index=0, vector_index=int(env_id), env_id=int(env_id)),
            render_mode=None,
        )
        for env_id in range(int(num_envs))
    ]
    env_info = extract_env_info(envs[0], cfg)
    actor_critic = create_actor_critic(cfg, envs[0].observation_space, envs[0].action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    checkpoint_path = config["checkpoint_path"]
    checkpoint, _ = load_sample_factory_policy_state_dict(
        cfg,
        actor_critic,
        device,
        checkpoint_path,
    )
    return {
        "envs": envs,
        "env_info": env_info,
        "actor_critic": actor_critic,
        "device": device,
        "preprocess_actions": preprocess_actions,
        "argmax_actions": argmax_actions,
        "prepare_and_normalize_obs": prepare_and_normalize_obs,
        "unsqueeze_tensor": unsqueeze_tensor,
        "rnn_size": int(get_rnn_size(cfg)),
        "checkpoint": checkpoint,
    }


@contextlib.contextmanager
def sample_factory_init_log(output_dir: Path):
    """Redirect Sample Factory initialization logs to a file.

    Args:
        output_dir: Rollout export directory.

    Yields:
        Context with stdout/stderr and Sample Factory logger redirected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sample_factory_init.log"
    with log_path.open("a", encoding="utf-8") as handle:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        sf_logger = logging.getLogger("rl")
        previous_handlers = list(sf_logger.handlers)
        previous_propagate = sf_logger.propagate
        log_handler = logging.StreamHandler(handle)
        sf_logger.handlers = [log_handler]
        sf_logger.propagate = False
        try:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            handle.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            log_handler.close()
            sf_logger.handlers = previous_handlers
            sf_logger.propagate = previous_propagate


def step_teacher_policy(
    runtime: Mapping[str, Any],
    driver: SerialTeacherRolloutDriver | BatchedTeacherRolloutDriver,
    *,
    active_slots: list[int],
    active_obs: list[Any | None],
    rnn_states: torch.Tensor,
    deterministic: bool,
    preserve_leading_batch: bool = False,
) -> tuple[list[Any], torch.Tensor, dict[int, Any]]:
    """Run one teacher policy/env step for active rollout slots.

    Args:
        runtime: Runtime mapping with policy, env info, and SF helper functions.
        driver: Driver that abstracts serial vs batched env stepping.
        active_slots: Slot ids that currently hold rollout attempts.
        active_obs: Latest observations for every driver slot.
        rnn_states: Recurrent state tensor aligned with driver slots.
        deterministic: Whether to use deterministic teacher actions.

    Returns:
        Issued native commands, new RNN states for the active batch, and step results.
    """
    actor_critic = runtime["actor_critic"]
    device = runtime["device"]
    prepare_and_normalize_obs = runtime["prepare_and_normalize_obs"]
    argmax_actions = runtime["argmax_actions"]
    preprocess_actions = runtime["preprocess_actions"]
    unsqueeze_tensor = runtime["unsqueeze_tensor"]
    obs_batch = batch_observations(
        [active_obs[slot] for slot in active_slots],
        preserve_leading_batch=preserve_leading_batch,
    )
    batched = isinstance(driver, BatchedTeacherRolloutDriver)
    policy_rnn_states = rnn_states[active_slots]
    if batched:
        # GPU kernels can change near-tied argmax actions with the batch shape.
        # Keep the declared slots through probe, replay, and inactive episodes.
        padded = {}
        for key, value in obs_batch.items():
            padded[key] = np.zeros((driver.slot_count, *value.shape[1:]), dtype=value.dtype)
            padded[key][active_slots] = value
        obs_batch = padded
        policy_rnn_states = rnn_states
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs_batch)
    with torch.no_grad():
        policy_outputs = actor_critic(normalized_obs, policy_rnn_states)
        actions = policy_outputs["actions"]
        if deterministic:
            actions = deterministic_actions_for_space(
                actor_critic.action_distribution(),
                action_space=driver.action_space,
                argmax_actions=argmax_actions,
                device=device,
            )
        new_rnn_states = policy_outputs["new_rnn_states"]
        if batched:
            actions = actions[active_slots]
            new_rnn_states = new_rnn_states[active_slots]
        actions = normalize_actions_for_space(
            actions,
            batch_size=len(active_slots),
            action_space=driver.action_space,
            device=device,
            unsqueeze_tensor=unsqueeze_tensor,
        )
        processed_actions = preprocess_actions(runtime["env_info"], actions)
        raw_actions = split_batch_actions(
            processed_actions,
            len(active_slots),
            action_space=driver.action_space,
        )
    # Discrete commands are immutable; vector commands need ownership across env.step.
    issued_actions = raw_actions if isinstance(driver.action_space, Discrete) else deepcopy(raw_actions)
    step_results = driver.step(active_slots, raw_actions)
    return issued_actions, new_rnn_states, step_results


def rollout_step_flags(
    env_name: str,
    step_result: RolloutStepResult,
    *,
    next_raw_frame: int,
    max_steps_per_episode: int,
) -> tuple[bool, bool]:
    raw_done = step_result.terminated or step_result.truncated
    life_loss_boundary = False
    true_episode_end = raw_done
    if env_name == "demon_attack":
        true_episode_end = is_true_atari_episode_end(
            step_result.terminated,
            step_result.truncated,
            step_result.info,
        )
        life_loss_boundary = raw_done and not true_episode_end
    if next_raw_frame >= max_steps_per_episode:
        return life_loss_boundary, True
    return life_loss_boundary, true_episode_end


def build_rollout_driver_runtime(
    config: dict[str, Any],
    *,
    output_dir: Path,
    num_envs: int,
    export_env_raw_rgb_frames: bool,
):
    """Build Sample Factory cfg, runtime, and driver for one rollout phase.

    Args:
        config: Rollout descriptor loaded from a checkpoint.
        output_dir: Rollout export directory.
        num_envs: Number of rollout envs or batched agents for this phase.
        export_env_raw_rgb_frames: Whether env info should include raw RGB frames.

    Returns:
        Sample Factory cfg, runtime mapping, driver, and checkpoint frame stack.
    """
    env_name = str(config["env_name"])
    integration_name = config["integration_name"]
    with sample_factory_init_log(output_dir):
        module, cfg = build_sample_factory_rollout_cfg(config)
        frame_stack = int(cfg.frame_stack)
        if ((integration_name in {"flappy", "demon_attack", "deadly_corridor"}
             or env_name in _GYMNASIUM_ATARI_GPU_ENVS)
                and getattr(cfg, "action_history_decisions", None) == 8
                and cfg.latency_type == "profile_sample"):
            setattr(cfg, "rollout_profile_seed_by_reset", True)
        setattr(cfg, "export_env_raw_rgb_frames", bool(export_env_raw_rgb_frames))
        if integration_name == "flappy" and cfg.use_gpu_render:
            maybe_set_cli_override(cfg, "gpu_render_batch_size", int(num_envs))
        use_gymnasium_atari_gpu_batch = (
            integration_name == "gymnasium"
            and env_name in _GYMNASIUM_ATARI_GPU_ENVS
            and str(cfg.device) == "gpu"
            and int(num_envs) > 1
        )
        use_batched_driver = (
            (integration_name == "flappy" and int(num_envs) > 1 and cfg.use_gpu_render)
            or (integration_name == "demon_attack" and cfg.simulator == "gpu")
            or use_gymnasium_atari_gpu_batch
        )
        if use_batched_driver:
            if integration_name == "demon_attack":
                maybe_set_cli_override(cfg, "gpu_env_batch_size", int(num_envs))
            runtime = build_env_and_policy(module, cfg, config, 1)
            if use_gymnasium_atari_gpu_batch:
                make_kwargs = json.loads(config["flat_cfg"]["gym_make_kwargs_json"])
                base_make_kwargs = make_kwargs["base_make_kwargs"]
                maybe_set_cli_override(cfg, "noop_max", make_kwargs["noop_max"])
                maybe_set_cli_override(cfg, "terminal_on_life_loss", False)
                maybe_set_cli_override(cfg, "clip_reward", False)
                maybe_set_cli_override(
                    cfg,
                    "max_episode_steps",
                    base_make_kwargs["max_num_frames_per_episode"],
                )
                maybe_set_cli_override(cfg, "gpu_env_device", "cuda")
                maybe_set_cli_override(cfg, "gpu_env_batch_size", int(num_envs))
                maybe_set_cli_override(cfg, "episode_metrics_path", "")
                runtime["envs"][0].close()
                env_type = _GYMNASIUM_ATARI_GPU_ENVS[env_name]
                runtime["envs"] = [
                    env_type(
                        cfg=cfg,
                        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
                        gpu_env_device=cfg.gpu_env_device,
                        gpu_env_batch_size=cfg.gpu_env_batch_size,
                    )
                ]
        else:
            runtime = build_env_and_policy(module, cfg, config, int(num_envs))

    if use_batched_driver:
        driver = BatchedTeacherRolloutDriver(runtime["envs"][0], int(num_envs))
    else:
        if integration_name == "deadly_corridor" and (
            "export_zero_action_history" in config or getattr(cfg, "rollout_profile_seed_by_reset", False)
        ):
            from sample_factory.algo.utils.make_env import make_env_func_batched
            from sample_factory.utils.attr_dict import AttrDict

            def recreate_env(slot: int):
                return make_env_func_batched(
                    cfg,
                    env_config=AttrDict(worker_index=0, vector_index=slot, env_id=slot),
                    render_mode=None,
                )

            driver = SerialTeacherRolloutDriver(runtime["envs"], recreate_env=recreate_env)
        else:
            driver = SerialTeacherRolloutDriver(runtime["envs"])
    return cfg, runtime, driver, frame_stack
