"""Sample Factory runtime construction for teacher rollout export."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from latency_bench.latency.config import latency_type_from_config
from latency_bench.integrations.sample_factory.runtime import (
    load_sample_factory_policy_state_dict,
    resolve_torch_device,
)
from training.common.config_adapter import import_integration_module
from training.common.utils import maybe_set_cli_override


def _register_components(module: Any, env_name: str) -> None:
    """Register Sample Factory integration components for an environment.

    Args:
        module: Imported training integration module.
        env_name: Canonical benchmark environment name.
    """
    if env_name == "deadly_corridor":
        module.register_deadly_corridor_components()
    elif env_name == "demon_attack":
        module.register_demon_attack_envs()
    else:
        module.register_flappy_envs()


def _apply_latency_to_flat_cfg(cfg: Any, latency_cfg: Mapping[str, Any]) -> None:
    """Apply nested latency config values to a Sample Factory cfg object.

    Args:
        cfg: Mutable Sample Factory config object.
        latency_cfg: Nested latency config mapping.
    """
    maybe_set_cli_override(cfg, "latency_type", latency_type_from_config(latency_cfg))
    field_map = {
        "fixed_latency_ms": "fixed_latency_ms",
        "mean_latency_ms": "mean_latency_ms",
        "std_latency_ms": "std_latency_ms",
        "min_latency_ms": "min_latency_ms",
        "max_latency_ms": "max_latency_ms",
        "seed": "latency_seed",
        "add_latency_info": "add_latency_info",
        "max_pending_actions": "max_pending_actions",
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

    env_name = str(config["env_name"])
    module = import_integration_module({"env": {"name": env_name}})
    _register_components(module, env_name)
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
    if config["device_override"] is not None:
        argv.extend(["--device", str(config["device_override"])])
    if "load_checkpoint_kind" in flat_cfg:
        argv.extend(["--load_checkpoint_kind", str(flat_cfg["load_checkpoint_kind"])])
    cfg = module.parse_cfg(argv)
    cfg = load_from_checkpoint(cfg)
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

    device = torch.device(resolve_torch_device(cfg.device))
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
    checkpoint = load_sample_factory_policy_state_dict(cfg, actor_critic, device, checkpoint_path)
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
