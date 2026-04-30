from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from training.sample_factory_common.common import list_to_csv


ENV_INTEGRATIONS = {
    "deadly_corridor": "training.deadly_corridor_sf.integration",
    "demon_attack": "training.demon_attack_sf.integration",
    "flappy": "training.flappy_sf.integration",
}


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    validate_training_config(config)
    return config


def apply_training_overrides(
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    train_dir: str | Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    backend_cfg = updated.setdefault("backend", {})
    experiment_cfg = updated.setdefault("experiment", {})
    if run_name is not None:
        experiment_cfg["name"] = str(run_name)
    if train_dir is not None:
        backend_cfg["train_dir"] = str(train_dir)
    if mode is not None:
        backend_cfg["run_mode"] = str(mode)
    return updated


def validate_training_config(config: dict[str, Any]) -> None:
    backend_cfg = config.get("backend")
    if not isinstance(backend_cfg, dict):
        raise ValueError("backend must be a mapping for Sample Factory training")
    backend_type = str(backend_cfg.get("type", "")).strip().lower()
    if backend_type != "sample_factory":
        raise ValueError("backend.type must be 'sample_factory' for training.train")
    training_cfg = config.get("training")
    if not isinstance(training_cfg, dict):
        raise ValueError("training must be a mapping for Sample Factory training")
    evaluation_cfg = config.get("evaluation")
    if evaluation_cfg is not None and not isinstance(evaluation_cfg, dict):
        raise ValueError("evaluation must be a mapping when provided")
    env_name = str(config.get("env", {}).get("name", "")).strip().lower()
    if env_name not in ENV_INTEGRATIONS:
        raise ValueError(
            f"Unsupported env.name for Sample Factory backend: {env_name}. "
            f"Supported envs: {sorted(ENV_INTEGRATIONS)}"
        )


def import_integration_module(config: dict[str, Any]):
    env_name = str(config["env"]["name"]).strip().lower()
    module_name = ENV_INTEGRATIONS[env_name]
    return importlib.import_module(module_name)


def append_cli_arg(args: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        args.extend([flag, "True" if value else "False"])
        return
    if isinstance(value, (list, tuple)):
        args.extend([flag, list_to_csv(value)])
        return
    args.extend([flag, str(value)])


def build_common_sf_args(config: dict[str, Any], *, sf_env_name: str) -> list[str]:
    backend_cfg = config["backend"]
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    latency_cfg = config.get("latency", {})
    evaluation_cfg = config.get("evaluation", {})
    logging_cfg = config.get("logging", {})

    args: list[str] = []
    append_cli_arg(args, "--mode", backend_cfg.get("run_mode", "train"))
    append_cli_arg(args, "--algo", backend_cfg.get("algo", "APPO"))
    append_cli_arg(args, "--env", sf_env_name)
    append_cli_arg(args, "--experiment", experiment_cfg["name"])
    append_cli_arg(args, "--train_dir", backend_cfg.get("train_dir"))
    append_cli_arg(args, "--restart_behavior", backend_cfg.get("restart_behavior", "overwrite"))
    append_cli_arg(args, "--device", backend_cfg.get("device", "cpu"))
    append_cli_arg(args, "--seed", experiment_cfg.get("seed", 0))

    scalar_fields = (
        "train_for_env_steps",
        "num_workers",
        "num_envs_per_worker",
        "num_policies",
        "batch_size",
        "rollout",
        "recurrence",
        "num_epochs",
        "num_batches_per_epoch",
        "worker_num_splits",
        "num_batches_to_accumulate",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "ppo_clip_ratio",
        "ppo_clip_value",
        "exploration_loss",
        "exploration_loss_coeff",
        "value_loss_coeff",
        "max_grad_norm",
        "rnn_type",
        "rnn_size",
        "save_every_sec",
        "keep_checkpoints",
        "save_milestones_sec",
        "save_best_every_sec",
        "save_best_after",
    )
    bool_fields = (
        "async_rl",
        "use_rnn",
        "normalize_returns",
        "normalize_input",
        "with_wandb",
    )

    for field in scalar_fields:
        append_cli_arg(args, f"--{field}", training_cfg.get(field))
    for field in bool_fields:
        if field in training_cfg:
            append_cli_arg(args, f"--{field}", training_cfg.get(field))

    if "train_values" in latency_cfg:
        append_cli_arg(args, "--queue-latency-values", latency_cfg.get("train_values"))
    append_cli_arg(args, "--latency-mode", latency_cfg.get("train_mode", "episode_fixed"))
    append_cli_arg(args, "--latency-sampling-mode", latency_cfg.get("train_sampling_mode", "uniform_all"))
    append_cli_arg(args, "--episodes-per-stage", latency_cfg.get("episodes_per_stage"))
    append_cli_arg(args, "--randomize-within-stage", latency_cfg.get("randomize_within_stage", False))
    append_cli_arg(args, "--within-episode-change-interval", latency_cfg.get("within_episode_change_interval"))
    append_cli_arg(args, "--within-episode-change-prob", latency_cfg.get("within_episode_change_prob"))
    append_cli_arg(args, "--add-latency-info", latency_cfg.get("add_latency_info", True))

    append_cli_arg(args, "--eval-episodes", evaluation_cfg.get("eval_episodes"))
    append_cli_arg(args, "--eval-queue-latency-values", evaluation_cfg.get("eval_latency_values"))
    append_cli_arg(args, "--eval-max-steps", evaluation_cfg.get("eval_max_steps"))
    append_cli_arg(args, "--eval-deterministic", evaluation_cfg.get("eval_deterministic", True))
    if evaluation_cfg.get("eval_raw_reward"):
        args.append("--eval-raw-reward")

    if logging_cfg.get("wandb_project"):
        append_cli_arg(args, "--with_wandb", True)
        append_cli_arg(args, "--wandb_project", logging_cfg.get("wandb_project"))
        append_cli_arg(args, "--wandb_group", logging_cfg.get("wandb_group"))
        append_cli_arg(args, "--wandb_job_type", logging_cfg.get("wandb_job_type"))
        append_cli_arg(args, "--wandb_tags", logging_cfg.get("wandb_tags"))

    extra_args = backend_cfg.get("extra_args", ())
    if extra_args:
        args.extend(str(item) for item in extra_args)
    return args
