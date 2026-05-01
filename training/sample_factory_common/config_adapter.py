from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

ENV_INTEGRATIONS = {
    "deadly_corridor": "training.deadly_corridor_sf.integration",
    "demon_attack": "training.demon_attack_sf.integration",
    "flappy": "training.flappy_sf.integration",
}


def _list_to_csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def load_training_config(path: str | Path) -> dict[str, Any]:
    config = _load_training_config_unvalidated(Path(path))
    validate_training_config(config)
    return config


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_training_config_unvalidated(path: Path) -> dict[str, Any]:
    import yaml

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    base_ref = current.get("__base__")
    if not base_ref:
        return {key: value for key, value in current.items() if key != "__base__"}

    if isinstance(base_ref, (str, Path)):
        base_paths = [base_ref]
    elif isinstance(base_ref, list):
        base_paths = base_ref
    else:
        raise ValueError("__base__ must be a path string or list of path strings")

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge_dicts(merged, _load_training_config_unvalidated(base_path))

    return _merge_dicts(merged, current)


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
    experiment_cfg = config.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise ValueError("experiment must be a mapping for Sample Factory training")
    if not str(experiment_cfg.get("name", "")).strip():
        raise ValueError("experiment.name must be a non-empty string")
    backend_cfg = config.get("backend")
    if not isinstance(backend_cfg, dict):
        raise ValueError("backend must be a mapping for Sample Factory training")
    backend_type = str(backend_cfg.get("type", "")).strip().lower()
    if backend_type != "sample_factory":
        raise ValueError("backend.type must be 'sample_factory' for training.train")
    env_cfg = config.get("env")
    if not isinstance(env_cfg, dict):
        raise ValueError("env must be a mapping for Sample Factory training")
    latency_cfg = config.get("latency")
    if not isinstance(latency_cfg, dict):
        raise ValueError("latency must be a mapping for Sample Factory training")
    training_cfg = config.get("training")
    if not isinstance(training_cfg, dict):
        raise ValueError("training must be a mapping for Sample Factory training")
    evaluation_cfg = config.get("evaluation")
    if evaluation_cfg is not None and not isinstance(evaluation_cfg, dict):
        raise ValueError("evaluation must be a mapping when provided")
    logging_cfg = config.get("logging")
    if logging_cfg is not None and not isinstance(logging_cfg, dict):
        raise ValueError("logging must be a mapping when provided")
    env_name = str(env_cfg.get("name", "")).strip().lower()
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
    if isinstance(value, str) and not value.strip():
        return
    if isinstance(value, bool):
        args.extend([flag, "True" if value else "False"])
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        args.extend([flag, _list_to_csv(value)])
        return
    args.extend([flag, str(value)])


def apply_runtime_logging_config(sf_cfg: Any, logging_cfg: dict[str, Any] | None) -> None:
    if not isinstance(logging_cfg, dict):
        return

    output_dir = logging_cfg.get("output_dir")
    if output_dir:
        sf_cfg.output_dir = str(output_dir)

    wandb_name = logging_cfg.get("wandb_name")
    if isinstance(wandb_name, str):
        wandb_name = wandb_name.strip()
    if wandb_name:
        sf_cfg.wandb_unique_id = str(wandb_name)


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
    if backend_cfg.get("train_dir") and experiment_cfg.get("name"):
        metrics_path = Path(str(backend_cfg["train_dir"])).expanduser() / str(experiment_cfg["name"]) / "episode_metrics.jsonl"
        append_cli_arg(args, "--episode_metrics_path", str(metrics_path))

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
        "stats_avg",
        "experiment_summaries_interval",
    )
    bool_fields = (
        "async_rl",
        "batched_sampling",
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
    append_cli_arg(args, "--add-latency-info", latency_cfg.get("add_latency_info", False))

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
