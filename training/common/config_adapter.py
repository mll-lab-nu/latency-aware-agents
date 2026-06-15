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


def _wandb_tags_to_cli_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_checkpoint_kind(checkpoint_kind: str) -> str:
    return str(checkpoint_kind).strip().lower()


def apply_checkpoint_kind_override(backend_cfg: dict[str, Any], checkpoint_kind: str) -> None:
    normalized = normalize_checkpoint_kind(checkpoint_kind)
    extra_args = list(backend_cfg.get("extra_args") or [])
    filtered_args: list[str] = []
    skip_next = False
    for item in extra_args:
        text = str(item)
        if skip_next:
            skip_next = False
            continue
        if text == "--load_checkpoint_kind":
            skip_next = True
            continue
        if text.startswith("--load_checkpoint_kind="):
            continue
        filtered_args.append(text)
    filtered_args.extend(["--load_checkpoint_kind", normalized])
    backend_cfg["extra_args"] = filtered_args


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


def load_training_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}

    base_ref = current.get("__base__")
    if not base_ref:
        return {key: value for key, value in current.items() if key != "__base__"}

    if isinstance(base_ref, (str, Path)):
        base_paths = [base_ref]
    else:
        base_paths = base_ref

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge_dicts(merged, load_training_config(base_path))

    return _merge_dicts(merged, current)


def apply_training_overrides(
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    train_dir: str | Path | None = None,
    mode: str | None = None,
    checkpoint_kind: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    backend_cfg = updated.setdefault("backend", {})
    experiment_cfg = updated.setdefault("experiment", {})
    logging_cfg = updated.setdefault("logging", {})
    if run_name is not None:
        experiment_cfg["name"] = str(run_name)
    if train_dir is not None:
        backend_cfg["train_dir"] = str(train_dir)
    if mode is not None:
        backend_cfg["run_mode"] = str(mode)
    if checkpoint_kind is not None:
        apply_checkpoint_kind_override(backend_cfg, checkpoint_kind)
    if output_dir is not None:
        logging_cfg["output_dir"] = str(output_dir)
    return updated


def _parse_actor_worker_gpus_config(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parts = stripped.replace(",", " ").split()
    else:
        parts = list(value)
    return [int(item) for item in parts]


def estimate_optimizer_steps_per_1m_env_steps(training_cfg: dict[str, Any]) -> float:
    batch_size = int(training_cfg.get("batch_size", 0) or 0)
    num_epochs = int(training_cfg.get("num_epochs", 0) or 0)
    return 1_000_000.0 * float(num_epochs) / float(batch_size)


def sample_factory_training_summary(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    num_workers = int(training_cfg.get("num_workers", 0) or 0)
    num_envs_per_worker = int(training_cfg.get("num_envs_per_worker", 0) or 0)
    return {
        "effective_envs": num_workers * num_envs_per_worker,
        "batch_size": int(training_cfg.get("batch_size", 0) or 0),
        "num_epochs": int(training_cfg.get("num_epochs", 0) or 0),
        "num_batches_per_epoch": int(training_cfg.get("num_batches_per_epoch", 0) or 0),
        "optimizer_steps_per_1m_env_steps": estimate_optimizer_steps_per_1m_env_steps(training_cfg),
        "async_rl": bool(training_cfg.get("async_rl", True)),
    }


def gpu_env_sample_factory_layout(training_cfg: dict[str, Any]) -> dict[str, int]:
    effective_envs_per_worker = int(training_cfg["num_envs_per_worker"])
    worker_num_splits = int(training_cfg["worker_num_splits"])
    if effective_envs_per_worker % worker_num_splits != 0:
        raise ValueError(
            "GPU env Sample Factory layout requires "
            f"num_envs_per_worker={effective_envs_per_worker} to be divisible by "
            f"worker_num_splits={worker_num_splits}"
        )
    return {
        "sf_num_envs_per_worker": worker_num_splits,
        "worker_num_splits": worker_num_splits,
        "gpu_env_batch_size": effective_envs_per_worker // worker_num_splits,
    }


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


def append_cli_multi_value_arg(args: list[str], flag: str, values: Any) -> None:
    parsed = _parse_actor_worker_gpus_config(values)
    if parsed is None:
        return
    args.append(flag)
    args.extend(str(item) for item in parsed)


def apply_runtime_logging_config(sf_cfg: Any, logging_cfg: dict[str, Any]) -> None:
    output_dir = logging_cfg.get("output_dir")
    if output_dir:
        sf_cfg.output_dir = str(output_dir)

    wandb_name = logging_cfg.get("wandb_name")
    if isinstance(wandb_name, str):
        wandb_name = wandb_name.strip()
    if wandb_name:
        sf_cfg.wandb_unique_id = str(wandb_name)


def build_common_sf_args(config: dict[str, Any], *, sf_env_name: str) -> list[str]:
    from latency_bench.policy.sample_factory_common import resolve_sample_factory_device

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
    append_cli_arg(args, "--device", resolve_sample_factory_device(backend_cfg.get("device", "cpu")))
    env_cfg = config.get("env", {})
    use_gpu_render = env_cfg.get("use_gpu_render", False)
    gpu_render_device = str(env_cfg.get("gpu_render_device", "auto") or "auto").strip().lower()
    simulator = env_cfg.get("simulator")
    use_gpu_env = simulator == "gpu"
    gpu_env_device = str(env_cfg.get("gpu_env_device", "auto") or "auto").strip().lower()
    actor_worker_gpus = backend_cfg.get("actor_worker_gpus")
    if use_gpu_render and gpu_render_device != "cpu" and actor_worker_gpus is None:
        actor_worker_gpus = [0]
    if use_gpu_env and gpu_env_device != "cpu" and actor_worker_gpus is None:
        actor_worker_gpus = [0]
    append_cli_multi_value_arg(args, "--actor_worker_gpus", actor_worker_gpus)
    if use_gpu_render:
        append_cli_arg(args, "--env_gpu_observations", gpu_render_device != "cpu")
        append_cli_arg(args, "--env_gpu_actions", env_cfg["simulator"] == "gpu" and gpu_render_device != "cpu")
        original_num_envs_per_worker = int(training_cfg["num_envs_per_worker"])
        append_cli_arg(args, "--gpu-render-batch-size", original_num_envs_per_worker)
    gpu_env_layout = gpu_env_sample_factory_layout(training_cfg) if use_gpu_env and not use_gpu_render else None
    if use_gpu_env and not use_gpu_render:
        append_cli_arg(args, "--env_gpu_observations", gpu_env_device != "cpu")
        append_cli_arg(args, "--env_gpu_actions", gpu_env_device != "cpu")
        # GPU batched-env parallelism is driven by num_envs_per_worker (mirrors gpu-render-batch-size).
        append_cli_arg(args, "--gpu-env-batch-size", gpu_env_layout["gpu_env_batch_size"])
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
        "max_policy_lag",
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
        value = training_cfg.get(field)
        if use_gpu_env and not use_gpu_render and field == "num_envs_per_worker":
            value = gpu_env_layout["sf_num_envs_per_worker"]
        elif use_gpu_env and not use_gpu_render and field == "worker_num_splits":
            value = gpu_env_layout["worker_num_splits"]
        elif use_gpu_render and field == "num_envs_per_worker":
            value = 1
        elif use_gpu_render and field == "worker_num_splits" and value is None:
            value = 1
        append_cli_arg(args, f"--{field}", value)
    if use_gpu_render or use_gpu_env:
        append_cli_arg(args, "--batched_sampling", True)
    for field in bool_fields:
        if use_gpu_render and field == "batched_sampling":
            continue
        if field in training_cfg:
            append_cli_arg(args, f"--{field}", training_cfg.get(field))

    append_cli_arg(args, "--latency-type", latency_cfg.get("type"))
    append_cli_arg(args, "--fixed-latency-ms", latency_cfg.get("fixed_latency_ms"))
    append_cli_arg(args, "--mean-latency-ms", latency_cfg.get("mean_latency_ms"))
    append_cli_arg(args, "--std-latency-ms", latency_cfg.get("std_latency_ms"))
    append_cli_arg(args, "--min-latency-ms", latency_cfg.get("min_latency_ms"))
    append_cli_arg(args, "--max-latency-ms", latency_cfg.get("max_latency_ms"))
    append_cli_arg(args, "--latency-seed", latency_cfg.get("seed"))
    append_cli_arg(args, "--add-latency-info", latency_cfg.get("add_latency_info", False))
    append_cli_arg(args, "--max-pending-actions", latency_cfg.get("max_pending_actions"))

    append_cli_arg(args, "--eval-episodes", evaluation_cfg.get("eval_episodes"))
    append_cli_arg(args, "--eval-parallel-envs", evaluation_cfg.get("eval_parallel_envs"))
    append_cli_arg(args, "--eval-latency-raw-frame-values", evaluation_cfg.get("eval_latency_values"))
    append_cli_arg(args, "--eval-max-steps", evaluation_cfg.get("eval_max_steps"))
    append_cli_arg(args, "--eval-deterministic", evaluation_cfg.get("eval_deterministic", True))
    if evaluation_cfg.get("eval_raw_reward"):
        args.append("--eval-raw-reward")

    if logging_cfg.get("wandb_project"):
        append_cli_arg(args, "--with_wandb", True)
        append_cli_arg(args, "--wandb_project", logging_cfg.get("wandb_project"))
        append_cli_arg(args, "--wandb_group", logging_cfg.get("wandb_group"))
        append_cli_arg(args, "--wandb_job_type", logging_cfg.get("wandb_job_type"))
        wandb_tags = logging_cfg.get("wandb_tags")
        if wandb_tags:
            args.append("--wandb_tags")
            args.extend(_wandb_tags_to_cli_values(wandb_tags))

    extra_args = backend_cfg.get("extra_args", ())
    if extra_args:
        args.extend(str(item) for item in extra_args)
    return args
