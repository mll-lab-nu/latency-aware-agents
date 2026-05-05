from __future__ import annotations

import json
import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.latency_distribution import parse_latency_distribution

ENV_INTEGRATIONS = {
    "deadly_corridor": "training.deadly_corridor_sf.integration",
    "demon_attack": "training.demon_attack_sf.integration",
    "flappy": "training.flappy_sf.integration",
}

CHECKPOINT_KINDS = {"latest", "best"}


def _list_to_csv(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def normalize_checkpoint_kind(checkpoint_kind: str) -> str:
    normalized = str(checkpoint_kind).strip().lower()
    if normalized not in CHECKPOINT_KINDS:
        raise ValueError("checkpoint kind must be 'latest' or 'best'")
    return normalized


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
    _validate_latency_training_config(latency_cfg)
    if env_name == "flappy":
        _validate_flappy_sf_config(config)


def _validate_latency_training_config(latency_cfg: dict[str, Any]) -> None:
    has_distribution = latency_cfg.get("train_distribution") is not None
    train_values = latency_cfg.get("train_values")
    has_train_values = train_values is not None and not (
        isinstance(train_values, (list, tuple)) and len(train_values) == 0
    )
    if has_distribution and has_train_values:
        raise ValueError("latency.train_values and latency.train_distribution are mutually exclusive")
    if has_distribution:
        parse_latency_distribution(latency_cfg["train_distribution"])

    if "within_episode_change_interval" in latency_cfg:
        try:
            interval = int(latency_cfg["within_episode_change_interval"])
        except (TypeError, ValueError) as exc:
            raise ValueError("latency.within_episode_change_interval must be a positive integer") from exc
        if interval < 1:
            raise ValueError("latency.within_episode_change_interval must be >= 1")
    if "within_episode_change_prob" in latency_cfg:
        try:
            probability = float(latency_cfg["within_episode_change_prob"])
        except (TypeError, ValueError) as exc:
            raise ValueError("latency.within_episode_change_prob must be a number in [0, 1]") from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError("latency.within_episode_change_prob must be in [0, 1]")


def _validate_flappy_sf_config(config: dict[str, Any]) -> None:
    screen_size_path = _find_key_path(config, "screen_size")
    if screen_size_path is not None:
        raise ValueError(
            f"{screen_size_path} is not supported for Flappy Sample Factory; "
            "use env.obs_resize to resize observations without changing env geometry"
        )

    env_cfg = config["env"]
    obs_resize = env_cfg.get("obs_resize")
    if obs_resize is not None:
        _parse_obs_resize_config(obs_resize)

    obs_mode = str(env_cfg.get("obs_mode", "image_only")).strip().lower()
    if obs_mode not in {"image_only", "latency_channel"}:
        raise ValueError("env.obs_mode must be 'image_only' or 'latency_channel' for Flappy Sample Factory")

    profile_interval = env_cfg.get("gpu_render_profile_interval")
    if profile_interval is not None:
        try:
            profile_interval_int = int(profile_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("env.gpu_render_profile_interval must be a positive integer") from exc
        if profile_interval_int <= 0:
            raise ValueError("env.gpu_render_profile_interval must be a positive integer")

    if _config_bool(env_cfg.get("use_gpu_render", False)):
        training_cfg = config["training"]
        try:
            original_num_envs_per_worker = int(training_cfg.get("num_envs_per_worker", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.num_envs_per_worker must be a positive integer when env.use_gpu_render=true") from exc
        if original_num_envs_per_worker < 1:
            raise ValueError("training.num_envs_per_worker must be >= 1 when env.use_gpu_render=true")
        try:
            worker_num_splits = int(training_cfg.get("worker_num_splits", 1) or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.worker_num_splits must be 1 when env.use_gpu_render=true") from exc
        if worker_num_splits != 1:
            raise ValueError("training.worker_num_splits must be 1 when env.use_gpu_render=true")
        if "batched_sampling" in config["training"] and not _config_bool(config["training"].get("batched_sampling")):
            raise ValueError("training.batched_sampling must be true when env.use_gpu_render=true")
        if obs_resize is None:
            raise ValueError("env.obs_resize is required when env.use_gpu_render=true for Flappy Sample Factory")
        height, width = _parse_obs_resize_config(obs_resize)
        if height != width:
            raise ValueError("env.obs_resize must be square when env.use_gpu_render=true for Flappy Sample Factory")
        gpu_render_device = str(env_cfg.get("gpu_render_device", "auto") or "auto").strip().lower()
        if not (gpu_render_device in {"auto", "cpu", "cuda"} or gpu_render_device.startswith("cuda:")):
            raise ValueError("env.gpu_render_device must be 'auto', 'cpu', 'cuda', or a cuda device string")
        actor_worker_gpus = _parse_actor_worker_gpus_config(config["backend"].get("actor_worker_gpus"))
        if gpu_render_device != "cpu" and actor_worker_gpus == []:
            raise ValueError(
                "backend.actor_worker_gpus must be omitted or non-empty when Flappy GPU rendering uses CUDA"
            )


def _find_key_path(value: Any, target_key: str, prefix: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) == target_key:
                return path
            found = _find_key_path(item, target_key, path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_key_path(item, target_key, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def _parse_obs_resize_config(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("env.obs_resize must be [height, width] for Flappy Sample Factory")
    if len(parts) != 2:
        raise ValueError("env.obs_resize must contain exactly two positive integers: [height, width]")
    height, width = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise ValueError("env.obs_resize height and width must be positive")
    return height, width


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_actor_worker_gpus_config(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parts = stripped.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("backend.actor_worker_gpus must be a list of GPU indices")
    return [int(item) for item in parts]


def estimate_optimizer_steps_per_1m_env_steps(training_cfg: dict[str, Any]) -> float:
    batch_size = int(training_cfg.get("batch_size", 0) or 0)
    num_epochs = int(training_cfg.get("num_epochs", 0) or 0)
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive to estimate optimizer update ratio")
    if num_epochs <= 0:
        raise ValueError("training.num_epochs must be positive to estimate optimizer update ratio")
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
    env_cfg = config.get("env", {})
    use_gpu_render = _config_bool(env_cfg.get("use_gpu_render", False))
    gpu_render_device = str(env_cfg.get("gpu_render_device", "auto") or "auto").strip().lower()
    actor_worker_gpus = backend_cfg.get("actor_worker_gpus")
    if use_gpu_render and gpu_render_device != "cpu" and actor_worker_gpus is None:
        actor_worker_gpus = [0]
    elif use_gpu_render and gpu_render_device != "cpu" and _parse_actor_worker_gpus_config(actor_worker_gpus) == []:
        raise ValueError("backend.actor_worker_gpus must be omitted or non-empty when Flappy GPU rendering uses CUDA")
    append_cli_multi_value_arg(args, "--actor_worker_gpus", actor_worker_gpus)
    if use_gpu_render:
        append_cli_arg(args, "--env_gpu_observations", gpu_render_device != "cpu")
        append_cli_arg(args, "--env_gpu_actions", False)
        original_num_envs_per_worker = int(training_cfg.get("num_envs_per_worker", 0) or 0)
        if original_num_envs_per_worker < 1:
            raise ValueError("training.num_envs_per_worker must be >= 1 when env.use_gpu_render=true")
        try:
            worker_num_splits = int(training_cfg.get("worker_num_splits", 1) or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.worker_num_splits must be 1 when env.use_gpu_render=true") from exc
        if worker_num_splits != 1:
            raise ValueError("training.worker_num_splits must be 1 when env.use_gpu_render=true")
        append_cli_arg(args, "--gpu-render-batch-size", original_num_envs_per_worker)
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
        value = training_cfg.get(field)
        if use_gpu_render and field == "num_envs_per_worker":
            value = 1
        elif use_gpu_render and field == "worker_num_splits" and value is None:
            value = 1
        append_cli_arg(args, f"--{field}", value)
    if use_gpu_render:
        append_cli_arg(args, "--batched_sampling", True)
    for field in bool_fields:
        if use_gpu_render and field == "batched_sampling":
            continue
        if field in training_cfg:
            append_cli_arg(args, f"--{field}", training_cfg.get(field))

    if latency_cfg.get("train_distribution") is not None:
        distribution_spec = parse_latency_distribution(latency_cfg["train_distribution"]).to_dict()
        args.extend(
            [
                "--latency-distribution-spec",
                json.dumps(distribution_spec, sort_keys=True, separators=(",", ":")),
            ]
        )
    elif "train_values" in latency_cfg:
        append_cli_arg(args, "--queue-latency-values", latency_cfg.get("train_values"))
    append_cli_arg(args, "--latency-mode", latency_cfg.get("train_mode", "episode_fixed"))
    append_cli_arg(args, "--latency-sampling-mode", latency_cfg.get("train_sampling_mode", "uniform_all"))
    append_cli_arg(args, "--episodes-per-stage", latency_cfg.get("episodes_per_stage"))
    append_cli_arg(args, "--randomize-within-stage", latency_cfg.get("randomize_within_stage", False))
    append_cli_arg(args, "--within-episode-change-interval", latency_cfg.get("within_episode_change_interval"))
    append_cli_arg(args, "--within-episode-change-prob", latency_cfg.get("within_episode_change_prob"))
    append_cli_arg(args, "--add-latency-info", latency_cfg.get("add_latency_info", False))
    append_cli_arg(args, "--max-pending-actions", latency_cfg.get("max_pending_actions"))

    append_cli_arg(args, "--eval-episodes", evaluation_cfg.get("eval_episodes"))
    append_cli_arg(args, "--eval-parallel-envs", evaluation_cfg.get("eval_parallel_envs"))
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
