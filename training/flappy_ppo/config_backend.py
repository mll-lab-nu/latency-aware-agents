from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from modeling.flappy_ppo import FlappyPpoObservationConfig
from training.common.checkpoints import experiment_dir
from training.flappy_ppo.cli import build_parser
from training.sample_factory_common.latency_config import latency_ms_to_raw_frames


UNSUPPORTED_TOP_LEVEL_FIELDS = {
    "executor",
    "policy",
    "scheduler",
    "evaluation",
}

UNSUPPORTED_BACKEND_FIELDS = {
    "algo",
    "restart_behavior",
    "extra_args",
}

UNSUPPORTED_TRAINING_FIELDS = {
    "async_rl",
    "batched_sampling",
    "exploration_loss",
    "exploration_loss_coeff",
    "experiment_summaries_interval",
    "keep_checkpoints",
    "normalize_input",
    "normalize_returns",
    "num_batches_per_epoch",
    "num_batches_to_accumulate",
    "num_policies",
    "ppo_clip_value",
    "recurrence",
    "rnn_size",
    "rnn_type",
    "save_best_after",
    "save_best_every_sec",
    "save_every_sec",
    "save_milestones_sec",
    "stats_avg",
    "use_rnn",
    "worker_num_splits",
    "with_wandb",
}

SUPPORTED_TRAINING_FIELDS = {
    "batch_size",
    "gae_lambda",
    "gamma",
    "learning_rate",
    "max_grad_norm",
    "num_envs_per_worker",
    "num_epochs",
    "num_workers",
    "ppo_clip_ratio",
    "rollout",
    "train_for_env_steps",
    "value_loss_coeff",
}

SUPPORTED_BACKEND_FIELDS = {
    "device",
    "run_mode",
    "train_dir",
    "type",
}

SUPPORTED_ENV_FIELDS = {
    "env_fps",
    "frame_stack",
    "name",
    "observation_mode",
    "render_mode",
    "screen_size",
}

SUPPORTED_FLAPPY_PPO_FIELDS = {
    "anneal_lr",
    "checkpoint_freq",
    "clip_coef",
    "clip_coef_final",
    "clip_vloss",
    "ent_coef",
    "grayscale",
    "image_size",
    "learning_rate_final",
    "load",
    "log_interval_seconds",
    "norm_adv",
    "profile",
    "profile_csv",
    "profile_sync_cuda",
    "profile_warmup_updates",
    "stats_window_size",
    "target_kl",
    "torch_deterministic",
}

SUPPORTED_LOGGING_FIELDS = {
    "wandb_entity",
    "wandb_name",
    "wandb_project",
    "wandb_tags",
}


def validate_flappy_ppo_config(config: Mapping[str, Any]) -> None:
    _require_mapping(config, "config")
    _reject_present_fields(config, UNSUPPORTED_TOP_LEVEL_FIELDS, "")

    experiment_cfg = _section(config, "experiment")
    backend_cfg = _section(config, "backend")
    env_cfg = _section(config, "env")
    latency_cfg = _section(config, "latency")
    training_cfg = _section(config, "training")
    flappy_cfg = _section(config, "flappy_ppo", required=False)
    logging_cfg = _section(config, "logging", required=False)

    _require_non_empty_string(experiment_cfg, "name", "experiment.name")
    if str(backend_cfg.get("type", "")).strip().lower() != "flappy_ppo":
        raise ValueError("backend.type must be 'flappy_ppo' for the flappy_ppo backend")
    if str(env_cfg.get("name", "")).strip().lower() != "flappy":
        raise ValueError("env.name must be 'flappy' for the flappy_ppo backend")
    if "run_mode" in backend_cfg and str(backend_cfg.get("run_mode", "train")).strip().lower() != "train":
        raise ValueError("backend.run_mode must be 'train' for flappy_ppo training")

    _reject_present_fields(backend_cfg, UNSUPPORTED_BACKEND_FIELDS, "backend")
    _reject_present_fields(training_cfg, UNSUPPORTED_TRAINING_FIELDS, "training")
    _reject_unknown_fields(backend_cfg, SUPPORTED_BACKEND_FIELDS, "backend")
    _reject_unknown_fields(env_cfg, SUPPORTED_ENV_FIELDS, "env")
    _reject_unknown_fields(training_cfg, SUPPORTED_TRAINING_FIELDS, "training")
    _reject_unknown_fields(flappy_cfg, SUPPORTED_FLAPPY_PPO_FIELDS, "flappy_ppo")
    _reject_unknown_fields(logging_cfg, SUPPORTED_LOGGING_FIELDS, "logging")

    if env_cfg.get("screen_size") is not None:
        raise ValueError("env.screen_size is not supported by flappy_ppo; resize is controlled by flappy_ppo.image_size")
    if "render_mode" in env_cfg and env_cfg.get("render_mode") is not None:
        raise ValueError("env.render_mode is not used by flappy_ppo training; omit it from flappy_ppo configs")
    if "observation_mode" in env_cfg and env_cfg.get("observation_mode") not in (None, "image"):
        raise ValueError("env.observation_mode must be omitted or 'image' for flappy_ppo")

    if bool(latency_cfg.get("add_latency_info", False)):
        raise ValueError("latency.add_latency_info must be false for flappy_ppo image observations")

    n_envs = _positive_int(training_cfg.get("num_workers"), "training.num_workers") * _positive_int(
        training_cfg.get("num_envs_per_worker"),
        "training.num_envs_per_worker",
    )
    rollout = _positive_int(training_cfg.get("rollout"), "training.rollout")
    batch_size = _positive_int(training_cfg.get("batch_size"), "training.batch_size")
    if batch_size > n_envs * rollout:
        raise ValueError("training.batch_size must be <= training.num_workers * training.num_envs_per_worker * training.rollout")


def build_observation_config(config: Mapping[str, Any]) -> FlappyPpoObservationConfig:
    validate_flappy_ppo_config(config)
    env_cfg = config["env"]
    flappy_cfg = config.get("flappy_ppo", {})
    return FlappyPpoObservationConfig(
        image_size=int(flappy_cfg.get("image_size", 84)),
        grayscale=bool(flappy_cfg.get("grayscale", False)),
        stack_frames=int(env_cfg.get("frame_stack", 4)),
    )


def build_flappy_ppo_args(config: Mapping[str, Any]) -> Namespace:
    validate_flappy_ppo_config(config)
    parser = build_parser()
    args = parser.parse_args([])

    experiment_cfg = config["experiment"]
    backend_cfg = config["backend"]
    env_cfg = config["env"]
    latency_cfg = config["latency"]
    training_cfg = config["training"]
    flappy_cfg = config.get("flappy_ppo", {})
    logging_cfg = config.get("logging", {})

    args.train_steps = int(training_cfg["train_for_env_steps"])
    args.n_envs = int(training_cfg["num_workers"]) * int(training_cfg["num_envs_per_worker"])
    args.n_steps = int(training_cfg["rollout"])
    args.batch_size = int(training_cfg["batch_size"])
    args.n_epochs = int(training_cfg["num_epochs"])
    args.seed = int(experiment_cfg.get("seed", 0))
    args.device = str(backend_cfg.get("device", "auto"))
    args.env_fps = float(env_cfg.get("env_fps", 30))
    args.learning_rate = float(training_cfg["learning_rate"])
    args.gamma = float(training_cfg["gamma"])
    args.gae_lambda = float(training_cfg["gae_lambda"])
    args.vf_coef = float(training_cfg["value_loss_coeff"])
    args.max_grad_norm = float(training_cfg["max_grad_norm"])
    if "ppo_clip_ratio" in training_cfg:
        args.clip_coef = float(training_cfg["ppo_clip_ratio"])

    obs_config = build_observation_config(config)
    args.image_size = obs_config.image_size
    args.grayscale = obs_config.grayscale
    args.stack_frames = obs_config.stack_frames
    args.latency_type = str(latency_cfg["type"])
    args.fixed_latency_ms = latency_cfg.get("fixed_latency_ms")
    args.mean_latency_ms = latency_cfg.get("mean_latency_ms")
    args.std_latency_ms = latency_cfg.get("std_latency_ms")
    args.min_latency_ms = latency_cfg.get("min_latency_ms")
    args.max_latency_ms = latency_cfg.get("max_latency_ms")
    args.latency_seed = latency_cfg.get("seed")
    if str(latency_cfg["type"]) == "zero":
        args.frame_skip_values = "0"
    elif str(latency_cfg["type"]) == "fixed":
        args.frame_skip_values = str(latency_ms_to_raw_frames(latency_cfg["fixed_latency_ms"], env_fps=args.env_fps))
    else:
        args.frame_skip_values = "0"
    args.frame_skip_training_setting = "action_delay_frame_skip"

    for field in SUPPORTED_FLAPPY_PPO_FIELDS:
        if field not in flappy_cfg:
            continue
        value = flappy_cfg[field]
        target = "checkpoint_freq" if field == "checkpoint_freq" else field
        setattr(args, target, value)
    args.image_size = int(args.image_size)
    args.stack_frames = int(args.stack_frames)
    args.checkpoint_freq = int(args.checkpoint_freq)

    train_root = Path(str(backend_cfg["train_dir"])).expanduser()
    exp_dir = experiment_dir(train_root, str(experiment_cfg["name"]))
    args.train_dir = str(train_root)
    args.experiment_name = str(experiment_cfg["name"])
    args.experiment_dir = str(exp_dir)
    args.metrics_jsonl = str(exp_dir / "episode_metrics.jsonl")
    args.resolved_config_json = str(exp_dir / "resolved_config.json")
    args.metadata_json = str(exp_dir / "metadata.json")
    args.save = str(exp_dir / "checkpoint_p0" / "checkpoint.pth")
    args.unified_checkpoint_layout = True
    args.resolved_config = resolved_config(config)

    args.wandb_project = str(logging_cfg.get("wandb_project", args.wandb_project))
    args.wandb_entity = logging_cfg.get("wandb_entity", args.wandb_entity)
    args.wandb_run_name = str(logging_cfg.get("wandb_name") or experiment_cfg["name"])
    args.wandb_tags = _csv(logging_cfg.get("wandb_tags", ""))
    args.run_id = str(experiment_cfg["name"])
    return args


def resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_flappy_ppo_config(config)
    return deepcopy(dict(config))


def write_resolved_artifacts(config: Mapping[str, Any], args: Namespace) -> None:
    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    obs_config = build_observation_config(config)
    resolved = resolved_config(config)
    (exp_dir / "config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (exp_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "backend_type": "flappy_ppo",
        "experiment_name": args.experiment_name,
        "frame_skip_training_setting": "action_delay_frame_skip",
        "latency": deepcopy(dict(config["latency"])),
        "obs_shape": [obs_config.in_channels, obs_config.image_size, obs_config.image_size],
        "observation_config": obs_config.to_dict(),
    }
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_from_config(config: Mapping[str, Any]) -> int:
    from training.flappy_ppo.logging import setup_wandb
    from training.flappy_ppo.ppo_loop import train

    args = build_flappy_ppo_args(config)
    write_resolved_artifacts(config, args)
    wandb_module = setup_wandb(args)
    try:
        train(args, wandb_module=wandb_module)
    finally:
        wandb_module.finish()
    return 0


def _section(config: Mapping[str, Any], name: str, *, required: bool = True) -> Mapping[str, Any]:
    value = config.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")


def _require_non_empty_string(config: Mapping[str, Any], key: str, path: str) -> None:
    if not str(config.get(key, "")).strip():
        raise ValueError(f"{path} must be a non-empty string")


def _reject_present_fields(config: Mapping[str, Any], fields: set[str], prefix: str) -> None:
    for field in sorted(fields):
        if field in config and config[field] is not None:
            path = f"{prefix}.{field}" if prefix else field
            raise ValueError(f"{path} is not supported by flappy_ppo")


def _reject_unknown_fields(config: Mapping[str, Any], supported: set[str], prefix: str) -> None:
    for field in sorted(config):
        if field not in supported:
            path = f"{prefix}.{field}" if prefix else field
            raise ValueError(f"{path} is not supported by flappy_ppo")


def _positive_int(value: Any, path: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return parsed


def _csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)
