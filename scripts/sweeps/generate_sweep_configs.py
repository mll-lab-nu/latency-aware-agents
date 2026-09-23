from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config, merge_dicts
from latency_bench.utils.io import read_yaml, write_yaml
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    distribution_frames_to_ms_latency_config,
)
from scripts.experiment_runner.job_manifest import (
    build_job_id,
    build_upload_repo_path,
    format_number_label,
    SCHEMA_VERSION,
    write_manifest,
)
from training.common.config_adapter import apply_checkpoint_kind_override


SPEC_ONLY_KEYS = {
    "generated_root",
    "train_dir",
    "output_root",
    "device",
    "restart_behavior",
    "train_upload_repo_id",
    "train_upload_repo_type",
    "train_upload_path_prefix",
    "eval_upload_repo_id",
    "eval_upload_repo_type",
    "eval_upload_path_prefix",
    "experiment_family",
    "seeds",
    "latency_settings",
    "sweeps",
}

TOP_LEVEL_SAMPLE_FACTORY_KEYS = (
    "experiment",
    "backend",
    "executor",
    "env",
    "latency",
    "scheduler",
    "training",
    "evaluation",
    "logging",
)

EXECUTOR_KEYS = (
    "mode",
    "simulated_worker_capacity",
)

LATENCY_KEYS = (
    "method",
    "fixed_latency_ms",
    "mean_latency_ms",
    "std_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "seed",
    "profile_path",
    "profile_worker_slot",
    "per_episode",
    "add_latency_info",
    "max_pending_actions",
)

LOGGING_KEYS = (
    "output_dir",
    "wandb_project",
    "wandb_group",
    "wandb_job_type",
    "wandb_tags",
    "wandb_name",
)

EXPERIMENT_KEYS = (
    "name",
    "seed",
)

ENV_KEYS_BY_NAME = {
    "flappy": (
        "name",
        "env_fps",
        "obs_fps",
        "gym_id",
        "use_lidar",
        "normalize_obs",
        "audio_on",
        "frame_stack",
        "action_history_decisions",
        "obs_resize",
        "use_gpu_render",
        "simulator",
        "gpu_render_device",
        "gpu_render_profile",
        "gpu_render_profile_interval",
        "pipe_gap",
        "bird_color",
        "pipe_color",
        "background",
        "score_limit",
        "debug",
        "debug_timelimit_diagnostics",
        "max_episode_steps",
    ),
    "demon_attack": (
        "name",
        "env_id",
        "env_fps",
        "obs_fps",
        "screen_size",
        "noop_max",
        "reset_mode",
        "frame_stack",
        "terminal_on_life_loss",
        "action_history_decisions",
        "clip_reward",
        "repeat_action_probability",
        "full_action_space",
        "max_episode_steps",
        "simulator",
        "gpu_env_device",
    ),
    "deadly_corridor": (
        "name",
        "env_id",
        "env_fps",
        "obs_fps",
        "noop_action",
        "frame_stack",
        "res_w",
        "action_history_decisions",
        "res_h",
        "wide_aspect_ratio",
        "max_episode_steps",
        "simulator",
        "gpu_env_device",
        "gpu_env_profile",
        "gpu_env_profile_interval",
    ),
    "defend_the_line": (
        "name",
        "env_id",
        "env_fps",
        "obs_fps",
        "noop_action",
        "frame_stack",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
        "max_episode_steps",
        "simulator",
        "gpu_env_device",
        "gpu_env_profile",
        "gpu_env_profile_interval",
        "vizdoom_init_max_parallel",
        "vizdoom_init_lock_timeout_sec",
        "vizdoom_init_lock_max_wait_sec",
        "vizdoom_init_retry_jitter_sec",
    ),
    "gymnasium": (
        "name",
        "task_name",
        "env_id",
        "registration_imports",
        "make_kwargs",
        "env_fps",
        "obs_fps",
        "frame_stack",
        "simulator",
        "gpu_env_device",
        "terminal_on_life_loss",
        "clip_reward",
        "noop_action",
        "action_map",
        "action_order",
        "base_prompt",
    ),
}

SCHEDULER_KEYS = (
    "hold_policy",
    "ordering_policy",
)

LATENCY_BENCH_POLICY_TYPES = {
    "flappy": "flappy_sf",
    "demon_attack": "demon_attack_sf",
    "deadly_corridor": "deadly_corridor_sf",
    "defend_the_line": "deadly_corridor_sf",
    "gymnasium": "gymnasium_sf",
}


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _frame_ms(config: dict[str, Any]) -> float:
    return 1000.0 / float(config["env"]["env_fps"])


def _env_name(config: dict[str, Any]) -> str:
    return str(config["env"]["name"])


def _task_name(config: dict[str, Any]) -> str:
    env_cfg = config["env"]
    if env_cfg["name"] == "gymnasium":
        return env_cfg["task_name"]
    return env_cfg["name"]


def _env_slug(env_name: str) -> str:
    return str(env_name).replace("_", "-")


def _number_label(value: int | float) -> str:
    return format_number_label(value)


def _parse_int_csv(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _apply_latency_setting(config: dict[str, Any], setting: dict[str, Any], seed: int) -> None:
    latency_cfg = config.setdefault("latency", {})
    kind = str(setting["kind"])
    for key in (
        "fixed_latency_ms",
        "mean_latency_ms",
        "std_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "seed",
        "profile_path",
        "profile_worker_slot",
        "per_episode",
    ):
        latency_cfg.pop(key, None)

    if kind == "fixed":
        latency_cfg["method"] = "fixed"
        latency_cfg["fixed_latency_ms"] = float(setting["frames"]) * _frame_ms(config)
    elif kind in {"iid", "temporal"}:
        latency_cfg.update(
            method=kind,
            profile_path=str(setting["profile_path"]),
            profile_worker_slot=int(setting["profile_worker_slot"]),
            seed=int(seed),
        )
    else:
        spec = _latency_distribution_spec(kind, setting)
        latency_cfg.update(
            distribution_frames_to_ms_latency_config(
                spec,
                env_fps=float(config["env"]["env_fps"]),
                seed=int(seed),
            )
        )
        if kind == "mixed_normal":
            latency_cfg["per_episode"] = True
        elif kind == "mixed_uniform":
            latency_cfg["per_episode"] = True
    latency_cfg["add_latency_info"] = False


def _latency_manifest_fields(setting: dict[str, Any]) -> dict[str, str]:
    row = {
        "latency_kind": str(setting["kind"]),
        "latency_fixed_frames": "",
        "latency_mean_frames": "",
        "latency_std_frames": "",
        "latency_min_frames": "",
        "latency_max_frames": "",
    }
    if row["latency_kind"] == "fixed":
        row["latency_fixed_frames"] = _number_label(setting["frames"])
        return row
    if row["latency_kind"] in {"iid", "temporal"}:
        return row
    if row["latency_kind"] in {"normal", "mixed_normal"}:
        row["latency_mean_frames"] = _number_label(setting["mean"])
        row["latency_std_frames"] = _number_label(setting["std"])
    row["latency_min_frames"] = _number_label(setting["min"])
    row["latency_max_frames"] = _number_label(setting["max"])
    return row


def _latency_distribution_spec(kind: str, setting: dict[str, Any]) -> LatencyDistributionSpec:
    if kind in {"normal", "mixed_normal"}:
        return LatencyDistributionSpec(
            type="normal",
            mean=float(setting["mean"]),
            std=float(setting["std"]),
            min=float(setting["min"]),
            max=float(setting["max"]),
        )
    return LatencyDistributionSpec(
        type="uniform",
        mean=None,
        std=None,
        min=float(setting["min"]),
        max=float(setting["max"]),
    )


def _matched_eval_suites(
    setting: dict[str, Any],
    *,
    latency_label: str,
    seed: int,
) -> dict[str, list[Any]]:
    if "eval_suites" in setting:
        return deepcopy(setting["eval_suites"])

    kind = setting["kind"]
    if kind in {"iid", "temporal"}:
        return {
            kind: [
                {
                    "name": latency_label,
                    "profile_path": setting["profile_path"],
                    "seed": int(seed),
                }
            ]
        }

    def fixed_suite() -> dict[str, list[list[Any]]]:
        return {
            "fixed": [[setting["frames"]]],
            "normal": [],
            "uniform": [],
        }

    def normal_suite() -> dict[str, list[list[Any]]]:
        return {
            "fixed": [],
            "normal": [[setting["mean"], setting["std"], setting["min"], setting["max"]]],
            "uniform": [],
        }

    def uniform_suite() -> dict[str, list[list[Any]]]:
        return {
            "fixed": [],
            "normal": [],
            "uniform": [[setting["min"], setting["max"]]],
        }

    builders = {
        "fixed": fixed_suite,
        "normal": normal_suite,
        "uniform": uniform_suite,
    }
    return builders[kind]()


def _run_axis_fragment(
    sweep_name: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
) -> str:
    if sweep_name == "frame_stack":
        return f"fs{int(frame_stack)}"
    if sweep_name == "observation_stride":
        return f"stride{_number_label(observation_stride)}"
    return f"obs{_number_label(obs_fps)}"


def _run_name(
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> str:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
    return f"{env_name}_{sweep_name}_{latency_label}_{axis}_seed{int(seed)}"


def _wandb_name(
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> str:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
    return f"{_env_slug(env_name)}-{axis}-{latency_label}-s{int(seed)}"


def _wandb_group(
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
) -> str:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
    return f"{_env_slug(env_name)}-{axis}-{latency_label}"


def _wandb_tags(
    experiment_family: str,
    env_name: str,
    sweep_name: str,
    latency_label: str,
    latency_setting: dict[str, Any],
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> list[str]:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
    latency_kind = str(latency_setting["kind"])
    return [
        str(experiment_family),
        _env_slug(env_name),
        str(sweep_name),
        latency_kind,
        str(latency_label),
        axis,
        f"seed{int(seed)}",
    ]


def _run_dir(
    generated_root: Path,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> Path:
    return (
        generated_root
        / sweep_name
        / latency_label
        / _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
        / f"seed_{int(seed)}"
    )


def _output_dir(
    output_root: Path,
    stage: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> Path:
    return (
        output_root
        / stage
        / sweep_name
        / latency_label
        / _run_axis_fragment(sweep_name, frame_stack, obs_fps, observation_stride)
        / f"seed_{int(seed)}"
    )


def _build_manifest_row(
    *,
    experiment_family: str,
    env_name: str,
    sweep_name: str,
    latency_label: str,
    latency_setting: dict[str, Any],
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
    train_config: dict[str, Any],
    eval_config: dict[str, Any],
    train_path: Path,
    eval_path: Path,
    train_command: str,
    eval_command: str,
    train_upload_repo_id: str,
    train_upload_repo_type: str,
    train_upload_prefix: str,
    eval_upload_repo_id: str,
    eval_upload_repo_type: str,
    eval_upload_prefix: str,
) -> dict[str, str]:
    run_name = str(train_config["experiment"]["name"])
    train_dir = Path(str(train_config["backend"]["train_dir"]))
    train_output_dir = Path(str(train_config["logging"]["output_dir"]))
    eval_output_dir = Path(str(eval_config["logging"]["output_dir"]))
    job_id = build_job_id(
        experiment_family=experiment_family,
        env_name=env_name,
        sweep_name=sweep_name,
        latency_label=latency_label,
        frame_stack=frame_stack,
        obs_fps=obs_fps,
        observation_stride=observation_stride,
        seed=seed,
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "env": env_name,
        "experiment_family": experiment_family,
        "sweep_name": sweep_name,
        "latency_label": latency_label,
        "seed": str(int(seed)),
        "frame_stack": str(int(frame_stack)),
        "obs_fps": _number_label(obs_fps),
        "observation_stride": _number_label(observation_stride),
        "train_config": str(train_path),
        "eval_kind": "latency_bench",
        "eval_config": str(eval_path),
        "train_command": train_command,
        "eval_command": eval_command,
        "run_name": run_name,
        "wandb_group": str(train_config["logging"].get("wandb_group", "")),
        "wandb_name": str(train_config["logging"].get("wandb_name", run_name)),
        "train_dir": str(train_dir),
        "checkpoint_dir": str(train_dir / run_name / "checkpoint_p0"),
        "train_output_dir": str(train_output_dir),
        "eval_output_dir": str(eval_output_dir),
        "train_artifact_dir": str(train_dir / run_name),
        "eval_artifact_dir": str(eval_output_dir),
        "train_upload_repo_id": train_upload_repo_id,
        "train_upload_repo_type": train_upload_repo_type,
        "eval_upload_repo_id": eval_upload_repo_id,
        "eval_upload_repo_type": eval_upload_repo_type,
        "train_artifact_repo_path": build_upload_repo_path(upload_prefix=train_upload_prefix, stage="train", job_id=job_id),
        "eval_artifact_repo_path": build_upload_repo_path(upload_prefix=eval_upload_prefix, stage="eval", job_id=job_id),
    }
    row.update(_latency_manifest_fields(latency_setting))
    return row


def _prune_sample_factory_config(config: dict[str, Any]) -> dict[str, Any]:
    pruned = {key: deepcopy(config[key]) for key in TOP_LEVEL_SAMPLE_FACTORY_KEYS}

    experiment_cfg = pruned["experiment"]
    pruned["experiment"] = {
        key: deepcopy(experiment_cfg[key])
        for key in EXPERIMENT_KEYS
        if key in experiment_cfg
    }

    executor_cfg = pruned["executor"]
    pruned["executor"] = {
        key: deepcopy(executor_cfg[key])
        for key in EXECUTOR_KEYS
        if key in executor_cfg
    }

    env_cfg = pruned["env"]
    env_keys = ENV_KEYS_BY_NAME[str(env_cfg["name"]).strip().lower()]
    pruned["env"] = {
        key: deepcopy(env_cfg[key])
        for key in env_keys
        if key in env_cfg
    }

    latency_cfg = pruned["latency"]
    pruned["latency"] = {
        key: deepcopy(latency_cfg[key])
        for key in LATENCY_KEYS
        if key in latency_cfg
    }

    scheduler_cfg = pruned["scheduler"]
    pruned["scheduler"] = {
        key: deepcopy(scheduler_cfg[key])
        for key in SCHEDULER_KEYS
        if key in scheduler_cfg
    }

    logging_cfg = pruned["logging"]
    pruned["logging"] = {
        key: deepcopy(logging_cfg[key])
        for key in LOGGING_KEYS
        if key in logging_cfg and not (isinstance(logging_cfg[key], str) and not logging_cfg[key].strip())
    }
    return pruned


def _build_full_train_config(
    base_config: dict[str, Any],
    *,
    sweep_name: str,
    sweep_cfg: dict[str, Any],
    latency_label: str,
    latency_setting: dict[str, Any],
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
    train_dir: Path,
    output_root: Path,
    device: str,
    restart_behavior: str,
    experiment_family: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    env_name = _task_name(config)
    config.setdefault("experiment", {})["name"] = _run_name(
        env_name,
        sweep_name,
        latency_label,
        frame_stack,
        obs_fps,
        observation_stride,
        seed,
    )
    config["experiment"]["seed"] = int(seed)
    config.setdefault("backend", {})["type"] = "sample_factory"
    config["backend"]["device"] = str(device)
    config["backend"]["train_dir"] = str(train_dir)
    config["backend"]["restart_behavior"] = str(restart_behavior)
    config["backend"]["run_mode"] = "train"
    config.setdefault("env", {})["frame_stack"] = int(frame_stack)
    config["env"]["obs_fps"] = obs_fps
    _apply_latency_setting(config, latency_setting, int(seed))
    config.setdefault("evaluation", {})["eval_suites"] = _matched_eval_suites(
        latency_setting,
        latency_label=latency_label,
        seed=seed,
    )
    config.setdefault("logging", {})["output_dir"] = str(
        _output_dir(output_root, "train", sweep_name, latency_label, frame_stack, obs_fps, observation_stride, seed)
    )
    config["logging"]["wandb_group"] = _wandb_group(
        env_name,
        sweep_name,
        latency_label,
        frame_stack,
        obs_fps,
        observation_stride,
    )
    config["logging"]["wandb_name"] = _wandb_name(
        env_name,
        sweep_name,
        latency_label,
        frame_stack,
        obs_fps,
        observation_stride,
        seed,
    )
    specified_tags = config["logging"]["wandb_tags"]
    config["logging"]["wandb_tags"] = _wandb_tags(
        experiment_family,
        env_name,
        sweep_name,
        latency_label,
        latency_setting,
        frame_stack,
        obs_fps,
        observation_stride,
        seed,
    )
    if isinstance(specified_tags, list):
        config["logging"]["wandb_tags"].extend(specified_tags)
    return config


def _drop_eval_time_limit_fields(config: dict[str, Any]) -> None:
    config["env"].pop("max_episode_steps", None)


def _align_demon_attack_raw_eval_config(config: dict[str, Any], env_name: str) -> None:
    if env_name != "demon_attack":
        return
    if not config["evaluation"]["eval_raw_reward"]:
        return
    config["env"]["terminal_on_life_loss"] = True
    config["env"]["clip_reward"] = False


def _build_latency_bench_eval_config(
    base_config: dict[str, Any],
    full_train_config: dict[str, Any],
    *,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
    output_root: Path,
    latency_bench_policy_device: str,
    checkpoint_kind: str = "latest",
) -> dict[str, Any]:
    config = merge_dicts(base_config, full_train_config)
    config["evaluation"]["eval_suites"] = deepcopy(full_train_config["evaluation"]["eval_suites"])
    env_name = _env_name(config)
    config["backend"]["run_mode"] = "eval"
    if env_name in ("asterix", "atlantis"):
        config["env"]["simulator"] = "cpu"
    _drop_eval_time_limit_fields(config)
    _align_demon_attack_raw_eval_config(config, env_name)
    config["policy"] = merge_dicts(
        base_config["policy"],
        {
            "type": LATENCY_BENCH_POLICY_TYPES[env_name],
            "checkpoint_path": "",
            "device": str(latency_bench_policy_device),
            "deterministic": True,
        },
    )
    config["policy"]["checkpoint_kind"] = checkpoint_kind
    apply_checkpoint_kind_override(config["backend"], checkpoint_kind)
    config["logging"]["output_dir"] = str(
        _output_dir(
            output_root,
            "latency_bench",
            sweep_name,
            latency_label,
            frame_stack,
            obs_fps,
            observation_stride,
            seed,
        )
    )
    config["logging"]["save_step_records"] = False
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False
    if env_name == "deadly_corridor":
        config["env"]["noop_action"] = [0, 0, 0, 0]
    elif env_name == "defend_the_line":
        config["env"]["noop_action"] = [0, 0]
    _disable_periodic_eval(config)
    return config


def _disable_periodic_eval(config: dict[str, Any]) -> None:
    del config["evaluation"]["eval_interval_steps"]


def _iter_runs(base_config: dict[str, Any], spec: dict[str, Any], *, seeds: Sequence[int] | None = None):
    effective_seeds = [int(seed) for seed in (spec["seeds"] if seeds is None else seeds)]
    latency_settings = spec["latency_settings"]
    env_fps = float(base_config["env"]["env_fps"])
    for sweep_name, sweep_cfg in spec["sweeps"].items():
        for latency_label, latency_setting in latency_settings.items():
            for frame_stack in sweep_cfg["frame_stack"]:
                if "observation_stride" in sweep_cfg:
                    obs_settings = [
                        (env_fps / float(observation_stride), float(observation_stride))
                        for observation_stride in sweep_cfg["observation_stride"]
                    ]
                else:
                    obs_settings = [
                        (float(obs_fps), env_fps / float(obs_fps))
                        for obs_fps in sweep_cfg["obs_fps"]
                    ]
                for obs_fps, observation_stride in obs_settings:
                    for seed in effective_seeds:
                        yield (
                            sweep_name,
                            sweep_cfg,
                            latency_label,
                            latency_setting,
                            int(frame_stack),
                            float(obs_fps),
                            float(observation_stride),
                            int(seed),
                        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate configs for sweep-based experiments.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--restart-behavior", type=str)
    parser.add_argument("--latency-bench-policy-device", type=str, default="auto")
    parser.add_argument(
        "--eval-checkpoint-kind",
        choices=("latest", "best"),
        default="latest",
    )
    parser.add_argument("--train-upload-repo-id", type=str, default="")
    parser.add_argument("--train-upload-repo-type", type=str, default="model")
    parser.add_argument("--train-upload-path-prefix", type=str, default="")
    parser.add_argument("--eval-upload-repo-id", type=str, default="")
    parser.add_argument("--eval-upload-repo-type", type=str, default="dataset")
    parser.add_argument("--eval-upload-path-prefix", type=str, default="")
    args = parser.parse_args(argv)

    spec_path = args.spec.expanduser()
    if not spec_path.is_absolute():
        spec_path = (REPO_ROOT / spec_path).resolve()
    spec = read_yaml(spec_path)
    explicit_eval_interval = "eval_interval_steps" in spec.get("evaluation", {})
    if "__base__" in spec:
        spec = load_config(spec_path)
        base_config = {
            key: value
            for key, value in spec.items()
            if key not in SPEC_ONLY_KEYS
        }
    else:
        base_config_path = _resolve_path(spec["base_config"], base_dir=REPO_ROOT)
        base_config = load_config(base_config_path)
    generated_root = Path(args.generated_root or spec["generated_root"]).expanduser()
    output_root = Path(args.output_root or spec["output_root"])
    train_dir = Path(args.train_dir or spec["train_dir"])
    device = str(args.device or spec.get("device", "cuda"))
    restart_behavior = str(args.restart_behavior or spec.get("restart_behavior", "resume"))
    train_upload_repo_id = str(args.train_upload_repo_id or spec.get("train_upload_repo_id", ""))
    train_upload_repo_type = str(args.train_upload_repo_type or spec.get("train_upload_repo_type", "model"))
    train_upload_path_prefix = str(args.train_upload_path_prefix or spec.get("train_upload_path_prefix", ""))
    eval_upload_repo_id = str(args.eval_upload_repo_id or spec.get("eval_upload_repo_id", ""))
    eval_upload_repo_type = str(args.eval_upload_repo_type or spec.get("eval_upload_repo_type", "dataset"))
    eval_upload_path_prefix = str(args.eval_upload_path_prefix or spec.get("eval_upload_path_prefix", ""))
    experiment_family = str(spec["experiment_family"])
    seeds = None if args.seeds is None else _parse_int_csv(args.seeds)

    manifest_rows: list[dict[str, str]] = []
    train_commands: list[str] = []
    eval_commands: list[str] = []

    for (
        sweep_name,
        sweep_cfg,
        latency_label,
        latency_setting,
        frame_stack,
        obs_fps,
        observation_stride,
        seed,
    ) in _iter_runs(
        base_config,
        spec,
        seeds=seeds,
    ):
        full_train_config = _build_full_train_config(
            base_config,
            sweep_name=sweep_name,
            sweep_cfg=sweep_cfg,
            latency_label=latency_label,
            latency_setting=latency_setting,
            frame_stack=frame_stack,
            obs_fps=obs_fps,
            observation_stride=observation_stride,
            seed=seed,
            train_dir=train_dir,
            output_root=output_root,
            device=device,
            restart_behavior=restart_behavior,
            experiment_family=experiment_family,
        )
        train_config = _prune_sample_factory_config(full_train_config)
        if not explicit_eval_interval:
            _disable_periodic_eval(train_config)
        latency_bench_eval_config = _build_latency_bench_eval_config(
            base_config,
            full_train_config,
            sweep_name=sweep_name,
            latency_label=latency_label,
            frame_stack=frame_stack,
            obs_fps=obs_fps,
            observation_stride=observation_stride,
            seed=seed,
            output_root=output_root,
            latency_bench_policy_device=args.latency_bench_policy_device,
            checkpoint_kind=args.eval_checkpoint_kind,
        )

        run_dir = _run_dir(generated_root, sweep_name, latency_label, frame_stack, obs_fps, observation_stride, seed)
        train_path = run_dir / "train.yaml"
        latency_bench_eval_path = run_dir / "latency_bench_eval.yaml"
        write_yaml(train_path, train_config)
        write_yaml(latency_bench_eval_path, latency_bench_eval_config)

        train_cmd = f"python -m training.train --config {shlex.quote(str(train_path))}"
        eval_cmd = f"python -m latency_bench.run --config {shlex.quote(str(latency_bench_eval_path))}"
        train_commands.append(train_cmd)
        eval_commands.append(eval_cmd)
        manifest_rows.append(
            _build_manifest_row(
                experiment_family=experiment_family,
                env_name=_task_name(train_config),
                sweep_name=sweep_name,
                latency_label=latency_label,
                latency_setting=latency_setting,
                frame_stack=frame_stack,
                obs_fps=obs_fps,
                observation_stride=observation_stride,
                seed=seed,
                train_config=train_config,
                eval_config=latency_bench_eval_config,
                train_path=train_path,
                eval_path=latency_bench_eval_path,
                train_command=train_cmd,
                eval_command=eval_cmd,
                train_upload_repo_id=train_upload_repo_id,
                train_upload_repo_type=train_upload_repo_type,
                train_upload_prefix=train_upload_path_prefix,
                eval_upload_repo_id=eval_upload_repo_id,
                eval_upload_repo_type=eval_upload_repo_type,
                eval_upload_prefix=eval_upload_path_prefix,
            )
        )

    generated_root.mkdir(parents=True, exist_ok=True)
    write_manifest(generated_root / "manifest.tsv", manifest_rows)
    (generated_root / "train_commands.txt").write_text("\n".join(train_commands) + "\n", encoding="utf-8")
    (generated_root / "latency_bench_eval_commands.txt").write_text(
        "\n".join(eval_commands) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote sweep configs to {generated_root}")
    print(f"Train commands: {generated_root / 'train_commands.txt'}")
    print(f"Latency bench eval commands: {generated_root / 'latency_bench_eval_commands.txt'}")
    print(f"Manifest: {generated_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
