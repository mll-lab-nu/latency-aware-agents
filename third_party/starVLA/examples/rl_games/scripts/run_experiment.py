#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.setup_training_assets import setup_assets
from starVLA.training.rl_games.auth import login_training_services


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read experiment configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _parse_scalar(value: str) -> Any:
    try:
        import yaml
        return yaml.safe_load(value)
    except Exception:
        return value


def _apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override}")
    key, raw_value = override.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override key: {key}")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {key}: {part} is not a mapping")
        node = child
    node[parts[-1]] = _parse_scalar(raw_value)


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _cross_task_eval_enabled(task_cfg: Any) -> bool:
    if not isinstance(task_cfg, dict):
        return True
    mid_train = task_cfg.get("mid_train") or {}
    post_train = task_cfg.get("post_train") or {}
    return _as_bool(mid_train.get("enabled", True)) or _as_bool(post_train.get("enabled", True))


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str(workspace_dir / path)


def _repo_or_workspace_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return str(workspace_dir / path)


def _workspace_dir(cfg: dict[str, Any]) -> Path:
    configured = _get(cfg, "workspace_dir")
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    default_workspace = Path("/workspace")
    if default_workspace.exists():
        return default_workspace.resolve()
    return REPO_ROOT


def _load_auth_env(cfg: dict[str, Any], workspace_dir: Path) -> None:
    login_training_services(cfg, workspace_dir=workspace_dir, repo_root=REPO_ROOT)


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if isinstance(value, str) and any(ch.isspace() or ch in {",", ":", "{", "}", "[", "]"} for ch in value):
        return shlex.quote(value)
    return str(value)


CONFIG_GROUP_KEYS = {"model", "env", "init", "mode"}
TRAINER_COMMAND_EXCLUDED_ROOTS = {"config_name", "hydra", "launch", "conda", "preprocess_cmd"}


def _iter_leaf_overrides(node: Any, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_overrides(value, path)
        return
    yield prefix, node


def _is_trainer_command_leaf(path: str) -> bool:
    root = path.split(".", 1)[0]
    return root not in CONFIG_GROUP_KEYS and root not in TRAINER_COMMAND_EXCLUDED_ROOTS


def _override_prefix(path: str) -> str:
    return "" if path.startswith("trainer.") else "++"


def _append_leaf_override(cmd: list[str], path: str, value: Any) -> None:
    cmd.append(f"{_override_prefix(path)}{path}={_hydra_value(value)}")


def _append_config_leaf_overrides(cmd: list[str], cfg: dict[str, Any]) -> None:
    for path, value in _iter_leaf_overrides(cfg):
        if _is_trainer_command_leaf(path):
            _append_leaf_override(cmd, path, value)


def _safe_suffix(value: Any) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "debug")).strip("_")
    return suffix or "debug"


def _dataset_setup_values(cfg: dict[str, Any]) -> tuple[str, int | None]:
    converted_name = str(_get(cfg, "dataset.converted_name", "flappy_train"))
    max_episodes = (
        None
        if _get(cfg, "dataset.max_episodes") in (None, "")
        else int(_get(cfg, "dataset.max_episodes"))
    )

    if _as_bool(_get(cfg, "dataset.debug_subset.enabled", False)):
        debug_max = _get(cfg, "dataset.debug_subset.max_episodes", max_episodes)
        if debug_max in (None, ""):
            debug_max = 5
        max_episodes = int(debug_max)
        suffix = _safe_suffix(_get(cfg, "dataset.debug_subset.suffix", "debug"))
        debug_suffix = "debug" if suffix == "debug" else f"debug_{suffix}"
        converted_name = f"{converted_name}__{debug_suffix}_{max_episodes}ep"

    return converted_name, max_episodes


def _env_action_dim_from_cfg(cfg: dict[str, Any]) -> int | None:
    task = str(_get(cfg, "env") or _get(cfg, "rl_games.task") or "")
    if task == "flappy":
        return 2
    if task == "demon_attack":
        return 6
    if task == "air_raid":
        return 6
    if task == "defend_the_line":
        return 6
    if task == "asterix":
        layout = str(_get(cfg, "rl_games.env_eval.asterix.action_layout", "discrete_9")).strip().lower()
        aliases = {
            "discrete_9": "discrete_9",
            "joint_9": "discrete_9",
            "asterix_discrete_9": "discrete_9",
            "factorized_6": "factorized_6",
            "factorized6": "factorized_6",
            "asterix_factorized_6": "factorized_6",
            "asterix_factorized6": "factorized_6",
        }
        layout = aliases.get(layout, layout)
        if layout == "discrete_9":
            return 9
        if layout == "factorized_6":
            return 6
        raise ValueError(f"Unsupported Asterix action layout: {layout}")
    if task == "atlantis":
        return 4
    if task == "deadly_corridor":
        layout = str(_get(cfg, "rl_games.env_eval.deadly.action_layout", "multibinary_7"))
        if layout == "multibinary_7":
            return 7
        if layout == "factorized_11":
            return 11
        raise ValueError(f"Unsupported deadly action layout: {layout}")
    return None


def _is_bridge_cfg(cfg: dict[str, Any]) -> bool:
    init_mode = str(_get(cfg, "rl_games.initialization_mode", "") or "").lower()
    action_carrier = str(_get(cfg, "rl_games.action_carrier", "") or "").lower()
    return init_mode in {"bridge", "pre-trained", "pretrained"} or action_carrier == "bridge"


def _validate_bridge_cfg(cfg: dict[str, Any], config_path: Path) -> None:
    if not _is_bridge_cfg(cfg):
        return

    initialization_repo = _get(cfg, "initialization.checkpoint_hf_repo_id")
    initialization_local_dir = _get(cfg, "initialization.checkpoint_local_dir")
    if not initialization_repo and not initialization_local_dir:
        raise ValueError(
            f"{config_path}: bridge configs must set initialization.checkpoint_local_dir "
            "or initialization.checkpoint_hf_repo_id"
        )
    initialization_filename = _get(cfg, "initialization.checkpoint_filename")
    if not initialization_filename:
        raise ValueError(f"{config_path}: bridge configs must set initialization.checkpoint_filename")
    base_repo = str(_get(cfg, "base_model.repo_id", "") or "")
    base_dir = str(_get(cfg, "paths.base_model_dir", "") or "")
    if base_repo == "StarVLA/Qwen3-VL-4B-Instruct-Action" or "Qwen3-VL-4B-Instruct-Action" in base_dir:
        raise ValueError(
            f"{config_path}: bridge mode must not use StarVLA/Qwen3-VL-4B-Instruct-Action "
            "as its base model. Use Qwen/Qwen3-VL-4B-Instruct plus a Bridge/RT-1 checkpoint "
            "under initialization.checkpoint_local_dir or initialization.checkpoint_hf_repo_id."
        )

    env_dim = _env_action_dim_from_cfg(cfg)
    if env_dim is None:
        return
    if env_dim > 7:
        raise ValueError(
            f"{config_path}: bridge configs use a 7D action carrier, but this task resolves "
            f"active action dim={env_dim}. Use native/scratch initialization, or a task layout "
            "that fits the 7D bridge carrier."
        )

    model = str(_get(cfg, "model", "") or "")
    action_dim = _get(cfg, "framework.action_model.action_dim")
    action_env_dim = _get(cfg, "framework.action_model.action_env_dim")
    state_dim = _get(cfg, "framework.action_model.state_dim")
    if action_dim not in (None, "") and int(action_dim) != 7:
        raise ValueError(f"{config_path}: bridge framework.action_model.action_dim must be 7, got {action_dim}")
    if action_env_dim not in (None, "") and int(action_env_dim) != env_dim:
        raise ValueError(
            f"{config_path}: bridge framework.action_model.action_env_dim must be {env_dim}, got {action_env_dim}"
        )
    if model == "openvla":
        if _as_bool(_get(cfg, "datasets.vla_data.include_state", False)):
            raise ValueError(f"{config_path}: openvla bridge datasets.vla_data.include_state must be false")
    else:
        if state_dim not in (None, "") and int(state_dim) != 7:
            raise ValueError(f"{config_path}: bridge framework.action_model.state_dim must be 7, got {state_dim}")
        if not _as_bool(_get(cfg, "datasets.vla_data.include_state", False)):
            raise ValueError(f"{config_path}: bridge datasets.vla_data.include_state must be true")


def _optional_int_list(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _setup_namespace(cfg: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_get(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    converted_dataset_name, max_episodes = _dataset_setup_values(cfg)
    return SimpleNamespace(
        model=str(_get(cfg, "model")),
        env=str(_get(cfg, "env")),
        mode=str(_get(cfg, "mode")),
        initialization_mode=str(_get(cfg, "rl_games.initialization_mode", "") or ""),
        action_carrier=str(_get(cfg, "rl_games.action_carrier", "") or ""),
        latency_mode=str(_get(cfg, "rl_games.env_eval.latency.mode", "") or ""),
        source_dataset_hf=str(_get(cfg, "dataset.source_hf", "") or ""),
        source_dataset_revision=_get(cfg, "dataset.revision"),
        source_dataset_config_name=(
            None
            if _get(cfg, "dataset.config_name") in (None, "")
            else str(_get(cfg, "dataset.config_name"))
        ),
        source_dataset_subdir=(
            None
            if _get(cfg, "dataset.source_subdir") in (None, "")
            else str(_get(cfg, "dataset.source_subdir"))
        ),
        dataset_local_dir=_resolve_path(_get(cfg, "paths.dataset_local_dir"), workspace_dir),
        converted_dataset_name=converted_dataset_name,
        dataset_cache_dir=(
            _resolve_path(_get(cfg, "paths.dataset_cache_dir"), workspace_dir)
            if _get(cfg, "paths.dataset_cache_dir") not in (None, "")
            else None
        ),
        dataset_force_download=str(_as_bool(_get(cfg, "dataset.force_download", False))).lower(),
        setup_force=str(_as_bool(_get(cfg, "dataset.setup_force", False))).lower(),
        skip_verification=str(_as_bool(_get(cfg, "dataset.skip_verification", False))).lower(),
        target_latency_unit=_get(cfg, "dataset.target_latency_unit", "observation_steps"),
        verify_rows=int(_get(cfg, "dataset.verify_rows", 200)),
        max_episodes=max_episodes,
        episodes_per_latency=(
            None
            if _get(cfg, "dataset.episodes_per_latency") in (None, "")
            else int(_get(cfg, "dataset.episodes_per_latency"))
        ),
        latency_filter=_optional_int_list(_get(cfg, "dataset.latency_filter")),
        cross_task=_get(cfg, "rl_games.cross_task", {}),
        base_model_dir=_resolve_path(_get(cfg, "paths.base_model_dir"), workspace_dir),
        base_model_repo_id=_get(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint_load=str(_get(cfg, "checkpoint.load", "auto")),
        checkpoint_hf_repo_id=str(_get(cfg, "checkpoint.hf_repo_id", "") or ""),
        checkpoint_save_best_model=str(_as_bool(_get(cfg, "checkpoint.save_best_model", True))).lower(),
        initialization_local_dir=(
            _resolve_path(_get(cfg, "initialization.checkpoint_local_dir"), workspace_dir)
            if _get(cfg, "initialization.checkpoint_local_dir") not in (None, "")
            else ""
        ),
        initialization_hf_repo_id=str(_get(cfg, "initialization.checkpoint_hf_repo_id", "") or ""),
        initialization_checkpoint_filename=str(_get(cfg, "initialization.checkpoint_filename", "") or ""),
        checkpoint_sync_enabled=str(_as_bool(_get(cfg, "checkpoint.sync.enabled", False))).lower(),
        checkpoint_sync_repo_id=str(_get(cfg, "checkpoint.sync.repo_id", "") or ""),
        hf_repo_id="",
    )


def _trainer_command(cfg: dict[str, Any], setup: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> list[str]:
    cmd = [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        str(_get(cfg, "config_name", "train")),
    ]
    for group_key in ("model", "env", "init", "mode"):
        value = _get(cfg, group_key)
        if value not in (None, ""):
            cmd.append(f"{group_key}={_hydra_value(value)}")
    _append_config_leaf_overrides(cmd, cfg)
    _append_leaf_override(cmd, "run_root_dir", run_root_dir)
    _append_leaf_override(cmd, "trainer.is_resume", bool(setup.get("resume_found")))
    if setup.get("resume_checkpoint"):
        _append_leaf_override(cmd, "trainer.pretrained_checkpoint", setup["resume_checkpoint"])
        _append_leaf_override(cmd, "trainer.resume_step", int(setup.get("resume_step") or 0))
    elif setup.get("pretrained_checkpoint"):
        _append_leaf_override(cmd, "trainer.pretrained_checkpoint", setup["pretrained_checkpoint"])
        _append_leaf_override(cmd, "trainer.resume_step", 0)

    data_root = setup.get("dataset_local_dir") or _resolve_path(_get(cfg, "paths.dataset_local_dir"), workspace_dir)
    _append_leaf_override(cmd, "datasets.vla_data.data_root_dir", data_root)
    if setup.get("data_mix"):
        _append_leaf_override(cmd, "datasets.vla_data.data_mix", setup["data_mix"])
    if setup.get("eval_data_mix"):
        _append_leaf_override(cmd, "datasets.vla_data.eval_data_mix", setup["eval_data_mix"])
    if setup.get("custom_mixtures_path"):
        _append_leaf_override(cmd, "datasets.vla_data.custom_mixtures_path", setup["custom_mixtures_path"])
    if setup.get("base_model_dir"):
        _append_leaf_override(cmd, "framework.qwenvl.base_vlm", setup["base_model_dir"])

    prompt_map = setup.get("latency_prompt_map_path")
    if prompt_map:
        _append_leaf_override(cmd, "rl_games.env_eval.latency.prompt_map_path", prompt_map)
    eval_tasks_cfg = _get(cfg, "rl_games.cross_task.eval_tasks", {}) or {}
    eval_task_names = {
        str(task_name)
        for task_name, task_cfg in eval_tasks_cfg.items()
        if _cross_task_eval_enabled(task_cfg)
    }
    for task_name, task_prompt_map in (setup.get("cross_task_prompt_maps") or {}).items():
        if task_name not in eval_task_names:
            continue
        _append_leaf_override(cmd, f"rl_games.cross_task.eval_tasks.{task_name}.prompt_map_path", task_prompt_map)

    return cmd


def _launch_command(cfg: dict[str, Any], trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    if str(_get(cfg, "trainer.distributed_backend", "deepspeed")).lower() == "none":
        return [sys.executable, *trainer_cmd]
    if _as_bool(_get(cfg, "launch.use_accelerate", True)):
        accelerate_config = _repo_or_workspace_path(_get(cfg, "paths.accelerate_config"), workspace_dir)
        return [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(_get(cfg, "launch.num_processes", 1)),
            *trainer_cmd,
        ]
    return [sys.executable, *trainer_cmd]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    cfg = _load_yaml(config_path)
    for override in args.overrides:
        _apply_override(cfg, override)
    _validate_bridge_cfg(cfg, config_path)

    workspace_dir = _workspace_dir(cfg)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_get(cfg, "paths.run_root_dir", "results/Checkpoints"), workspace_dir)
    _load_auth_env(cfg, workspace_dir)

    gpus = _get(cfg, "launch.gpus")
    if gpus not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    with contextlib.redirect_stdout(sys.stderr):
        setup = setup_assets(_setup_namespace(cfg, workspace_dir, run_root_dir))

    preprocess_cmd = _get(cfg, "preprocess_cmd")
    if preprocess_cmd not in (None, ""):
        subprocess.run(str(preprocess_cmd), shell=True, check=True, cwd=str(REPO_ROOT))

    trainer_cmd = _trainer_command(cfg, setup, workspace_dir, run_root_dir)
    launch_cmd = _launch_command(cfg, trainer_cmd, workspace_dir)

    print("Setup summary:")
    for key in sorted(setup):
        print(f"  {key}: {setup[key]}")
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))

    if _as_bool(_get(cfg, "launch.dry_run", False)):
        return 0
    subprocess.run(launch_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
