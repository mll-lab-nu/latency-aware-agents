#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"
DEFAULT_CONFIG_NAME = "train"
DEFAULT_MODEL = "openvla"
DEFAULT_ENV = "flappy"
DEFAULT_INIT = "scratch"
DEFAULT_MODE = "single"
DEFAULT_RUN_ROOT_DIR = "results/Checkpoints"
STOP_FILE_NAME = "STOP"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.setup_training_assets import setup_assets
from starVLA.training.rl_games import apply_action_spec, apply_model_alias, validate_rl_games_config
from starVLA.training.rl_games.auth import login_training_services


def _cfg_get(cfg: Any, path: str) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, path)
    node = cfg
    for part in path.split("."):
        if node is None:
            return None
        if isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
            continue
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _as_bool_default(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    return _as_bool(value)


def _cross_task_eval_enabled(task_cfg: Any) -> bool:
    if OmegaConf.is_config(task_cfg):
        task_cfg = OmegaConf.to_container(task_cfg, resolve=True)
    if not isinstance(task_cfg, dict):
        return True
    mid_train = task_cfg.get("mid_train") or {}
    post_train = task_cfg.get("post_train") or {}
    return _as_bool(mid_train.get("enabled", True)) or _as_bool(post_train.get("enabled", True))


def _default_if_empty(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    return value


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((workspace_dir / path).resolve())


def _repo_or_workspace_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return str(repo_path)
    return str((workspace_dir / path).resolve())


def _workspace_dir(cfg: Any) -> Path:
    configured = _cfg_get(cfg, "workspace_dir")
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    return REPO_ROOT


def _csv_int_list_expr(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return "[" + ",".join(str(int(item)) for item in value) + "]"


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if isinstance(value, str) and any(ch.isspace() or ch in {",", ":", "{", "}", "[", "]"} for ch in value):
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


CONFIG_GROUP_KEYS = {"model", "env", "init", "mode"}
TRAINER_COMMAND_EXCLUDED_ROOTS = {
    "config_name",
    "hydra",
    "launch",
    "conda",
    "preprocess_cmd",
    "wandb_entity",
    "wandb_project",
}


def _iter_leaf_overrides(node: Any, prefix: str = ""):
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=False)
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_overrides(value, path)
        return
    yield prefix, node


def _is_trainer_command_leaf(path: str) -> bool:
    root = path.split(".", 1)[0]
    return root not in CONFIG_GROUP_KEYS and root not in TRAINER_COMMAND_EXCLUDED_ROOTS


def _append_leaf_override(cmd: list[str], path: str, value: Any) -> None:
    cmd.append(f"++{path}={_hydra_value(value)}")


def _append_config_leaf_overrides(cmd: list[str], cfg: Any) -> None:
    for path, value in _iter_leaf_overrides(cfg):
        if _is_trainer_command_leaf(path):
            if isinstance(cfg, DictConfig):
                value = OmegaConf.select(cfg, path)
            _append_leaf_override(cmd, path, value)


def _append_cross_task_train_task_cli_overrides(hydra_overrides: list[str], cli_args: argparse.Namespace) -> None:
    for role, index in (("a", 0), ("b", 1)):
        prefix = f"cross_task_{role}"
        hydra_prefix = f"rl_games.cross_task.train_tasks.{index}"
        for arg_name, cfg_name in (
            ("name", "name"),
            ("train_source_hf", "train_source_hf"),
            ("prompt_source_hf", "prompt_source_hf"),
            ("converted_name", "converted_name"),
            ("episodes_per_latency", "episodes_per_latency"),
            ("max_episodes", "max_episodes"),
            ("weight", "weight"),
        ):
            value = getattr(cli_args, f"{prefix}_{arg_name}", None)
            if value not in (None, ""):
                hydra_overrides.append(f"{hydra_prefix}.{cfg_name}={_hydra_value(value)}")

        latencies = _csv_int_list_expr(getattr(cli_args, f"{prefix}_latencies", None))
        if latencies:
            hydra_overrides.append(f"{hydra_prefix}.train_latency_filter={latencies}")


def _safe_suffix(value: Any) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "debug")).strip("_")
    return suffix or "debug"


def _dataset_setup_values(cfg: Any) -> tuple[str, int | None]:
    converted_name = str(_cfg_get(cfg, "dataset.converted_name") or "flappy_train")
    max_episodes_value = _cfg_get(cfg, "dataset.max_episodes")
    max_episodes = None if max_episodes_value in (None, "") else int(max_episodes_value)

    if _as_bool(_cfg_get(cfg, "dataset.debug_subset.enabled")):
        debug_max = _cfg_get(cfg, "dataset.debug_subset.max_episodes")
        if debug_max in (None, ""):
            debug_max = max_episodes if max_episodes is not None else 5
        max_episodes = int(debug_max)
        suffix = _safe_suffix(_cfg_get(cfg, "dataset.debug_subset.suffix"))
        debug_suffix = "debug" if suffix == "debug" else f"debug_{suffix}"
        converted_name = f"{converted_name}__{debug_suffix}_{max_episodes}ep"

    return converted_name, max_episodes


def _optional_int_list(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _optional_latency_episode_map(value: Any) -> dict[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        pairs: dict[int, int] = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "dataset.episodes_per_latency_by_latency must use 'latency:episodes' items, "
                    f"got {value!r}"
                )
            latency, episodes = item.split(":", 1)
            pairs[int(latency.strip())] = int(episodes.strip())
        return pairs or None
    raise TypeError(f"Unsupported dataset.episodes_per_latency_by_latency={value!r}")


def _setup_eval_latencies(cfg: Any) -> list[int] | None:
    if not _as_bool_default(_cfg_get(cfg, "rl_games.env_eval.enabled"), True):
        return None
    latencies: set[int] = set()
    for stage in ("mid_train", "post_train"):
        if not _as_bool_default(_cfg_get(cfg, f"rl_games.env_eval.{stage}.enabled"), True):
            continue
        if _as_bool_default(_cfg_get(cfg, f"rl_games.env_eval.{stage}.latencies_from_prompt_map"), False):
            continue
        latencies.update(_optional_int_list(_cfg_get(cfg, f"rl_games.env_eval.{stage}.latencies")) or [])
    return sorted(latencies) or None


def _task_action_layout(cfg: Any) -> str:
    task = str(_cfg_get(cfg, "env") or _cfg_get(cfg, "rl_games.task") or "")
    if task == "deadly_corridor":
        return str(_cfg_get(cfg, "rl_games.env_eval.deadly.action_layout") or "")
    if task == "asterix":
        return str(_cfg_get(cfg, "rl_games.env_eval.asterix.action_layout") or "")
    return ""


def _checkpoint_request(load_value: Any) -> tuple[str, str]:
    raw_value = str(load_value or "auto")
    if raw_value in {"auto", "none", "local", "hf"}:
        return raw_value, ""
    return "none", raw_value


def compose_training_config(
    config_name: str,
    model: str,
    env: str,
    init: str,
    mode: str,
    overrides: Sequence[str],
) -> DictConfig:
    compose_overrides = [
        f"model={model}",
        f"env={env}",
        f"init={init}",
        f"mode={mode}",
        *overrides,
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=compose_overrides)
    validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    return cfg


def setup_namespace_from_cfg(cfg: Any, workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_cfg_get(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    checkpoint_load, checkpoint = _checkpoint_request(_cfg_get(cfg, "checkpoint.load"))
    converted_dataset_name, max_episodes = _dataset_setup_values(cfg)

    dataset_cache_dir = _cfg_get(cfg, "paths.dataset_cache_dir")
    initialization_local_dir = _cfg_get(cfg, "initialization.checkpoint_local_dir")
    cross_task_cfg = _cfg_get(cfg, "rl_games.cross_task")
    cross_task = (
        OmegaConf.to_container(cross_task_cfg, resolve=True)
        if OmegaConf.is_config(cross_task_cfg)
        else (cross_task_cfg or {})
    )
    gymnasium_task_contract_cfg = _cfg_get(cfg, "rl_games.gymnasium.task_contract")
    gymnasium_task_contract = (
        OmegaConf.to_container(gymnasium_task_contract_cfg, resolve=True)
        if OmegaConf.is_config(gymnasium_task_contract_cfg)
        else gymnasium_task_contract_cfg
    )

    return SimpleNamespace(
        model=str(_cfg_get(cfg, "model")),
        env=str(_cfg_get(cfg, "env")),
        mode=str(_cfg_get(cfg, "mode")),
        initialization_mode=str(_cfg_get(cfg, "rl_games.initialization_mode") or ""),
        action_carrier=str(_cfg_get(cfg, "rl_games.action_carrier") or ""),
        action_layout=_task_action_layout(cfg),
        deadly_action_layout=str(_cfg_get(cfg, "rl_games.env_eval.deadly.action_layout") or ""),
        latency_mode=str(_cfg_get(cfg, "rl_games.env_eval.latency.mode") or ""),
        eval_latencies=_setup_eval_latencies(cfg),
        source_dataset_hf=str(_cfg_get(cfg, "dataset.source_hf") or ""),
        source_dataset_revision=_cfg_get(cfg, "dataset.revision"),
        source_dataset_config_name=(
            None
            if _cfg_get(cfg, "dataset.config_name") in (None, "")
            else str(_cfg_get(cfg, "dataset.config_name"))
        ),
        source_dataset_subdir=(
            None
            if _cfg_get(cfg, "dataset.source_subdir") in (None, "")
            else str(_cfg_get(cfg, "dataset.source_subdir"))
        ),
        dataset_local_dir=_resolve_path(_cfg_get(cfg, "paths.dataset_local_dir"), workspace_dir),
        converted_dataset_name=converted_dataset_name,
        dataset_cache_dir=(
            _resolve_path(dataset_cache_dir, workspace_dir)
            if dataset_cache_dir not in (None, "")
            else None
        ),
        dataset_force_download=str(_as_bool(_cfg_get(cfg, "dataset.force_download"))).lower(),
        setup_force=str(_as_bool(_cfg_get(cfg, "dataset.setup_force"))).lower(),
        skip_verification=str(_as_bool(_cfg_get(cfg, "dataset.skip_verification"))).lower(),
        target_latency_unit=_cfg_get(cfg, "dataset.target_latency_unit"),
        verify_rows=int(_cfg_get(cfg, "dataset.verify_rows") or 200),
        max_episodes=max_episodes,
        max_steps_per_episode=(
            None
            if _cfg_get(cfg, "dataset.max_steps_per_episode") in (None, "")
            else int(_cfg_get(cfg, "dataset.max_steps_per_episode"))
        ),
        episodes_per_latency=(
            None
            if _cfg_get(cfg, "dataset.episodes_per_latency") in (None, "")
            else int(_cfg_get(cfg, "dataset.episodes_per_latency"))
        ),
        episodes_per_latency_by_latency=_optional_latency_episode_map(
            _cfg_get(cfg, "dataset.episodes_per_latency_by_latency")
        ),
        latency_filter=_optional_int_list(_cfg_get(cfg, "dataset.latency_filter")),
        base_model_dir=_resolve_path(_cfg_get(cfg, "paths.base_model_dir"), workspace_dir),
        base_model_repo_id=_cfg_get(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint=checkpoint,
        checkpoint_load=checkpoint_load,
        checkpoint_hf_repo_id=str(_cfg_get(cfg, "checkpoint.hf_repo_id") or ""),
        checkpoint_save_best_model=str(_as_bool_default(_cfg_get(cfg, "checkpoint.save_best_model"), True)).lower(),
        initialization_local_dir=(
            _resolve_path(initialization_local_dir, workspace_dir)
            if initialization_local_dir not in (None, "")
            else ""
        ),
        initialization_hf_repo_id=str(_cfg_get(cfg, "initialization.checkpoint_hf_repo_id") or ""),
        initialization_checkpoint_filename=str(_cfg_get(cfg, "initialization.checkpoint_filename") or ""),
        cross_task=cross_task,
        gymnasium_task_contract=gymnasium_task_contract,
        checkpoint_sync_enabled=str(_as_bool(_cfg_get(cfg, "checkpoint.sync.enabled"))).lower(),
        checkpoint_sync_repo_id=str(_cfg_get(cfg, "checkpoint.sync.repo_id") or ""),
        hf_repo_id="",
    )


def build_trainer_command(cfg: Any, setup: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> list[str]:
    cmd = [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        str(_cfg_get(cfg, "config_name") or DEFAULT_CONFIG_NAME),
        f"model={_cfg_get(cfg, 'model')}",
        f"env={_cfg_get(cfg, 'env')}",
        f"init={_cfg_get(cfg, 'init')}",
        f"mode={_cfg_get(cfg, 'mode')}",
    ]
    _append_config_leaf_overrides(cmd, cfg)
    _append_leaf_override(cmd, "run_root_dir", run_root_dir)
    _append_leaf_override(cmd, "trainer.is_resume", bool(setup.get("resume_found")))

    if setup.get("resume_checkpoint"):
        _append_leaf_override(cmd, "trainer.pretrained_checkpoint", setup["resume_checkpoint"])
        _append_leaf_override(cmd, "trainer.resume_step", int(setup.get("resume_step") or 0))
    elif setup.get("pretrained_checkpoint"):
        _append_leaf_override(cmd, "trainer.pretrained_checkpoint", setup["pretrained_checkpoint"])
        _append_leaf_override(cmd, "trainer.resume_step", 0)

    data_root = setup.get("dataset_local_dir") or _resolve_path(_cfg_get(cfg, "paths.dataset_local_dir"), workspace_dir)
    _append_leaf_override(cmd, "datasets.vla_data.data_root_dir", data_root)

    if setup.get("data_mix"):
        _append_leaf_override(cmd, "datasets.vla_data.data_mix", setup["data_mix"])
    if setup.get("eval_data_mix"):
        _append_leaf_override(cmd, "datasets.vla_data.eval_data_mix", setup["eval_data_mix"])
    if setup.get("custom_mixtures_path"):
        _append_leaf_override(cmd, "datasets.vla_data.custom_mixtures_path", setup["custom_mixtures_path"])
    if setup.get("base_model_dir"):
        _append_leaf_override(cmd, "framework.qwenvl.base_vlm", setup["base_model_dir"])
        if _cfg_get(cfg, "framework.world_model.base_wm") not in (None, ""):
            _append_leaf_override(cmd, "framework.world_model.base_wm", setup["base_model_dir"])

    prompt_map = setup.get("latency_prompt_map_path")
    if prompt_map not in (None, ""):
        _append_leaf_override(cmd, "rl_games.env_eval.latency.prompt_map_path", prompt_map)
    eval_tasks_cfg = _cfg_get(cfg, "rl_games.cross_task.eval_tasks") or {}
    if OmegaConf.is_config(eval_tasks_cfg):
        eval_tasks_cfg = OmegaConf.to_container(eval_tasks_cfg, resolve=True)
    eval_task_names = {
        str(task_name)
        for task_name, task_cfg in eval_tasks_cfg.items()
        if _cross_task_eval_enabled(task_cfg)
    } if isinstance(eval_tasks_cfg, dict) else set()
    for task_name, task_prompt_map in (setup.get("cross_task_prompt_maps") or {}).items():
        if task_name not in eval_task_names:
            continue
        _append_leaf_override(cmd, f"rl_games.cross_task.eval_tasks.{task_name}.prompt_map_path", task_prompt_map)

    return cmd


def build_launch_command(cfg: Any, trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    distributed_backend = str(_cfg_get(cfg, "trainer.distributed_backend") or "deepspeed").lower()
    if distributed_backend == "none":
        return [sys.executable, *trainer_cmd]

    if _as_bool(_cfg_get(cfg, "launch.use_accelerate")):
        accelerate_config = _repo_or_workspace_path(_cfg_get(cfg, "paths.accelerate_config"), workspace_dir)
        return [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(_cfg_get(cfg, "launch.num_processes") or 1),
            *trainer_cmd,
        ]

    return [sys.executable, *trainer_cmd]


def graceful_run_launch_command(launch_cmd: list[str], *, cwd: Path, stop_file: Path, env: dict[str, str] | None = None) -> int:
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()

    child_env = dict(os.environ if env is None else env)
    child_env["STARVLA_STOP_FILE"] = str(stop_file)
    interrupt_count = 0
    process = subprocess.Popen(
        launch_cmd,
        cwd=str(cwd),
        env=child_env,
        start_new_session=True,
    )

    def _request_stop(signum, frame):
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            stop_file.write_text(
                f"signal={signal.Signals(signum).name}\ntime={time.time():.6f}\n",
                encoding="utf-8",
            )
            print(
                f"\nGraceful stop requested; wrote {stop_file}. "
                "Waiting for the trainer to stop at a safe step boundary. "
                "Press Ctrl-C again to send SIGTERM.",
                file=sys.stderr,
                flush=True,
            )
            return
        if interrupt_count == 2:
            print("\nSecond interrupt received; sending SIGTERM to training process group.", file=sys.stderr, flush=True)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        print("\nThird interrupt received; sending SIGKILL to training process group.", file=sys.stderr, flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        return process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--run-root-dir", default=None)
    parser.add_argument("--deadly-loss-type", default=None, choices=("l1", "multibinary_bce", "multibinary_ce"))
    parser.add_argument("--eval-distributed-mode", default=None, choices=("none", "rank_sharded"))
    for role in ("a", "b"):
        parser.add_argument(f"--cross-task-{role}-name", default=None)
        parser.add_argument(f"--cross-task-{role}-train-source-hf", f"--cross-task-{role}-source-hf", default=None)
        parser.add_argument(f"--cross-task-{role}-prompt-source-hf", default=None)
        parser.add_argument(f"--cross-task-{role}-converted-name", default=None)
        parser.add_argument(f"--cross-task-{role}-latencies", default=None)
        parser.add_argument(f"--cross-task-{role}-episodes-per-latency", type=int, default=None)
        parser.add_argument(f"--cross-task-{role}-max-episodes", type=int, default=None)
        parser.add_argument(f"--cross-task-{role}-weight", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = _parse_args(sys.argv[1:] if argv is None else argv)

    hydra_overrides = list(cli_args.overrides)
    if cli_args.workspace_dir not in (None, ""):
        hydra_overrides.append(f"workspace_dir={cli_args.workspace_dir}")
    if cli_args.run_root_dir not in (None, ""):
        hydra_overrides.append(f"paths.run_root_dir={cli_args.run_root_dir}")
        hydra_overrides.append(f"run_root_dir={cli_args.run_root_dir}")
    if cli_args.deadly_loss_type not in (None, ""):
        hydra_overrides.append(f"rl_games.deadly_corridor_loss_type={cli_args.deadly_loss_type}")
    if cli_args.eval_distributed_mode not in (None, ""):
        hydra_overrides.append(f"rl_games.env_eval.distributed_mode={cli_args.eval_distributed_mode}")
    _append_cross_task_train_task_cli_overrides(hydra_overrides, cli_args)
    if cli_args.dry_run:
        hydra_overrides.append("launch.dry_run=true")

    cfg = compose_training_config(
        config_name=str(cli_args.config_name),
        model=str(cli_args.model),
        env=str(cli_args.env),
        init=str(cli_args.init),
        mode=str(cli_args.mode),
        overrides=hydra_overrides,
    )

    workspace_dir = _workspace_dir(cfg)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_cfg_get(cfg, "paths.run_root_dir") or DEFAULT_RUN_ROOT_DIR, workspace_dir)

    login_training_services(cfg, workspace_dir=workspace_dir, repo_root=REPO_ROOT)

    gpus = _cfg_get(cfg, "launch.gpus")
    if gpus not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    # wandb_entity/wandb_project are launcher-only overrides (see
    # TRAINER_COMMAND_EXCLUDED_ROOTS): the trainer subprocess resolves them via
    # ${oc.env:WANDB_ENTITY}/${oc.env:WANDB_PROJECT} instead of a CLI override,
    # so export whatever value was passed here into the child's environment.
    wandb_entity = _cfg_get(cfg, "wandb_entity")
    if wandb_entity not in (None, ""):
        os.environ["WANDB_ENTITY"] = str(wandb_entity)
    wandb_project = _cfg_get(cfg, "wandb_project")
    if wandb_project not in (None, ""):
        os.environ["WANDB_PROJECT"] = str(wandb_project)

    with contextlib.redirect_stdout(sys.stderr):
        setup = setup_assets(setup_namespace_from_cfg(cfg, workspace_dir, run_root_dir))

    preprocess_cmd = _cfg_get(cfg, "preprocess_cmd")
    if preprocess_cmd not in (None, ""):
        subprocess.run(str(preprocess_cmd), shell=True, check=True, cwd=str(REPO_ROOT))

    trainer_cmd = build_trainer_command(cfg, setup, workspace_dir, run_root_dir)
    launch_cmd = build_launch_command(cfg, trainer_cmd, workspace_dir)
    stop_file = Path(run_root_dir) / str(_cfg_get(cfg, "run_id")) / STOP_FILE_NAME

    print("Setup summary:")
    for key in sorted(setup):
        print(f"  {key}: {setup[key]}")
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))

    if _as_bool(_cfg_get(cfg, "launch.dry_run")):
        return 0

    returncode = graceful_run_launch_command(launch_cmd, cwd=REPO_ROOT, stop_file=stop_file)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, launch_cmd)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
