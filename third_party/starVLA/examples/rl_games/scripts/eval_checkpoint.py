#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

OmegaConf = None
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parents[1]

for import_root in (REPO_ROOT, WORKSPACE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(value: str) -> list[int]:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    if "," not in text and ":" in text:
        start, end = [int(part.strip()) for part in text.split(":", 1)]
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))
    if "," not in text and "-" in text and not text.startswith("-"):
        start, end = [int(part.strip()) for part in text.split("-", 1)]
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=VALUE assignment, got {value!r}")
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key or not raw:
        raise ValueError(f"Expected TASK=VALUE assignment, got {value!r}")
    return key, raw


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str(workspace_dir / path)


def _workspace_dir(cfg, override: str | None = None) -> Path:
    configured = override or OmegaConf.select(cfg, "workspace_dir", default=None)
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    default_workspace = Path("/workspace")
    if default_workspace.exists():
        return default_workspace.resolve()
    return REPO_ROOT


def _apply_path_overrides(cfg, workspace_dir: Path, *, base_model_dir_override: str | None = None, base_model_repo_override: str | None = None) -> None:
    base_model_dir = OmegaConf.select(cfg, "paths.base_model_dir", default=None)
    if base_model_dir_override not in (None, ""):
        base_model_path = _resolve_path(base_model_dir_override, workspace_dir)
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", base_model_path, merge=True)
    elif base_model_dir not in (None, ""):
        base_model_path = _resolve_path(base_model_dir, workspace_dir)
        if Path(base_model_path).exists():
            OmegaConf.update(cfg, "framework.qwenvl.base_vlm", base_model_path, merge=True)
        else:
            base_model_repo = base_model_repo_override or OmegaConf.select(cfg, "base_model.repo_id", default=None)
            if base_model_repo not in (None, ""):
                print(
                    f"Base model path does not exist: {base_model_path}; "
                    f"falling back to HF repo {base_model_repo}"
                )
                OmegaConf.update(cfg, "framework.qwenvl.base_vlm", str(base_model_repo), merge=True)
            else:
                OmegaConf.update(cfg, "framework.qwenvl.base_vlm", base_model_path, merge=True)
    elif base_model_repo_override not in (None, ""):
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", str(base_model_repo_override), merge=True)
    dataset_root = OmegaConf.select(cfg, "paths.dataset_local_dir", default=None)
    if dataset_root not in (None, ""):
        OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", _resolve_path(dataset_root, workspace_dir), merge=True)


def _has_cross_task_eval(cfg) -> bool:
    return OmegaConf.select(cfg, "rl_games.task", default=None) == "cross_task" and OmegaConf.select(
        cfg, "rl_games.cross_task.eval_tasks", default=None
    ) is not None


def _cross_task_names(cfg) -> list[str]:
    eval_tasks = OmegaConf.select(cfg, "rl_games.cross_task.eval_tasks", default=None)
    if eval_tasks is None:
        return []
    if OmegaConf.is_dict(eval_tasks) or isinstance(eval_tasks, dict):
        return [str(key) for key in eval_tasks.keys()]
    return [str(task) for task in eval_tasks]


def _set_stage_value(cfg, stage: str, key: str, value: Any) -> None:
    OmegaConf.update(cfg, f"rl_games.env_eval.{stage}.{key}", value, merge=True)


def _set_task_stage_value(cfg, task: str, stage: str, key: str, value: Any) -> None:
    OmegaConf.update(cfg, f"rl_games.cross_task.eval_tasks.{task}.{stage}.{key}", value, merge=True)


def _apply_eval_overrides(cfg, args):
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    stage = args.stage
    if args.latencies:
        latencies = _parse_int_list(args.latencies)
        if _has_cross_task_eval(cfg):
            for task in _cross_task_names(cfg):
                _set_task_stage_value(cfg, task, stage, "latencies", latencies)
        else:
            _set_stage_value(cfg, stage, "latencies", latencies)

    if args.num_episodes is not None:
        if _has_cross_task_eval(cfg):
            for task in _cross_task_names(cfg):
                _set_task_stage_value(cfg, task, stage, "num_episodes", int(args.num_episodes))
        else:
            _set_stage_value(cfg, stage, "num_episodes", int(args.num_episodes))

    if args.max_steps_per_episode is not None:
        if _has_cross_task_eval(cfg):
            for task in _cross_task_names(cfg):
                _set_task_stage_value(cfg, task, stage, "max_steps_per_episode", int(args.max_steps_per_episode))
        else:
            _set_stage_value(cfg, stage, "max_steps_per_episode", int(args.max_steps_per_episode))

    for assignment in args.task_latencies or []:
        task, raw = _parse_assignment(assignment)
        _set_task_stage_value(cfg, task, stage, "latencies", _parse_int_list(raw))

    for assignment in args.task_num_episodes or []:
        task, raw = _parse_assignment(assignment)
        _set_task_stage_value(cfg, task, stage, "num_episodes", int(raw))

    for assignment in args.task_max_steps_per_episode or []:
        task, raw = _parse_assignment(assignment)
        _set_task_stage_value(cfg, task, stage, "max_steps_per_episode", int(raw))

    return cfg


def _print_eval_plan(cfg, stage: str) -> None:
    if _has_cross_task_eval(cfg):
        print(f"Eval stage: {stage}")
        for task in _cross_task_names(cfg):
            base = f"rl_games.cross_task.eval_tasks.{task}.{stage}"
            latencies = OmegaConf.select(cfg, f"{base}.latencies", default=None)
            episodes = OmegaConf.select(cfg, f"{base}.num_episodes", default=None)
            max_steps = OmegaConf.select(cfg, f"{base}.max_steps_per_episode", default=None)
            frameskip = OmegaConf.select(cfg, f"rl_games.cross_task.eval_tasks.{task}.frameskip", default=None)
            print(
                f"  {task}: latencies={list(latencies) if latencies is not None else None}, "
                f"num_episodes={episodes}, max_steps_per_episode={max_steps}, frameskip={frameskip}"
            )
        return
    base = f"rl_games.env_eval.{stage}"
    print(
        f"Eval stage: {stage}; "
        f"latencies={OmegaConf.select(cfg, f'{base}.latencies', default=None)}, "
        f"num_episodes={OmegaConf.select(cfg, f'{base}.num_episodes', default=None)}, "
        f"max_steps_per_episode={OmegaConf.select(cfg, f'{base}.max_steps_per_episode', default=None)}"
    )


def _flatten_eval_for_wandb(result, stage: str) -> dict:
    payload = {}
    aggregate = result.aggregate
    prefix = f"rl_games_eval/{stage}"
    payload[f"{prefix}/total_episodes"] = float(aggregate.get("total_episodes", 0))
    payload[f"{prefix}/mean_reward"] = float(aggregate.get("mean_reward", 0.0))
    payload[f"{prefix}/mean_length"] = float(aggregate.get("mean_length", 0.0))
    payload[f"{prefix}/std_reward"] = float(aggregate.get("std_reward", 0.0))
    payload[f"{prefix}/std_length"] = float(aggregate.get("std_length", 0.0))
    payload[f"{prefix}/macro_mean_reward"] = float(
        aggregate.get("macro_mean_reward", aggregate.get("mean_reward", 0.0))
    )
    payload[f"{prefix}/macro_mean_length"] = float(
        aggregate.get("macro_mean_length", aggregate.get("mean_length", 0.0))
    )
    payload[f"{prefix}/task_count"] = float(aggregate.get("task_count", 0))

    for key, metrics in result.per_latency.items():
        key_slug = key.replace("/", "__")
        payload[f"{prefix}/{key_slug}/mean_reward"] = float(metrics.get("mean_reward", 0.0))
        payload[f"{prefix}/{key_slug}/mean_length"] = float(metrics.get("mean_length", 0.0))
        payload[f"{prefix}/{key_slug}/std_reward"] = float(metrics.get("std_reward", 0.0))
        payload[f"{prefix}/{key_slug}/std_length"] = float(metrics.get("std_length", 0.0))
        payload[f"{prefix}/{key_slug}/num_episodes"] = float(metrics.get("num_episodes", 0))
    return payload


def _log_eval_to_wandb(*, args, cfg, eval_runner, result, run_dir: Path, step: int) -> None:
    import wandb

    project = args.wandb_project or str(getattr(cfg, "wandb_project", "starVLA_rl_games"))
    entity = args.wandb_entity if args.wandb_entity is not None else getattr(cfg, "wandb_entity", None)
    init_kwargs = {
        "project": project,
        "entity": entity,
        "dir": os.path.join(str(run_dir), "wandb"),
    }
    runner_cls = type(eval_runner)
    provenance = {
        "eval_backend": cfg.rl_games.env_eval.eval_backend,
        "eval_runner": f"{runner_cls.__module__}.{runner_cls.__qualname__}",
        "checkpoint_step": step,
        "image_mode": cfg.datasets.vla_data.image_mode,
        "num_obs_frames": cfg.datasets.vla_data.num_obs_frames,
        "kv_memory_enabled": cfg.framework.kv_memory.enabled,
        "attn_implementation": cfg.framework.qwenvl.attn_implementation,
        "kv_memory_packed_train": cfg.framework.kv_memory.packed_train,
    }
    if args.wandb_run_id:
        init_kwargs.update(id=args.wandb_run_id, resume="must")
    else:
        init_kwargs.update(
            name=args.wandb_run_name or f"{run_dir.name}__{args.stage}__step_{step}",
            group="rl-games-eval",
            config=provenance,
        )

    wandb_run = wandb.init(**init_kwargs)
    payload = _flatten_eval_for_wandb(result=result, stage=args.stage)
    if args.wandb_run_id:
        payload.update(
            {f"eval_provenance/{key}": value for key, value in provenance.items()}
        )
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=step)
    wandb_run.finish()


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None, step: int | None) -> tuple[Path, int]:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        if not path.exists():
            fallback_paths = [
                checkpoint_dir / Path(checkpoint).name,
                checkpoint_dir / checkpoint,
            ]
            for fallback_path in fallback_paths:
                if fallback_path.exists():
                    path = fallback_path
                    break
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}. Also checked under {checkpoint_dir}."
                )
        if path.is_dir():
            for candidate in ("model.safetensors", "pytorch_model.bin"):
                candidate_path = path / candidate
                if candidate_path.exists():
                    return candidate_path, _step_from_name(path.name)
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in checkpoint state dir: {path}")
        return path, _step_from_path(path)

    if step is not None:
        pt = checkpoint_dir / f"steps_{step}_pytorch_model.pt"
        st = checkpoint_dir / f"steps_{step}_model.safetensors"
        state = checkpoint_dir / f"steps_{step}_state"
        if pt.exists():
            return pt, step
        if st.exists():
            return st, step
        if state.exists():
            for candidate in ("model.safetensors", "pytorch_model.bin"):
                candidate_path = state / candidate
                if candidate_path.exists():
                    return candidate_path, step
        raise FileNotFoundError(f"No checkpoint found for step={step} in {checkpoint_dir}")

    candidates = []
    for state_dir in checkpoint_dir.glob("steps_*_state"):
        s = _step_from_name(state_dir.name)
        for candidate in ("model.safetensors", "pytorch_model.bin"):
            candidate_path = state_dir / candidate
            if candidate_path.exists():
                candidates.append((s, candidate_path))
                break
    for file_path in checkpoint_dir.glob("steps_*_pytorch_model.pt"):
        s = _step_from_name(file_path.name)
        candidates.append((s, file_path))
    for file_path in checkpoint_dir.glob("steps_*_model.safetensors"):
        s = _step_from_name(file_path.name)
        candidates.append((s, file_path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1], candidates[-1][0]


def _step_from_name(name: str) -> int:
    try:
        return int(name.split("steps_")[1].split("_")[0])
    except Exception:
        return 0


def _step_from_path(path: Path) -> int:
    step = _step_from_name(path.name)
    if step:
        return step
    return _step_from_name(path.parent.name)


def _load_model(cfg, checkpoint_path: Path):
    import torch
    from starVLA.model.framework.base_framework import build_framework

    model = build_framework(cfg)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def main():
    global OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--step", default=None, type=int)
    parser.add_argument("--stage", default="post_train", type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--workspace-dir", default=None, type=str, help="Workspace for resolving paths.* entries in experiment YAMLs")
    parser.add_argument("--base-model-dir", default=None, type=str, help="Explicit local base VLM directory")
    parser.add_argument("--base-model-repo-id", default=None, type=str, help="Explicit HF base VLM repo id fallback")
    parser.add_argument("--latencies", default=None, type=str, help="Latency list/range for all evaluated tasks, e.g. 0-7 or 0,1,2")
    parser.add_argument("--task-latencies", action="append", default=None, help="Per-task latency list/range, e.g. flappy=0-7")
    parser.add_argument("--num-episodes", default=None, type=int, help="Episodes per latency for all evaluated tasks")
    parser.add_argument("--max-steps-per-episode", default=None, type=int, help="Max env steps per episode for all evaluated tasks")
    parser.add_argument("--task-num-episodes", action="append", default=None, help="Per-task episodes, e.g. flappy=20")
    parser.add_argument("--task-max-steps-per-episode", action="append", default=None, help="Per-task max steps, e.g. demon_attack=3600")
    parser.add_argument("--override", action="append", default=None, help="OmegaConf dotlist override, e.g. rl_games.env_eval.post_train.latencies=[0,1]")
    parser.add_argument("--print-plan-only", action="store_true", help="Print the resolved eval plan and exit without loading a checkpoint")
    parser.add_argument("--wandb-enabled", default="true", type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    parser.add_argument("--wandb-run-id", default=None, type=str)
    args = parser.parse_args()

    from omegaconf import OmegaConf as _OmegaConf

    OmegaConf = _OmegaConf

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config else run_dir / "config.full.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    cfg = _apply_eval_overrides(cfg, args)
    workspace_dir = _workspace_dir(cfg, args.workspace_dir)
    _apply_path_overrides(
        cfg,
        workspace_dir,
        base_model_dir_override=args.base_model_dir,
        base_model_repo_override=args.base_model_repo_id,
    )

    try:
        from starVLA.model.framework.share_tools import apply_config_compat

        cfg = apply_config_compat(cfg)
    except Exception as exc:
        missing_runtime_dep = isinstance(exc, (ImportError, ModuleNotFoundError)) or "No module named" in str(exc)
        if not args.print_plan_only or not missing_runtime_dep:
            raise
        print(
            "[warn] skipping config compatibility normalization in print-plan-only "
            f"because the local environment is missing a runtime dependency: {exc}",
            file=sys.stderr,
        )
    _print_eval_plan(cfg, args.stage)
    if args.print_plan_only:
        return

    from starVLA.training.rl_games import apply_action_spec, apply_model_alias

    apply_model_alias(cfg)
    apply_action_spec(cfg)
    cfg.output_dir = str(run_dir)

    checkpoint_path, step = _resolve_checkpoint(run_dir=run_dir, checkpoint=args.checkpoint, step=args.step)
    print(f"Using checkpoint: {checkpoint_path}")

    model = _load_model(cfg=cfg, checkpoint_path=checkpoint_path)
    from starVLA.training.rl_games import build_rl_games_eval_runner

    eval_runner = build_rl_games_eval_runner(cfg=cfg, output_dir=str(run_dir))
    result = eval_runner.run(model=model, step=step, stage=args.stage)
    print(result.aggregate)

    if _str2bool(args.wandb_enabled):
        _log_eval_to_wandb(
            args=args,
            cfg=cfg,
            eval_runner=eval_runner,
            result=result,
            run_dir=run_dir,
            step=step,
        )


if __name__ == "__main__":
    main()
