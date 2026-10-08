#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint as eval_checkpoint_lib  # noqa: E402


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_stage(stage: str) -> str:
    aliases = {
        "mid_train_eval": "mid_train",
        "post_train_eval": "post_train",
    }
    return aliases.get(str(stage), str(stage))


def _parse_int_list(value: str) -> list[int]:
    return eval_checkpoint_lib._parse_int_list(value)


def _checkpoint_file_candidates(step: int) -> list[str]:
    return [
        f"steps_{int(step)}_pytorch_model.pt",
        f"steps_{int(step)}_model.safetensors",
    ]


def _checkpoint_state_candidates(step: int) -> list[str]:
    return [
        f"steps_{int(step)}_state/model.safetensors",
        f"steps_{int(step)}_state/pytorch_model.bin",
    ]


def _find_local_checkpoint(checkpoint_dir: Path, step: int) -> Path | None:
    state_dir = checkpoint_dir / f"steps_{int(step)}_state"
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        candidate_path = state_dir / candidate
        if candidate_path.exists():
            return candidate_path
    for candidate in _checkpoint_file_candidates(step):
        candidate_path = checkpoint_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _download_optional_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download from HF") from exc

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(local_dir),
        )
    except Exception:
        return None
    return Path(local_path)


def _ensure_run_config(repo_id: str, run_dir: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        path = config_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    path = run_dir / "config.full.yaml"
    if path.exists():
        return path

    for filename in ("config.full.yaml", "config.yaml"):
        downloaded = _download_optional_hf_file(repo_id=repo_id, filename=filename, local_dir=run_dir)
        if downloaded is not None and downloaded.exists():
            return downloaded

    raise FileNotFoundError(
        f"Config not found at {path}, and no config.full.yaml/config.yaml could be downloaded from {repo_id}"
    )


def _maybe_ensure_run_config(repo_id: str, run_dir: Path, config_path: Path | None) -> Path | None:
    try:
        return _ensure_run_config(repo_id=repo_id, run_dir=run_dir, config_path=config_path)
    except FileNotFoundError:
        return None


def _local_run_config(run_dir: Path) -> Path | None:
    for filename in ("config.full.yaml", "config.yaml"):
        path = run_dir / filename
        if path.exists():
            return path
    return None


def _extract_launch_train_argv(script_path: Path) -> list[str]:
    if not script_path.exists():
        raise FileNotFoundError(f"Launch script not found: {script_path}")

    command_parts: list[str] = []
    collecting = False
    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not collecting and "launch_train.py" not in line:
            continue
        collecting = True
        has_continuation = line.endswith("\\")
        command_parts.append(line[:-1].strip() if has_continuation else line)
        if not has_continuation:
            break

    if not command_parts:
        raise ValueError(f"No launch_train.py command found in {script_path}")

    tokens = shlex.split(" ".join(command_parts))
    for index, token in enumerate(tokens):
        if token.endswith("launch_train.py") or token == "launch_train.py":
            return tokens[index + 1 :]
    raise ValueError(f"Could not locate launch_train.py token in {script_path}")


def _compose_launch_cfg_from_script(args: argparse.Namespace, script_path: Path):
    from examples.rl_games.scripts import launch_train as launch_train_lib

    launch_args = launch_train_lib._parse_args(_extract_launch_train_argv(script_path))
    hydra_overrides = list(launch_args.overrides)
    if launch_args.workspace_dir not in (None, ""):
        hydra_overrides.append(f"workspace_dir={launch_args.workspace_dir}")
    if args.workspace_dir not in (None, ""):
        hydra_overrides.append(f"workspace_dir={args.workspace_dir}")
    if launch_args.run_root_dir not in (None, ""):
        hydra_overrides.append(f"paths.run_root_dir={launch_args.run_root_dir}")
        hydra_overrides.append(f"run_root_dir={launch_args.run_root_dir}")
    if launch_args.deadly_loss_type not in (None, ""):
        hydra_overrides.append(f"rl_games.deadly_corridor_loss_type={launch_args.deadly_loss_type}")
    if launch_args.eval_distributed_mode not in (None, ""):
        hydra_overrides.append(f"rl_games.env_eval.distributed_mode={launch_args.eval_distributed_mode}")
    launch_train_lib._append_cross_task_train_task_cli_overrides(hydra_overrides, launch_args)
    if args.hf_repo_id:
        hydra_overrides.append(f"checkpoint.hf_repo_id={args.hf_repo_id}")
        hydra_overrides.append(f"checkpoint.sync.repo_id={args.hf_repo_id}")
    if args.override:
        hydra_overrides.extend(args.override)

    cfg = launch_train_lib.compose_training_config(
        config_name=str(launch_args.config_name),
        model=str(launch_args.model),
        env=str(launch_args.env),
        init=str(launch_args.init),
        mode=str(launch_args.mode),
        overrides=hydra_overrides,
    )
    return cfg, launch_train_lib


def _minimal_setup_from_launch_cfg(cfg: Any, launch_train_lib: Any) -> dict[str, Any]:
    workspace_dir = launch_train_lib._workspace_dir(cfg)
    converted_name, _ = launch_train_lib._dataset_setup_values(cfg)
    action_carrier = str(launch_train_lib._cfg_get(cfg, "rl_games.action_carrier") or "")
    init_mode = str(launch_train_lib._cfg_get(cfg, "rl_games.initialization_mode") or "")
    if action_carrier not in {"native", "bridge"}:
        action_carrier = "bridge" if init_mode in {"bridge", "pre-trained", "pretrained"} else "native"
    data_mix = converted_name
    if action_carrier == "bridge" and "__bridge" not in data_mix:
        data_mix = f"{data_mix}__bridge"
    data_root = launch_train_lib._resolve_path(
        launch_train_lib._cfg_get(cfg, "paths.dataset_local_dir"),
        workspace_dir,
    )
    prompt_map_path = Path(data_root) / data_mix / "latency_prompt_map.json"
    base_model_dir = launch_train_lib._resolve_path(
        launch_train_lib._cfg_get(cfg, "paths.base_model_dir"),
        workspace_dir,
    )
    return {
        "dataset_local_dir": data_root,
        "data_mix": data_mix,
        "eval_data_mix": f"{data_mix}__val",
        "latency_prompt_map_path": str(prompt_map_path) if prompt_map_path.exists() else None,
        "base_model_dir": base_model_dir,
        "resume_found": False,
        "pretrained_checkpoint": None,
    }


def _trainer_overrides_from_command(command: list[str]) -> tuple[str, list[str]]:
    config_name = "train"
    overrides: list[str] = []
    index = 1
    while index < len(command):
        token = command[index]
        if token == "--config-name":
            config_name = command[index + 1]
            index += 2
            continue
        if token.startswith("--config-name="):
            config_name = token.split("=", 1)[1]
            index += 1
            continue
        if token.startswith("--"):
            raise ValueError(f"Unsupported train_starvla_hydra.py option in reconstructed command: {token}")
        overrides.append(token)
        index += 1
    return config_name, overrides


def _compose_trainer_cfg_from_launch_script(args: argparse.Namespace, run_dir: Path):
    from hydra import compose, initialize_config_dir

    launch_cfg, launch_train_lib = _compose_launch_cfg_from_script(
        args=args,
        script_path=Path(args.launch_script).expanduser().resolve(),
    )
    workspace_dir = launch_train_lib._workspace_dir(launch_cfg)
    run_root_dir = launch_train_lib._resolve_path(
        launch_train_lib._cfg_get(launch_cfg, "paths.run_root_dir")
        or launch_train_lib.DEFAULT_RUN_ROOT_DIR,
        workspace_dir,
    )
    setup = _minimal_setup_from_launch_cfg(launch_cfg, launch_train_lib)
    trainer_command = launch_train_lib.build_trainer_command(
        launch_cfg,
        setup,
        workspace_dir,
        run_root_dir,
    )
    config_name, trainer_overrides = _trainer_overrides_from_command(trainer_command)
    with initialize_config_dir(version_base="1.1", config_dir=str(launch_train_lib.CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=trainer_overrides)
    return cfg


def _ensure_eval_result(repo_id: str, run_dir: Path, stage: str, step: int) -> Path | None:
    local_path = run_dir / "eval" / stage / f"step_{int(step)}.json"
    if local_path.exists():
        return local_path
    return _download_optional_hf_file(
        repo_id=repo_id,
        filename=f"eval/{stage}/step_{int(step)}.json",
        local_dir=run_dir,
    )


def _saved_episode_seed_overrides(eval_result_path: Path) -> dict[str, dict[int, int | None]]:
    with eval_result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    per_latency = payload.get("per_latency", {})
    if not isinstance(per_latency, dict):
        raise ValueError(f"Expected dict `per_latency` in {eval_result_path}")

    overrides: dict[str, dict[int, int | None]] = {}
    for key, metrics in per_latency.items():
        if not isinstance(metrics, dict):
            continue
        seeds = metrics.get("episode_seeds")
        if seeds in (None, []):
            continue
        indices = metrics.get("episode_indices")
        if indices in (None, []):
            indices = list(range(len(seeds)))
        if len(indices) != len(seeds):
            raise ValueError(
                f"Saved eval result {eval_result_path} has mismatched episode_indices/episode_seeds "
                f"for {key!r}: {len(indices)} != {len(seeds)}"
            )
        overrides[str(key)] = {
            int(index): None if seed is None else int(seed)
            for index, seed in zip(indices, seeds)
        }
    return overrides


def _legacy_seed_override_aliases(
    cfg: Any,
    overrides: dict[str, dict[int, int | None]],
) -> dict[str, dict[int, int | None]]:
    """Support older eval files whose per-latency keys did not include the task name."""
    if not overrides:
        return overrides
    task = str(getattr(cfg.rl_games, "task", "flappy"))
    if task == "cross_task":
        return overrides
    aliased = dict(overrides)
    for key, value in overrides.items():
        if "/" not in key and key.startswith("latency_"):
            aliased.setdefault(f"{task}/{key}", value)
    return aliased


def _download_hf_checkpoint(repo_id: str, checkpoint_dir: Path, step: int, *, force: bool = False) -> Path:
    local_checkpoint = None if force else _find_local_checkpoint(checkpoint_dir, step)
    if local_checkpoint is not None:
        return local_checkpoint

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download checkpoints from HF") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        repo_files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        raise RuntimeError(f"Could not list HF repo files for {repo_id}: {exc}") from exc

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for candidate in _checkpoint_state_candidates(step):
        if candidate in repo_files:
            state_dir = candidate.split("/", 1)[0]
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"{state_dir}/*"],
                local_dir=str(checkpoint_dir),
                token=token,
            )
            local_checkpoint = _find_local_checkpoint(checkpoint_dir, step)
            if local_checkpoint is not None:
                return local_checkpoint

    file_candidates = set(_checkpoint_file_candidates(step))
    matching_files = [file_path for file_path in repo_files if Path(file_path).name in file_candidates]
    if matching_files:
        chosen = sorted(matching_files)[0]
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=chosen,
            local_dir=str(checkpoint_dir),
            token=token,
        )
        return Path(local_path)

    raise FileNotFoundError(
        f"No checkpoint found for step {int(step)} in HF repo {repo_id}. "
        f"Expected one of {_checkpoint_file_candidates(step)} or steps_{int(step)}_state/."
    )


def _set_stage_latencies(cfg: Any, stage: str, latencies: list[int]) -> None:
    OmegaConf = eval_checkpoint_lib.OmegaConf
    if eval_checkpoint_lib._has_cross_task_eval(cfg):
        for task in eval_checkpoint_lib._cross_task_names(cfg):
            eval_checkpoint_lib._set_task_stage_value(cfg, task, stage, "latencies", latencies)
        return
    eval_checkpoint_lib._set_stage_value(cfg, stage, "latencies", latencies)
    # The older fallback path also reads rl_games.env_eval.latency.values.
    OmegaConf.update(cfg, "rl_games.env_eval.latency.values", latencies, merge=True)


def _apply_replay_eval_overrides(args: argparse.Namespace, run_dir: Path, cfg: Any):
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.rl_games import apply_action_spec, apply_model_alias

    eval_checkpoint_lib.OmegaConf = OmegaConf
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    stage = _normalize_stage(args.stage)
    if args.latencies:
        _set_stage_latencies(cfg, stage, _parse_int_list(args.latencies))

    if args.num_episodes is not None:
        if eval_checkpoint_lib._has_cross_task_eval(cfg):
            for task in eval_checkpoint_lib._cross_task_names(cfg):
                eval_checkpoint_lib._set_task_stage_value(cfg, task, stage, "num_episodes", int(args.num_episodes))
        else:
            eval_checkpoint_lib._set_stage_value(cfg, stage, "num_episodes", int(args.num_episodes))

    if args.max_steps_per_episode is not None:
        if eval_checkpoint_lib._has_cross_task_eval(cfg):
            for task in eval_checkpoint_lib._cross_task_names(cfg):
                eval_checkpoint_lib._set_task_stage_value(
                    cfg,
                    task,
                    stage,
                    "max_steps_per_episode",
                    int(args.max_steps_per_episode),
                )
        else:
            eval_checkpoint_lib._set_stage_value(
                cfg,
                stage,
                "max_steps_per_episode",
                int(args.max_steps_per_episode),
            )

    workspace_dir = eval_checkpoint_lib._workspace_dir(cfg, args.workspace_dir)
    eval_checkpoint_lib._apply_path_overrides(
        cfg,
        workspace_dir,
        base_model_dir_override=args.base_model_dir,
        base_model_repo_override=args.base_model_repo_id,
    )
    if _str2bool(args.record_videos):
        OmegaConf.update(cfg, "rl_games.env_eval.vectorized.enabled", False, merge=True)

    cfg = apply_config_compat(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    cfg.output_dir = str(run_dir)
    return cfg


def _load_eval_config(args: argparse.Namespace, run_dir: Path, config_path: Path):
    from omegaconf import OmegaConf

    eval_checkpoint_lib.OmegaConf = OmegaConf
    cfg = OmegaConf.load(str(config_path))
    return _apply_replay_eval_overrides(args=args, run_dir=run_dir, cfg=cfg)


def _maybe_log_wandb(args: argparse.Namespace, cfg: Any, result: Any, run_dir: Path, step: int, stage: str) -> None:
    if not _str2bool(args.wandb_enabled):
        return
    import wandb

    project = args.wandb_project or str(getattr(cfg, "wandb_project", "starVLA_rl_games"))
    entity = args.wandb_entity if args.wandb_entity is not None else getattr(cfg, "wandb_entity", None)
    run_name = args.wandb_run_name or f"{run_dir.name}__{stage}__step_{step}"
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        dir=str(run_dir / "wandb"),
        group="rl-games-replay-mid-train",
    )
    wandb.log(eval_checkpoint_lib._flatten_eval_for_wandb(result=result, stage=stage), step=step)
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--hf-repo-id", required=True, type=str)
    parser.add_argument("--steps", required=True, type=str, help="Checkpoint steps, e.g. 100,200,300 or 100-500")
    parser.add_argument("--latencies", required=True, type=str, help="Latency list/range, e.g. 0,2,4 or 0-7")
    parser.add_argument("--stage", default="mid_train_eval", type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--launch-script", default=None, type=str)
    parser.add_argument("--checkpoint-dir", default=None, type=str)
    parser.add_argument("--video-output-dir", default=None, type=str)
    parser.add_argument("--video-fps", default=30, type=int)
    parser.add_argument("--record-videos", default="true", type=str)
    parser.add_argument("--use-saved-eval-seeds", default="true", type=str)
    parser.add_argument("--require-saved-eval-seeds", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--workspace-dir", default=None, type=str)
    parser.add_argument("--base-model-dir", default=None, type=str)
    parser.add_argument("--base-model-repo-id", default=None, type=str)
    parser.add_argument("--num-episodes", default=None, type=int)
    parser.add_argument("--max-steps-per-episode", default=None, type=int)
    parser.add_argument("--override", action="append", default=None)
    parser.add_argument("--print-plan-only", action="store_true")
    parser.add_argument("--wandb-enabled", default="false", type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else run_dir / "checkpoints"
    if args.config:
        config_path = _ensure_run_config(
            repo_id=args.hf_repo_id,
            run_dir=run_dir,
            config_path=Path(args.config),
        )
    elif args.launch_script:
        config_path = _local_run_config(run_dir)
    else:
        config_path = _maybe_ensure_run_config(
            repo_id=args.hf_repo_id,
            run_dir=run_dir,
            config_path=None,
        )

    if config_path is None and not args.launch_script:
        config_path = _ensure_run_config(
            repo_id=args.hf_repo_id,
            run_dir=run_dir,
            config_path=None,
        )
    stage = _normalize_stage(args.stage)
    steps = _parse_int_list(args.steps)
    if not steps:
        raise ValueError("--steps resolved to an empty list")

    if config_path is not None:
        print(f"Using saved run config: {config_path}", flush=True)
        cfg = _load_eval_config(args=args, run_dir=run_dir, config_path=config_path)
    else:
        print(f"No saved run config found; reconstructing from launch script: {args.launch_script}", flush=True)
        cfg = _compose_trainer_cfg_from_launch_script(args=args, run_dir=run_dir)
        cfg = _apply_replay_eval_overrides(args=args, run_dir=run_dir, cfg=cfg)
    eval_checkpoint_lib._print_eval_plan(cfg, stage)
    if args.print_plan_only:
        return

    # Replay intentionally stays on eval_core: it supplies video and saved-seed
    # overrides that the latency_bench runner does not consume.
    from starVLA.training.rl_games import RlGamesEvalRunner

    video_output_dir = None
    if _str2bool(args.record_videos):
        video_output_dir = args.video_output_dir or str(run_dir / "eval_videos")

    for step in steps:
        episode_seed_overrides = None
        if _str2bool(args.use_saved_eval_seeds):
            print(f"Resolving saved eval result for step {int(step)}...", flush=True)
            eval_result_path = _ensure_eval_result(
                repo_id=args.hf_repo_id,
                run_dir=run_dir,
                stage=stage,
                step=int(step),
            )
            if eval_result_path is None or not eval_result_path.exists():
                message = (
                    f"No saved eval result found for step {int(step)} at eval/{stage}/step_{int(step)}.json; "
                    "falling back to config-derived eval seeds"
                )
                if args.require_saved_eval_seeds:
                    raise FileNotFoundError(message)
                print(message)
            else:
                episode_seed_overrides = _legacy_seed_override_aliases(
                    cfg,
                    _saved_episode_seed_overrides(eval_result_path),
                )
                override_count = sum(len(value) for value in episode_seed_overrides.values())
                if override_count == 0:
                    message = (
                        f"Saved eval result {eval_result_path} has no episode_seeds; "
                        "falling back to config-derived eval seeds"
                    )
                    if args.require_saved_eval_seeds:
                        raise ValueError(message)
                    print(message)
                    episode_seed_overrides = None
                else:
                    print(
                        f"Using saved eval seeds for step {int(step)} from {eval_result_path} "
                        f"({override_count} episodes)"
                    )

        print(f"Resolving checkpoint for step {int(step)}...", flush=True)
        checkpoint_path = _download_hf_checkpoint(
            repo_id=args.hf_repo_id,
            checkpoint_dir=checkpoint_dir,
            step=int(step),
            force=bool(args.force_download),
        )
        print(f"Using checkpoint for step {int(step)}: {checkpoint_path}", flush=True)
        print(f"Loading model for step {int(step)}...", flush=True)
        model = eval_checkpoint_lib._load_model(cfg=cfg, checkpoint_path=checkpoint_path)
        runner = RlGamesEvalRunner(
            cfg=cfg,
            output_dir=str(run_dir),
            video_output_dir=video_output_dir,
            video_fps=int(args.video_fps),
        )
        result = runner.run(
            model=model,
            step=int(step),
            stage=stage,
            episode_seed_overrides=episode_seed_overrides,
        )
        print(result.aggregate)
        _maybe_log_wandb(args=args, cfg=cfg, result=result, run_dir=run_dir, step=int(step), stage=stage)
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
