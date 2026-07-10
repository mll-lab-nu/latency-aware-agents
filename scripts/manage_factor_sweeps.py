from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_runner.job_manifest import load_manifest, train_config_from_command
from latency_bench.core.config import load_config


STATUS_COLUMNS = (
    "status",
    "train_config",
    "train_command",
    "experiment",
    "wandb_name",
    "wandb_group",
    "latest_env_steps",
    "target_env_steps",
    "checkpoint_dir",
    "final_eval_result",
    "reason",
)


def collect_status_rows(
    generated_root: str | Path,
    *,
    active_grace_seconds: float = 300.0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for train_command, train_config in _iter_train_commands(Path(generated_root)):
        config = load_config(train_config)
        rows.append(
            _status_row(
                config,
                train_config=train_config,
                train_command=train_command,
                active_grace_seconds=active_grace_seconds,
            )
        )
    return rows


def _iter_train_commands(generated_root: Path):
    manifest_files = sorted(generated_root.expanduser().rglob("manifest.tsv"))
    if manifest_files:
        for manifest_file in manifest_files:
            for row in load_manifest(manifest_file):
                yield row["train_command"], Path(row["train_config"]).expanduser()
        return
    for command_file in sorted(generated_root.expanduser().rglob("train_commands.txt")):
        for line in command_file.read_text(encoding="utf-8").splitlines():
            command = line.strip()
            if command:
                yield command, train_config_from_command(command, cwd=Path.cwd())


def _status_row(
    config: dict[str, Any],
    *,
    train_config: Path,
    train_command: str,
    active_grace_seconds: float,
) -> dict[str, str]:
    experiment = str(config["experiment"]["name"])
    train_dir = Path(str(config["backend"]["train_dir"])).expanduser()
    output_dir = Path(str(config["logging"]["output_dir"])).expanduser()
    checkpoint_dir = train_dir / experiment / "checkpoint_p0"
    latest_checkpoint = _latest_checkpoint(checkpoint_dir)
    latest_env_steps = _checkpoint_env_steps(latest_checkpoint) if latest_checkpoint is not None else None
    target_env_steps = int(config["training"]["train_for_env_steps"])
    final_eval_result = (
        _final_eval_result(output_dir, latest_env_steps) if latest_env_steps is not None else None
    )

    status, reason = _classify_status(
        train_dir / experiment,
        checkpoint_dir,
        latest_checkpoint,
        latest_env_steps,
        target_env_steps,
        final_eval_result,
        active_grace_seconds=active_grace_seconds,
    )
    logging_cfg = config.get("logging", {})
    return {
        "status": status,
        "train_config": str(train_config),
        "train_command": train_command,
        "experiment": experiment,
        "wandb_name": str(logging_cfg.get("wandb_name") or experiment),
        "wandb_group": str(logging_cfg.get("wandb_group") or ""),
        "latest_env_steps": "" if latest_env_steps is None else str(latest_env_steps),
        "target_env_steps": str(target_env_steps),
        "checkpoint_dir": str(checkpoint_dir),
        "final_eval_result": "" if final_eval_result is None else str(final_eval_result),
        "reason": reason,
    }


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _checkpoint_env_steps(checkpoint_path: Path) -> int:
    return int(checkpoint_path.stem.rsplit("_", 1)[1])


def _final_eval_result(output_dir: Path, env_steps: int) -> Path:
    return output_dir / "latency_bench_train_eval" / f"step_{int(env_steps)}_final" / "eval_result.json"


def _classify_status(
    experiment_dir: Path,
    checkpoint_dir: Path,
    latest_checkpoint: Path | None,
    latest_env_steps: int | None,
    target_env_steps: int,
    final_eval_result: Path | None,
    *,
    active_grace_seconds: float,
) -> tuple[str, str]:
    if not experiment_dir.exists():
        return "missing", "checkpoint experiment directory is missing"
    if latest_env_steps is not None and latest_env_steps >= target_env_steps:
        if final_eval_result is not None and final_eval_result.is_file():
            return "complete", "training and final eval outputs are present"
        if _recent_activity(experiment_dir, checkpoint_dir, latest_checkpoint, active_grace_seconds):
            return "running", "training reached target and final eval may still be running"
        return "eval_incomplete", "training reached target but final eval output is missing"
    if _recent_activity(experiment_dir, checkpoint_dir, latest_checkpoint, active_grace_seconds):
        return "running", "recent checkpoint/log activity"
    return "train_corrupt", "training output exists but target env steps were not reached"


def _recent_activity(
    experiment_dir: Path,
    checkpoint_dir: Path,
    latest_checkpoint: Path | None,
    active_grace_seconds: float,
) -> bool:
    if active_grace_seconds <= 0:
        return False
    threshold = time.time() - float(active_grace_seconds)
    candidates = [
        experiment_dir / "sf_log.txt",
        experiment_dir / "episode_metrics.jsonl",
        latest_checkpoint,
        checkpoint_dir,
    ]
    return any(path is not None and path.exists() and path.stat().st_mtime >= threshold for path in candidates)


def print_status(rows: list[dict[str, str]], *, jsonl: bool) -> None:
    if jsonl:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return
    print("\t".join(STATUS_COLUMNS))
    for row in rows:
        print("\t".join(row[column] for column in STATUS_COLUMNS))


def pending_commands(rows: list[dict[str, str]]) -> list[str]:
    return [
        row["train_command"]
        for row in rows
        if row["status"] in {"missing", "eval_incomplete"}
    ]


def repair_train(
    rows: list[dict[str, str]],
    *,
    train_config: str | None,
    wandb_name: str | None,
    wandb_entity: str | None = None,
) -> None:
    row = _select_repair_row(rows, train_config=train_config, wandb_name=wandb_name)
    config = load_config(row["train_config"])
    experiment = str(config["experiment"]["name"])
    experiment_dir = Path(str(config["backend"]["train_dir"])).expanduser() / experiment
    output_dir = Path(str(config["logging"]["output_dir"])).expanduser()
    shutil.rmtree(experiment_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    _delete_wandb_run(config, wandb_entity=wandb_entity)


def _select_repair_row(rows: list[dict[str, str]], *, train_config: str | None, wandb_name: str | None) -> dict[str, str]:
    for row in rows:
        if train_config is not None and Path(row["train_config"]) == Path(train_config).expanduser():
            return row
        if wandb_name is not None and row["wandb_name"] == wandb_name:
            return row
    raise SystemExit("No matching run found")


def _delete_wandb_run(config: dict[str, Any], *, wandb_entity: str | None) -> None:
    logging_cfg = config.get("logging", {})
    entity = wandb_entity or logging_cfg.get("wandb_entity") or logging_cfg.get("wandb_user")
    project = logging_cfg.get("wandb_project")
    run_id = logging_cfg.get("wandb_name") or config["experiment"]["name"]
    if not entity:
        print(f"Skipping W&B delete for {run_id}: no wandb entity configured")
        return
    import wandb

    wandb.Api().run(f"{entity}/{project}/{run_id}").delete()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and repair generated factor sweep training runs.")
    parser.add_argument("--generated-root", type=Path, default=Path("configs/train/factor_sweeps"))
    parser.add_argument("--active-grace-seconds", type=float, default=300.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print run status table.")
    status.add_argument("--jsonl", action="store_true")

    subparsers.add_parser("pending", help="Print training commands for missing or eval-incomplete runs.")

    repair = subparsers.add_parser("repair-train", help="Delete local corrupt train outputs and W&B run.")
    repair.add_argument("--config")
    repair.add_argument("--wandb-name")
    repair.add_argument("--wandb-entity")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = collect_status_rows(args.generated_root, active_grace_seconds=args.active_grace_seconds)
    if args.command == "status":
        print_status(rows, jsonl=args.jsonl)
    elif args.command == "pending":
        for command in pending_commands(rows):
            print(command)
    elif args.command == "repair-train":
        repair_train(rows, train_config=args.config, wandb_name=args.wandb_name, wandb_entity=args.wandb_entity)


if __name__ == "__main__":
    main()
