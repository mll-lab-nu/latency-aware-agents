from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from scripts.experiment_runner.job_manifest import (
    build_job_id,
    build_upload_repo_path,
    format_number_label,
    observation_stride_from_rates,
    SCHEMA_VERSION,
    write_manifest,
)


def _load_legacy_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _experiment_family(manifest_path: Path) -> str:
    text = str(manifest_path)
    if "jitter_robustness" in text:
        return "jitter_robustness"
    if "jitter" in text:
        return "jitter"
    return "factor_sweep" if "factor_sweeps" in text else "same_latency"


def _latency_fields(config: dict[str, Any]) -> dict[str, str]:
    latency_cfg = config["latency"]
    latency_type = str(latency_cfg["type"])
    row = {
        "latency_kind": "",
        "latency_fixed_frames": "",
        "latency_mean_frames": "",
        "latency_std_frames": "",
        "latency_min_frames": "",
        "latency_max_frames": "",
    }
    if latency_type == "zero":
        row["latency_kind"] = "zero"
        return row
    if latency_type == "fixed":
        env_fps = float(config["env"]["env_fps"])
        row["latency_kind"] = "fixed"
        row["latency_fixed_frames"] = format_number_label(float(latency_cfg["fixed_latency_ms"]) * env_fps / 1000.0)
        return row
    if latency_type == "normal_distribution":
        row["latency_kind"] = "normal"
        row["latency_mean_frames"] = format_number_label(float(latency_cfg["mean_latency_ms"]) * float(config["env"]["env_fps"]) / 1000.0)
        row["latency_std_frames"] = format_number_label(float(latency_cfg["std_latency_ms"]) * float(config["env"]["env_fps"]) / 1000.0)
    else:
        row["latency_kind"] = "uniform"
    row["latency_min_frames"] = format_number_label(float(latency_cfg["min_latency_ms"]) * float(config["env"]["env_fps"]) / 1000.0)
    row["latency_max_frames"] = format_number_label(float(latency_cfg["max_latency_ms"]) * float(config["env"]["env_fps"]) / 1000.0)
    return row


def _migrate_row(
    *,
    row: dict[str, str],
    manifest_path: Path,
    repo_root: Path,
    upload_repo_id: str,
    upload_repo_type: str,
    upload_prefix: str,
) -> dict[str, str]:
    train_config_path = (repo_root / row["train_config"]).resolve()
    eval_config_key = "eval_config" if "eval_config" in row else "latency_bench_eval_config"
    eval_config_path = (repo_root / row[eval_config_key]).resolve()
    train_cfg = _load_yaml(train_config_path)
    eval_cfg = _load_yaml(eval_config_path)
    env_name = str(train_cfg["env"]["name"])
    env_fps = float(train_cfg["env"]["env_fps"])
    obs_fps = float(train_cfg["env"]["obs_fps"])
    observation_stride = observation_stride_from_rates(env_fps=env_fps, obs_fps=obs_fps)
    experiment_family = _experiment_family(manifest_path)
    sweep_name = row.get("sweep_name") or row.get("sweep") or "same_latency"
    latency_label = row.get("latency_label") or row.get("latency") or ""
    seed = int(row["seed"])
    job_id = build_job_id(
        experiment_family=experiment_family,
        env_name=env_name,
        sweep_name=sweep_name,
        latency_label=latency_label,
        frame_stack=int(train_cfg["env"]["frame_stack"]),
        obs_fps=obs_fps,
        observation_stride=observation_stride,
        seed=seed,
    )
    train_dir = Path(str(train_cfg["backend"]["train_dir"]))
    run_name = str(train_cfg["experiment"]["name"])
    migrated = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "env": env_name,
        "experiment_family": experiment_family,
        "sweep_name": sweep_name,
        "latency_label": latency_label,
        "seed": str(seed),
        "frame_stack": str(int(train_cfg["env"]["frame_stack"])),
        "obs_fps": format_number_label(obs_fps),
        "observation_stride": format_number_label(observation_stride),
        "train_config": row["train_config"],
        "eval_kind": "latency_bench",
        "eval_config": row[eval_config_key],
        "train_command": row["train_command"],
        "eval_command": row.get("eval_command") or row.get("latency_bench_eval_command") or "",
        "run_name": run_name,
        "wandb_group": str(train_cfg.get("logging", {}).get("wandb_group", "")),
        "wandb_name": str(train_cfg.get("logging", {}).get("wandb_name", run_name)),
        "train_dir": str(train_dir),
        "checkpoint_dir": str(train_dir / run_name / "checkpoint_p0"),
        "train_output_dir": str(train_cfg["logging"]["output_dir"]),
        "eval_output_dir": str(eval_cfg["logging"]["output_dir"]),
        "train_artifact_dir": str(train_dir / run_name),
        "eval_artifact_dir": str(eval_cfg["logging"]["output_dir"]),
        "upload_repo_id": upload_repo_id,
        "upload_repo_type": upload_repo_type,
        "train_artifact_repo_path": build_upload_repo_path(upload_prefix=upload_prefix, stage="train", job_id=job_id),
        "eval_artifact_repo_path": build_upload_repo_path(upload_prefix=upload_prefix, stage="eval", job_id=job_id),
    }
    migrated.update(_latency_fields(train_cfg))
    return migrated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy manifest.tsv files to manifest V2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload-repo-id", type=str, default="")
    parser.add_argument("--upload-repo-type", type=str, default="model")
    parser.add_argument("--upload-path-prefix", type=str, default="")
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = (args.output or input_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    rows = _load_legacy_manifest(input_path)
    migrated = [
        _migrate_row(
            row=row,
            manifest_path=input_path,
            repo_root=repo_root,
            upload_repo_id=str(args.upload_repo_id),
            upload_repo_type=str(args.upload_repo_type),
            upload_prefix=str(args.upload_path_prefix),
        )
        for row in rows
    ]
    write_manifest(output_path, migrated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
