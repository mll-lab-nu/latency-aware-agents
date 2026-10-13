from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "v2"
MANIFEST_V2_COLUMNS = (
    "schema_version",
    "job_id",
    "env",
    "experiment_family",
    "sweep_name",
    "latency_label",
    "seed",
    "frame_stack",
    "obs_fps",
    "observation_stride",
    "train_config",
    "eval_kind",
    "eval_config",
    "train_command",
    "eval_command",
    "run_name",
    "wandb_group",
    "wandb_name",
    "train_dir",
    "checkpoint_dir",
    "train_output_dir",
    "eval_output_dir",
    "train_artifact_dir",
    "eval_artifact_dir",
    "latency_kind",
    "latency_fixed_frames",
    "latency_mean_frames",
    "latency_std_frames",
    "latency_min_frames",
    "latency_max_frames",
)


def format_manifest_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def format_number_label(value: int | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def relative_to_repo(path: str | Path, *, repo_root: Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()
    return str(resolved.relative_to(repo_root))


def observation_stride_from_rates(*, env_fps: int | float, obs_fps: int | float) -> float:
    return float(env_fps) / float(obs_fps)


def build_job_id(
    *,
    experiment_family: str,
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int | float,
    obs_fps: int | float,
    observation_stride: int | float,
    seed: int,
) -> str:
    return ":".join(
        [
            str(experiment_family),
            str(env_name),
            str(sweep_name),
            str(latency_label),
            f"fs{format_number_label(frame_stack)}",
            f"obs{format_number_label(obs_fps)}",
            f"stride{format_number_label(observation_stride)}",
            f"seed{int(seed)}",
        ]
    )




def dump_manifest_row(row: dict[str, Any]) -> str:
    return "\t".join(format_manifest_value(row[column]) for column in MANIFEST_V2_COLUMNS)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(MANIFEST_V2_COLUMNS)]
    lines.extend(dump_manifest_row(row) for row in rows)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_config_from_command(command: str, *, cwd: Path) -> Path:
    parts = shlex.split(command)
    config_path = Path(parts[parts.index("--config") + 1]).expanduser()
    if config_path.is_absolute():
        return config_path
    return cwd / config_path


def event_log_path_for_manifest(manifest_path: str | Path) -> Path:
    return Path(manifest_path).expanduser().resolve().with_name("runner_events.jsonl")


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_events(path: str | Path) -> list[dict[str, Any]]:
    event_path = Path(path)
    if not event_path.exists():
        return []
    return [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
