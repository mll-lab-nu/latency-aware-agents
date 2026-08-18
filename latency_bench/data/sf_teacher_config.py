"""Load rollout descriptors from Sample Factory teacher checkpoints."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from latency_bench.integrations.sample_factory.runtime import resolve_sample_factory_device


SF_ENV_TO_BENCH_ENV = {
    "doom_deadly_corridor": "deadly_corridor",
    "doom_defend_the_line": "defend_the_line",
    "latency_flappy": "flappy",
    "latency_demon_attack": "demon_attack",
    "deadly_corridor": "deadly_corridor",
    "defend_the_line": "defend_the_line",
    "flappy": "flappy",
    "demon_attack": "demon_attack",
}

DEFAULT_ENV_FPS_BY_BENCH_ENV = {
    "deadly_corridor": 35.0,
    "defend_the_line": 35.0,
    "flappy": 30.0,
    "demon_attack": 60.0,
}

DEFAULT_OBS_FPS_BY_BENCH_ENV = {
    "deadly_corridor": 8.75,
    "defend_the_line": 8.75,
    "flappy": 30.0,
    "demon_attack": 15.0,
}

LEGACY_SAME_LATENCY_RE = re.compile(r"(?:^|_)same_latency_l(?P<frames>\d+)(?:_|$)")


@dataclass(frozen=True)
class CheckpointRoot:
    """Resolved Sample Factory checkpoint location.

    Args:
        experiment_dir: Directory containing Sample Factory experiment files.
        checkpoint_path: Optional explicit checkpoint file.
    """
    experiment_dir: Path
    checkpoint_path: Path | None


def _infer_env_name_from_sf_env(sf_env_name: str) -> str:
    """Infer the benchmark environment name from a Sample Factory env id.

    Args:
        sf_env_name: Environment name stored in Sample Factory config.

    Returns:
        Canonical benchmark environment name.
    """
    normalized = str(sf_env_name).strip()
    if normalized in SF_ENV_TO_BENCH_ENV:
        return SF_ENV_TO_BENCH_ENV[normalized]
    raise ValueError(f"Could not infer benchmark env name from Sample Factory env {sf_env_name!r}")


def _resolve_checkpoint_root(checkpoint_root: str | Path) -> CheckpointRoot:
    """Resolve a checkpoint file or experiment directory.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.

    Returns:
        Experiment directory with optional explicit checkpoint path.
    """
    path = Path(checkpoint_root).expanduser().resolve()
    if path.is_file():
        if path.parent.name.startswith("checkpoint_p"):
            return CheckpointRoot(experiment_dir=path.parent.parent, checkpoint_path=path)
        return CheckpointRoot(experiment_dir=path.parent, checkpoint_path=path)
    return CheckpointRoot(experiment_dir=path, checkpoint_path=None)


def _read_checkpoint_config(experiment_dir: Path) -> dict[str, Any]:
    """Read Sample Factory's flat ``config.json``.

    Args:
        experiment_dir: Sample Factory experiment directory.

    Returns:
        Flat checkpoint config mapping.
    """
    config_path = experiment_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_latency_override(path: str | Path) -> dict[str, Any]:
    """Read a latency override YAML file.

    Args:
        path: YAML file containing either a latency section or latency fields.

    Returns:
        Latency config mapping.
    """
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return deepcopy(payload["latency"])


def _has_current_latency_config(flat_cfg: Mapping[str, Any]) -> bool:
    if "latency_type" not in flat_cfg:
        return False
    latency_type = flat_cfg["latency_type"]
    return latency_type is not None and str(latency_type).strip() != ""


def _env_fps(flat_cfg: Mapping[str, Any], env_name: str) -> float:
    value = flat_cfg.get("env_fps")
    if value is not None:
        return float(value)
    return DEFAULT_ENV_FPS_BY_BENCH_ENV[env_name]


def _obs_fps(flat_cfg: Mapping[str, Any], env_name: str) -> float:
    value = flat_cfg.get("obs_fps")
    if value is not None:
        return float(value)
    return DEFAULT_OBS_FPS_BY_BENCH_ENV[env_name]


def _legacy_same_latency_config(flat_cfg: Mapping[str, Any], *, env_name: str, experiment_dir: Path) -> dict[str, Any] | None:
    for name in (experiment_dir.name, str(flat_cfg["experiment"])):
        match = LEGACY_SAME_LATENCY_RE.search(name)
        if match is not None:
            latency_obs_steps = int(match.group("frames"))
            env_fps = _env_fps(flat_cfg, env_name)
            obs_fps = _obs_fps(flat_cfg, env_name)
            obs_stride_raw_frames = int(round(env_fps / obs_fps))
            return {
                "method": "fixed",
                "fixed_latency_ms": latency_obs_steps * obs_stride_raw_frames * 1000.0 / env_fps,
                "add_latency_info": False,
            }
    return None


def load_rollout_training_config(
    checkpoint_root: str | Path,
    *,
    latency_config_path: str | Path | None = None,
    device_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the rollout descriptor and metadata used for dataset export.

    Args:
        checkpoint_root: Experiment directory or checkpoint file path.
        latency_config_path: Optional latency-only YAML override.
        device_override: Optional rollout device override.

    Returns:
        Rollout descriptor and export metadata.
    """
    checkpoint = _resolve_checkpoint_root(checkpoint_root)
    flat_cfg = _read_checkpoint_config(checkpoint.experiment_dir)
    env_name = _infer_env_name_from_sf_env(str(flat_cfg["env"]))
    config_source = str((checkpoint.experiment_dir / "config.json").resolve())
    latency_override = None
    latency_source = "checkpoint"

    if latency_config_path is not None:
        latency_override_path = Path(latency_config_path).expanduser().resolve()
        latency_override = _load_latency_override(latency_override_path)
        latency_source = str(latency_override_path)
    else:
        # The same-latency checkpoint name is the experiment contract for these
        # teacher exports, and for frameskip envs lN is in observation steps.
        latency_override = _legacy_same_latency_config(
            flat_cfg,
            env_name=env_name,
            experiment_dir=checkpoint.experiment_dir,
        )
        if latency_override is not None:
            latency_source = "legacy_checkpoint_name"

    descriptor = {
        "env_name": env_name,
        "checkpoint_experiment_dir": str(checkpoint.experiment_dir),
        "checkpoint_path": None if checkpoint.checkpoint_path is None else str(checkpoint.checkpoint_path),
        "flat_cfg": deepcopy(flat_cfg),
        "device_override": None if not device_override else resolve_sample_factory_device(device_override),
        "latency_override": latency_override,
    }
    return descriptor, {
        "checkpoint_experiment_dir": str(checkpoint.experiment_dir),
        "checkpoint_path": None if checkpoint.checkpoint_path is None else str(checkpoint.checkpoint_path),
        "config_source": config_source,
        "latency_source": latency_source,
        "checkpoint_config": deepcopy(flat_cfg),
        "latency_override": deepcopy(latency_override),
    }
