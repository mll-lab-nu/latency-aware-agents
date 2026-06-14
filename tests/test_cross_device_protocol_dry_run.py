from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.cross_device.models import RuntimeSelection
from scripts.cross_device.runtime import normalize_hardware_name


PROTOCOL_MODULE = "scripts.cross_device.protocol"
REAL_PROTOCOL = Path("configs/experiments/cross_device_protocol.yaml")


RUNTIME_SPEC = {
    "runtime_envs": {
        "small_model": {"conda_env": "latency", "description": "small"},
        "openvla": {"conda_env": "starvla_rl_games_openvla", "description": "vla"},
    }
}


def _write_base_config(path: Path) -> None:
    config = {
        "experiment": {"name": "dry_run_policy_seed0", "seed": 0},
        "executor": {"mode": "simulated"},
        "env": {"name": "flappy", "env_fps": 30, "obs_fps": 30, "noop_action": "noop"},
        "latency": {"type": "fixed", "fixed_latency_ms": 33.333333333333336},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": "random", "actions": ["noop", "flap"]},
        "logging": {"output_dir": "unused"},
        "evaluation": {"eval_episodes": 2, "eval_parallel_envs": 4, "eval_latency_values": [0, 1], "eval_max_steps": 10},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_profile_manifest(path: Path, *, env_fps: float = 16.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profile_id": "L_A100",
                "policy_id": "dry-policy",
                "env_id": "flappy",
                "target_device": "a100",
                "env_fps": env_fps,
                "frame_ms": 1000.0 / env_fps,
                "n_samples": 3,
                "trace_path": "latencies.jsonl",
                "distribution_path": "latency_distribution.json",
                "plot_path": "latency_distribution.png",
                "summary": {"mean_ms": 125.0, "n_samples": 3},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_base_config_with_policy(path: Path, policy_type: str) -> None:
    config = {
        "experiment": {"name": f"{policy_type}_seed0", "seed": 0},
        "executor": {"mode": "simulated"},
        "env": {"name": "dummy", "env_fps": 30, "obs_fps": 30, "noop_action": "noop"},
        "latency": {"type": "zero"},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": policy_type, "actions": ["noop"]},
        "logging": {"output_dir": "unused"},
        "evaluation": {"eval_episodes": 1, "eval_parallel_envs": 1, "eval_latency_values": None, "eval_max_steps": 1},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_two_row_realtime_spec(path: Path, first_config: Path, second_config: Path) -> None:
    first = {
        "policy": "failing-policy",
        "env": "Dummy",
        "policy_type": "unknown_policy",
        "model_type": "test_model",
        "runtime_env": "small_model",
        "env_name": "dummy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(first_config),
    }
    second = {
        "policy": "passing-policy",
        "env": "Dummy",
        "policy_type": "random",
        "model_type": "test_model",
        "runtime_env": "small_model",
        "env_name": "dummy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(second_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 1,
        "seed_set": [0],
        "runtime_envs": {"small_model": {"conda_env": "latency", "description": "small"}},
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            }
        ],
        "experiments": {
            "exp2_same_device_calibration": {"panel": "b", "rows": [first, second]},
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _write_spec(path: Path, base_config: Path) -> None:
    row = {
        "policy": "dry-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "runtime_env": "small_model",
        "env_name": "flappy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 2,
        "seed_set": [0],
        "runtime_envs": {
            "small_model": {"conda_env": "latency", "description": "small"},
            "openvla": {"conda_env": "starvla_rl_games_openvla", "description": "vla"},
        },
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            }
        ],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a", "rows": [dict(row)]},
            "exp2_same_device_calibration": {"panel": "b", "rows": [dict(row)]},
            "exp3_cross_device_prediction": {
                "panel": "c",
                "rows": [dict(row)],
                "replay_pairs": [{"target_device": "a100", "replay_device": "a100"}],
            },
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _write_mixed_spec(path: Path, base_config: Path) -> None:
    small = {
        "policy": "small-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "runtime_env": "small_model",
        "env_name": "flappy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    vla = {
        "policy": "vla-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "runtime_env": "openvla",
        "env_name": "flappy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 2,
        "seed_set": [0],
        "runtime_envs": {
            "small_model": {"conda_env": "latency", "description": "small"},
            "openvla": {"conda_env": "starvla_rl_games_openvla", "description": "vla"},
        },
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            },
            {
                "id": "h200",
                "display_name": "H200",
                "profile_id": "L_H200",
                "aliases": ["NVIDIA H200"],
            },
            {
                "id": "rtx4090",
                "display_name": "RTX 4090",
                "profile_id": "L_4090",
                "aliases": ["NVIDIA GeForce RTX 4090"],
            },
        ],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a", "rows": [dict(vla)]},
            "exp2_same_device_calibration": {"panel": "b", "rows": [dict(small)]},
            "exp3_cross_device_prediction": {
                "panel": "c",
                "rows": [dict(vla)],
                "replay_pairs": [
                    {"target_device": "a100", "replay_device": "h200"},
                    {"target_device": "a100", "replay_device": "a100"},
                ],
            },
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _run_cli(tmp_path: Path, *args: str, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.setdefault("CROSS_DEVICE_RUNTIME_ENV", "latency")
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", PROTOCOL_MODULE, *args],
        cwd=Path.cwd(),
        check=check,
        capture_output=True,
        text=True,
        env=run_env,
    )


def _load_protocol_module():
    return importlib.import_module(PROTOCOL_MODULE)


def test_runtime_env_detection_maps_conda_envs_and_override(monkeypatch):
    module = _load_protocol_module()
    monkeypatch.delenv("CROSS_DEVICE_RUNTIME_ENV", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "latency")
    assert module.detect_runtime_env(RUNTIME_SPEC).matched_alias == "small_model"

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "starvla_rl_games_openvla")
    assert module.detect_runtime_env(RUNTIME_SPEC).matched_alias == "openvla"

    monkeypatch.setenv("CROSS_DEVICE_RUNTIME_ENV", "latency")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "starvla_rl_games_openvla")
    detected = module.detect_runtime_env(RUNTIME_SPEC)
    assert detected.detected_raw == "latency"
    assert detected.matched_alias == "small_model"


def test_hardware_detection_accepts_full_gpu_model_names_and_multi_gpu_text():
    devices = [
        {
            "id": "rtx3090",
            "display_name": "RTX 3090",
            "profile_id": "L_3090",
            "aliases": ["NVIDIA GeForce RTX 3090"],
        },
        {
            "id": "rtx4090",
            "display_name": "RTX 4090",
            "profile_id": "L_4090",
            "aliases": ["NVIDIA GeForce RTX 4090"],
        },
        {
            "id": "l40s",
            "display_name": "L40S",
            "profile_id": "L_L40S",
            "aliases": ["NVIDIA L40S"],
        },
        {
            "id": "a100",
            "display_name": "A100",
            "profile_id": "L_A100",
            "aliases": ["NVIDIA A100"],
        },
        {
            "id": "h200",
            "display_name": "H200",
            "profile_id": "L_H200",
            "aliases": ["NVIDIA H200"],
        },
        {
            "id": "b200",
            "display_name": "B200",
            "profile_id": "L_B200",
            "aliases": ["NVIDIA B200"],
        },
    ]

    cases = {
        "NVIDIA A100-SXM4-80GB": "a100",
        "NVIDIA A100-SXM4-40GB": "a100",
        "NVIDIA A100-SXM4-80GB; NVIDIA A100-SXM4-80GB": "a100",
        "NVIDIA L40S": "l40s",
        "NVIDIA GeForce RTX 3090": "rtx3090",
        "NVIDIA GeForce RTX 4090": "rtx4090",
        "NVIDIA H200": "h200",
        "NVIDIA B200": "b200",
    }
    for raw_name, expected_id in cases.items():
        assert normalize_hardware_name(raw_name, devices)["id"] == expected_id


def test_runtime_env_filtering_small_and_openvla_rows(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    small = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
    )
    assert "policy: small-policy" in small.stdout
    assert "policy: vla-policy" in small.stdout
    assert "policy: small-policy\n" in small.stdout
    assert "runtime_env mismatch" in small.stdout

    openvla = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )
    assert "policy: vla-policy" in openvla.stdout
    assert "skipped runtime_env mismatch: 1" in openvla.stdout


def test_real_openvla_protocol_dry_run_selects_flappy_demon_and_deadly(tmp_path: Path):
    out = tmp_path / "results"

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert "policy: openvla-flappy" in result.stdout
    assert "policy: demon-attack-openvla" in result.stdout
    assert "policy: deadly-corridor-openvla" in result.stdout
    assert "openvla-flappy/flappy" in result.stdout
    assert "demon-attack-openvla/demon_attack" in result.stdout
    assert "deadly-corridor-openvla/deadly_corridor" in result.stdout


def test_real_protocol_exp2_includes_openvla_calibration_rows():
    spec = yaml.safe_load(REAL_PROTOCOL.read_text(encoding="utf-8"))

    rows = spec["experiments"]["exp2_same_device_calibration"]["rows"]
    openvla_rows = {
        row["policy"]: row
        for row in rows
        if row.get("runtime_env") == "openvla"
    }

    assert set(openvla_rows) == {
        "openvla-flappy",
        "demon-attack-openvla",
        "deadly-corridor-openvla",
    }
    assert openvla_rows["openvla-flappy"]["policy_type"] == "starvla"
    assert openvla_rows["openvla-flappy"]["model_type"] == "openvla"
    assert openvla_rows["openvla-flappy"]["fps"] == 16
    assert openvla_rows["openvla-flappy"]["base_config"] == (
        "configs/experiments/flappy/realtime/starvla_openvla_fix_latency_2_16fps_l2_2gpu_warmup200_3ep.yaml"
    )
    assert openvla_rows["demon-attack-openvla"]["policy_type"] == "starvla"
    assert openvla_rows["demon-attack-openvla"]["model_type"] == "openvla"
    assert openvla_rows["demon-attack-openvla"]["base_config"] == (
        "configs/experiments/demon_attack/starvla_fixed_l2.yaml"
    )
    assert openvla_rows["deadly-corridor-openvla"]["policy_type"] == "starvla"
    assert openvla_rows["deadly-corridor-openvla"]["model_type"] == "openvla"
    assert openvla_rows["deadly-corridor-openvla"]["base_config"] == (
        "configs/experiments/deadly_corridor/starvla_single_latency.yaml"
    )


def test_real_protocol_exp3_uses_explicit_replay_pairs():
    spec = yaml.safe_load(REAL_PROTOCOL.read_text(encoding="utf-8"))

    replay_pairs = spec["experiments"]["exp3_cross_device_prediction"]["replay_pairs"]

    assert replay_pairs == [
        {"target_device": "rtx3090", "replay_device": "b200"},
        {"target_device": "rtx3090", "replay_device": "h200"},
        {"target_device": "rtx4090", "replay_device": "b200"},
        {"target_device": "rtx4090", "replay_device": "h200"},
        {"target_device": "rtx5090", "replay_device": "b200"},
        {"target_device": "rtx5090", "replay_device": "h200"},
        {"target_device": "a6000", "replay_device": "b200"},
        {"target_device": "a6000", "replay_device": "h200"},
        {"target_device": "l40s", "replay_device": "b200"},
        {"target_device": "l40s", "replay_device": "h200"},
        {"target_device": "a100", "replay_device": "b200"},
        {"target_device": "a100", "replay_device": "h200"},
    ]


def test_real_openvla_protocol_only_policy_accepts_slug_and_preserves_flappy_path(tmp_path: Path):
    out = tmp_path / "results"

    flappy = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-policy",
        "openvla-flappy",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )
    assert "policy: openvla-flappy" in flappy.stdout
    assert "policy: demon-attack-openvla" not in flappy.stdout
    assert "policy: deadly-corridor-openvla" not in flappy.stdout
    assert "openvla-flappy/flappy" in flappy.stdout

    demon = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-policy",
        "demon-attack-openvla",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )
    assert "policy: openvla-flappy" not in demon.stdout
    assert "policy: demon-attack-openvla" in demon.stdout
    assert "policy: deadly-corridor-openvla" not in demon.stdout


def test_real_openvla_protocol_only_env_accepts_raw_and_slug_values(tmp_path: Path):
    out = tmp_path / "results"

    slug = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-env",
        "deadly_corridor",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )
    assert "policy: deadly-corridor-openvla" in slug.stdout
    assert "policy: openvla-flappy" not in slug.stdout
    assert "policy: demon-attack-openvla" not in slug.stdout

    raw = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-env",
        "Deadly Corridor",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )
    assert "policy: deadly-corridor-openvla" in raw.stdout
    assert "policy: openvla-flappy" not in raw.stdout
    assert "policy: demon-attack-openvla" not in raw.stdout


def test_real_openvla_protocol_max_runs_limits_dry_run_rows(tmp_path: Path):
    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(tmp_path / "results"),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert result.stdout.count("policy: ") == 1


def test_starvla_openvla_dry_run_uses_starvla_config_metadata(tmp_path: Path):
    out = tmp_path / "results"

    _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-policy",
        "demon-attack-openvla",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["type"] == "starvla"
    assert "base_model_path" not in config["policy"]
    assert config["policy"]["checkpoint_path"] == "results/Checkpoints/openvla_demon_attack_fixed_l2/final_model"
    assert "eval_suites" not in config["evaluation"]


def test_openvla_smoke_dry_run_uses_isolated_run_id_and_step_cap(tmp_path: Path):
    out = tmp_path / "results"

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--smoke",
        "--smoke-episodes",
        "1",
        "--smoke-max-steps",
        "7",
        "--only-policy",
        "demon-attack-openvla",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert "smoke: true" in result.stdout
    assert "episodes: 1" in result.stdout
    assert "eval_max_steps: 7" in result.stdout
    assert "smoke_seed_0_n1" in result.stdout
    assert "profiles_smoke/demon-attack-openvla/demon_attack" in result.stdout
    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["evaluation"]["eval_episodes"] == 1
    assert config["evaluation"]["eval_max_steps"] == 7
    assert "smoke_seed_0_n1" in config["logging"]["output_dir"]
    assert not (out / "runs").exists()


def test_flappy_starvla_dry_run_uses_starvla_checkpoint_config(tmp_path: Path):
    out = tmp_path / "results"

    _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-policy",
        "openvla-flappy",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["type"] == "starvla"
    assert "base_model_path" not in config["policy"]
    assert config["policy"]["checkpoint_path"] == (
        "/mnt/data/latency_data/openvla_flappy_fix_latency_2/checkpoints/steps_5000_pytorch_model.pt"
    )


def test_deadly_openvla_dry_run_writes_factorized_eleven_dim_config(tmp_path: Path):
    out = tmp_path / "results"

    _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--dry-run",
        "--only-policy",
        "deadly-corridor-openvla",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["type"] == "starvla"
    assert config["env"]["obs_fps"] == 8.75
    assert "eval_suites" not in config["evaluation"]


def test_runtime_env_all_safety_and_strict_mode(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    dry_run = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "all",
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
    )
    assert "policy: small-policy" in dry_run.stdout
    assert "policy: vla-policy" in dry_run.stdout

    unsafe = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "all",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
        check=False,
    )
    assert unsafe.returncode != 0
    assert "--runtime-env all is only allowed" in unsafe.stderr

    strict = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--strict-runtime-env",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
        check=False,
    )
    assert strict.returncode != 0
    assert "--strict-runtime-env failed" in strict.stderr


def test_doctor_output_reports_runtime_compatible_and_skipped_rows(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "doctor",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
    )

    assert "Runtime environment:" in result.stdout
    assert "matched alias: small_model" in result.stdout
    assert "compatible rows: small-policy" in result.stdout
    assert "Skipped rows:" in result.stdout
    assert "vla-policy" in result.stdout
    assert "runtime_env" in result.stdout


def test_measure_dedups_duplicate_openvla_rows_across_experiments(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--runtime-env",
        "openvla",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert result.stdout.count("policy: vla-policy") == 1
    assert "experiment_ids: exp1_latency_profile_summary,exp3_cross_device_prediction" in result.stdout


def test_sim_same_dry_run_writes_guarded_generated_config_without_eval(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    profile_path = out / "profiles" / "dry-policy" / "flappy" / "L_A100" / "profile.json"
    _write_profile_manifest(profile_path, env_fps=16.0)

    result = _run_cli(
        tmp_path,
        "sim-same",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
    )

    assert "-m latency_bench.run" in result.stdout
    generated = next((out / "generated_configs" / "sim_same").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["executor"]["mode"] == "simulated"
    assert config["env"]["env_fps"] == 16.0
    assert config["env"]["obs_fps"] == 16.0
    assert config["latency"]["type"] == "profile_sample"
    assert config["latency"]["profile_path"].endswith("profiles/dry-policy/flappy/L_A100/profile.json")
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1
    assert not (out / "runs").exists()


def test_sim_same_uses_hf_profile_cache_before_skip(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("scripts.cross_device.planning")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    profiles_dir = out / "profiles"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    runtime = RuntimeSelection(
        requested="small_model",
        detected_raw="latency",
        matched_alias="small_model",
        matched_conda_env="latency",
        selected_alias="small_model",
        allow_all=False,
        allow_mismatch=False,
    )
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_experiment="exp2_same_device_calibration",
        only_policy=None,
        seed_set=None,
        smoke=False,
        smoke_max_steps=None,
    )
    calls: list[Path] = []

    def ensure_profile_cached(*, profile_path: Path, profiles_dir: Path, hf_repo: str, **kwargs):
        calls.append(profile_path)
        _write_profile_manifest(profile_path, env_fps=16.0)
        return profile_path

    monkeypatch.setenv("HF_PROFILE_REPO", "latency-sensitive-bench/profiles")
    monkeypatch.setattr(planning, "_ensure_profile_cached", ensure_profile_cached)

    plan = planning._build_plan(
        args,
        spec,
        "sim_same",
        out,
        profiles_dir,
        current_device,
        runtime,
    )

    assert [item.runnable for item in plan] == [True]
    assert calls == [profiles_dir / "dry-policy" / "flappy" / "L_A100" / "profile.json"]


def test_measure_dry_run_prints_eval_and_profile_commands(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
    )

    assert "-m latency_bench.run" in result.stdout
    assert "-m latency_bench.experiments.build_latency_profile_from_trace" in result.stdout
    assert "--env-fps 30" in result.stdout
    assert "--metadata-json" in result.stdout
    assert '"policy_type": "random"' in result.stdout
    assert "--n-warmup" not in result.stdout
    assert "latencies.jsonl" in result.stdout
    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    warmup_generated = next((out / "generated_configs" / "measure").rglob("*.warmup.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    warmup_config = yaml.safe_load(warmup_generated.read_text(encoding="utf-8"))
    assert config["executor"]["mode"] == "realtime"
    assert warmup_config["executor"]["mode"] == "realtime"
    assert str(warmup_generated) in result.stdout
    assert "type" not in config["latency"]
    assert config["latency"]["device"] == "cuda:0"
    assert config["latency"]["sync_cuda"] is True
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1


def test_measure_dry_run_threads_n_samples_to_profile_command(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
        "--n-episodes",
        "3",
        "--n-samples",
        "5",
    )

    assert "episodes: 3" in result.stdout
    assert "samples: 5" in result.stdout
    assert "--n-samples 5" in result.stdout
    assert "best_seed_0_n3_s5" in result.stdout


def test_measure_run_calibrates_env_fps_before_formal_eval(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("scripts.cross_device.planning")
    runner = importlib.import_module("scripts.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    runtime = RuntimeSelection(
        requested="small_model",
        detected_raw="latency",
        matched_alias="small_model",
        matched_conda_env="latency",
        selected_alias="small_model",
        allow_all=False,
        allow_mismatch=False,
    )
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_experiment="exp2_same_device_calibration",
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(args, spec, "measure", out, out / "profiles", current_device, runtime)[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_warmup" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "selected_env_fps": 16.0,
                        "selected_obs_fps": 16.0,
                        "target_latency_frames": 1,
                        "mean_latency_ms": 62.5,
                        "hit_count": 3,
                        "sample_count": 3,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        elif "latency_bench.run" in command:
            (task.output_dir / "latencies.jsonl").write_text('{"latency_ms": 62.5}\n', encoding="utf-8")
        elif "latency_bench.experiments.build_latency_profile_from_trace" in command:
            profile_output = Path(command[command.index("--output-dir") + 1])
            _write_profile_manifest(profile_output / "profile.json", env_fps=16.0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, runtime, args, out)

    assert [command[2] for command in commands] == [
        "latency_bench.experiments.calibrate_env_fps_from_warmup",
        "latency_bench.run",
        "latency_bench.experiments.build_latency_profile_from_trace",
    ]
    final_config = yaml.safe_load(task.generated_config_path.read_text(encoding="utf-8"))
    assert final_config["env"]["env_fps"] == 16.0
    assert final_config["env"]["obs_fps"] == 16.0
    assert "--env-fps" in commands[2]
    assert commands[2][commands[2].index("--env-fps") + 1] == "16.0"


def test_measure_run_continues_batches_after_samples_until_n_episodes(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("scripts.cross_device.planning")
    runner = importlib.import_module("scripts.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    runtime = RuntimeSelection(
        requested="small_model",
        detected_raw="latency",
        matched_alias="small_model",
        matched_conda_env="latency",
        selected_alias="small_model",
        allow_all=False,
        allow_mismatch=False,
    )
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=3,
        n_samples=2,
        only_device=None,
        only_env=None,
        only_experiment="exp2_same_device_calibration",
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(args, spec, "measure", out, out / "profiles", current_device, runtime)[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_warmup" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"selected_env_fps": 16.0, "selected_obs_fps": 16.0}),
                encoding="utf-8",
            )
        elif "latency_bench.run" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("latencies.jsonl").write_text(
                "".join(json.dumps({"latency_ms": sample}) + "\n" for sample in [10.0, 20.0]),
                encoding="utf-8",
            )
            output_dir.joinpath("episode_metrics.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "episode_return_env": 100.0 + episode_id,
                            "survival_steps": 10 + episode_id,
                        }
                    ) + "\n"
                    for episode_id in range(3)
                ),
                encoding="utf-8",
            )
        elif "latency_bench.experiments.build_latency_profile_from_trace" in command:
            profile_output = Path(command[command.index("--output-dir") + 1])
            _write_profile_manifest(profile_output / "profile.json", env_fps=16.0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, runtime, args, out)

    eval_commands = [command for command in commands if "latency_bench.run" in command]
    profile_command = commands[-1]
    assert len(eval_commands) == 1
    assert "--n-samples" in eval_commands[0]
    assert eval_commands[0][eval_commands[0].index("--n-samples") + 1] == "2"
    assert "--n-samples" in profile_command
    assert profile_command[profile_command.index("--n-samples") + 1] == "2"
    assert (task.output_dir / "latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 2
    assert (task.output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").count("episode_return_env") == 3


def test_measure_run_continues_batches_after_episodes_until_n_samples(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("scripts.cross_device.planning")
    runner = importlib.import_module("scripts.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    runtime = RuntimeSelection(
        requested="small_model",
        detected_raw="latency",
        matched_alias="small_model",
        matched_conda_env="latency",
        selected_alias="small_model",
        allow_all=False,
        allow_mismatch=False,
    )
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=2,
        n_samples=5,
        only_device=None,
        only_env=None,
        only_experiment="exp2_same_device_calibration",
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(args, spec, "measure", out, out / "profiles", current_device, runtime)[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_warmup" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"selected_env_fps": 16.0, "selected_obs_fps": 16.0}),
                encoding="utf-8",
            )
        elif "latency_bench.run" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("latencies.jsonl").write_text(
                "".join(
                    json.dumps({"latency_ms": sample}) + "\n"
                    for sample in [10.0, 20.0, 30.0, 40.0, 50.0]
                ),
                encoding="utf-8",
            )
            output_dir.joinpath("episode_metrics.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "episode_return_env": 100.0 + episode_id,
                            "survival_steps": 10 + episode_id,
                        }
                    ) + "\n"
                    for episode_id in range(2)
                ),
                encoding="utf-8",
            )
        elif "latency_bench.experiments.build_latency_profile_from_trace" in command:
            profile_output = Path(command[command.index("--output-dir") + 1])
            _write_profile_manifest(profile_output / "profile.json", env_fps=16.0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, runtime, args, out)

    eval_commands = [command for command in commands if "latency_bench.run" in command]
    profile_command = commands[-1]
    assert len(eval_commands) == 1
    assert "--n-samples" in eval_commands[0]
    assert eval_commands[0][eval_commands[0].index("--n-samples") + 1] == "5"
    assert "--n-samples" in profile_command
    assert profile_command[profile_command.index("--n-samples") + 1] == "5"
    assert (task.output_dir / "latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 5
    assert (task.output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").count("episode_return_env") == 2


def test_measure_run_records_failed_calibration_manifest_without_eval(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("scripts.cross_device.planning")
    runner = importlib.import_module("scripts.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    runtime = RuntimeSelection(
        requested="small_model",
        detected_raw="latency",
        matched_alias="small_model",
        matched_conda_env="latency",
        selected_alias="small_model",
        allow_all=False,
        allow_mismatch=False,
    )
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_experiment="exp2_same_device_calibration",
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(args, spec, "measure", out, out / "profiles", current_device, runtime)[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        raise subprocess.CalledProcessError(returncode=3, cmd=command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        runner._run_task(task, spec, {"raw": "a100", "device": current_device}, runtime, args, out)

    assert [command[2] for command in commands] == ["latency_bench.experiments.calibrate_env_fps_from_warmup"]
    manifest = json.loads((task.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "latency_bench.experiments.calibrate_env_fps_from_warmup" in manifest["calibration_command"]
    assert manifest["warmup_generated_config_path"].endswith(".warmup.yaml")
    assert manifest["calibration_output_path"].endswith("env_fps_calibration.json")
    records = [
        json.loads(line)
        for line in (out / "manifests" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["calibration_command"] == manifest["calibration_command"]


def test_measure_dry_run_uses_base_config_fps_not_protocol_row_fps(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    base_data = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    base_data["env"]["env_fps"] = 60
    base_data["env"]["obs_fps"] = 15
    base_config.write_text(yaml.safe_dump(base_data, sort_keys=False), encoding="utf-8")
    _write_spec(spec, base_config)
    spec_data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    row = spec_data["experiments"]["exp2_same_device_calibration"]["rows"][0]
    row.update(
        {
            "policy": "deadly-corridor-sf-l1",
            "env": "Deadly Corridor",
            "policy_type": "deadly_corridor_sf",
            "model_type": "sample_factory",
            "env_name": "deadly_corridor",
            "fps": 999,
        }
    )
    spec.write_text(yaml.safe_dump(spec_data, sort_keys=False), encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["env"]["name"] == "deadly_corridor"
    assert config["env"]["env_fps"] == 60
    assert config["env"]["obs_fps"] == 15
    assert "--env-fps 60" in result.stdout


def test_stage_overrides_can_raise_deadly_corridor_measure_episodes_only(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    spec_data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    spec_data["n_episodes"] = 10
    row = spec_data["experiments"]["exp2_same_device_calibration"]["rows"][0]
    row.update(
        {
            "policy": "deadly-corridor-sf-l1",
            "env": "Deadly Corridor",
            "policy_type": "deadly_corridor_sf",
            "env_name": "deadly_corridor",
            "fps": 35,
            "stage_overrides": {"measure": {"n_episodes": 1000}},
        }
    )
    spec.write_text(yaml.safe_dump(spec_data, sort_keys=False), encoding="utf-8")

    measure_out = tmp_path / "measure_results"
    _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(measure_out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
    )
    measure_config = yaml.safe_load(next((measure_out / "generated_configs" / "measure").rglob("*.yaml")).read_text())
    assert measure_config["evaluation"]["eval_episodes"] == 1000
    assert "best_seed_0_n1000" in measure_config["logging"]["output_dir"]


def test_cli_n_episodes_overrides_stage_overrides(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    spec_data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    spec_data["experiments"]["exp2_same_device_calibration"]["rows"][0]["stage_overrides"] = {
        "measure": {"n_episodes": 1000}
    }
    spec.write_text(yaml.safe_dump(spec_data, sort_keys=False), encoding="utf-8")

    _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
        "--n-episodes",
        "20",
    )

    config = yaml.safe_load(next((out / "generated_configs" / "measure").rglob("*.yaml")).read_text())
    assert config["evaluation"]["eval_episodes"] == 20
    assert "best_seed_0_n20" in config["logging"]["output_dir"]


def test_continue_on_error_runs_later_tasks_and_returns_nonzero_summary(tmp_path: Path):
    failing_config = tmp_path / "failing.yaml"
    passing_config = tmp_path / "passing.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config_with_policy(failing_config, "unknown_policy")
    _write_base_config_with_policy(passing_config, "random")
    _write_two_row_realtime_spec(spec, failing_config, passing_config)
    _write_profile_manifest(out / "profiles" / "failing-policy" / "dummy" / "L_A100" / "profile.json", env_fps=30.0)
    _write_profile_manifest(out / "profiles" / "passing-policy" / "dummy" / "L_A100" / "profile.json", env_fps=30.0)

    result = _run_cli(
        tmp_path,
        "sim-same",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--profiles-dir",
        str(out / "profiles"),
        "--runtime-env",
        "small_model",
        "--continue-on-error",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "FAILED RUNS:" in combined
    assert "policy=failing-policy" in combined
    assert "env=Dummy" in combined
    assert "stage=sim_same" in combined
    assert "output_dir=" in combined
    assert "error=" in combined
    assert "passing-policy" in combined
    failed_manifest = next((out / "runs" / "sim_same" / "failing-policy").rglob("manifest.json"))
    manifest = yaml.safe_load(failed_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert next((out / "runs" / "sim_same" / "passing-policy").rglob("manifest.json")).exists()


def test_resume_skips_completed_manifest_but_retries_failed_run(tmp_path: Path):
    failing_config = tmp_path / "failing.yaml"
    passing_config = tmp_path / "passing.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config_with_policy(failing_config, "unknown_policy")
    _write_base_config_with_policy(passing_config, "random")
    _write_two_row_realtime_spec(spec, failing_config, passing_config)
    _write_profile_manifest(out / "profiles" / "failing-policy" / "dummy" / "L_A100" / "profile.json", env_fps=30.0)
    _write_profile_manifest(out / "profiles" / "passing-policy" / "dummy" / "L_A100" / "profile.json", env_fps=30.0)

    completed_dir = out / "runs" / "sim_same" / "passing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "manifest.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    failed_dir = out / "runs" / "sim_same" / "failing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "manifest.json").write_text('{"status": "failed", "error": "old"}\n', encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "sim-same",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--profiles-dir",
        str(out / "profiles"),
        "--runtime-env",
        "small_model",
        "--resume",
        "--continue-on-error",
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency"},
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert f"resume: skipping completed run {completed_dir}" in combined
    assert f"resume: skipping completed run {failed_dir}" not in combined
    retried_manifest = yaml.safe_load((failed_dir / "manifest.json").read_text(encoding="utf-8"))
    assert retried_manifest["status"] == "failed"
    assert retried_manifest["error"] != "old"


def test_aggregate_dry_run_accepts_common_filters_without_writing_tables(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    root = tmp_path / "results"
    tables = root / "tables"
    _write_base_config(base_config)
    _write_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "aggregate",
        "--spec",
        str(spec),
        "--root",
        str(root),
        "--out",
        str(tables),
        "--dry-run",
        "--only-device",
        "a100",
        "--seed-set",
        "0,1",
    )

    assert "would_write:" in result.stdout
    assert "experiment3_cross_device_prediction.csv" in result.stdout
    assert not tables.exists()


def test_aggregate_dry_run_accepts_raw_root(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    root = tmp_path / "results" / "cross_device"
    raw_root = tmp_path / "results" / "raw" / "raw"
    tables = root / "tables"
    _write_base_config(base_config)
    _write_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "aggregate",
        "--spec",
        str(spec),
        "--root",
        str(root),
        "--raw-root",
        str(raw_root),
        "--out",
        str(tables),
        "--dry-run",
    )

    assert f"raw_root: {raw_root}" in result.stdout
    assert "would_write:" in result.stdout
    assert not tables.exists()


def test_sim_cross_skips_replay_hardware_not_in_explicit_pair(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "sim-cross",
        "--spec",
        str(spec),
        "--replay-hardware",
        "rtx4090",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        "--runtime-env",
        "openvla",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert "runnable rows: 0" in result.stdout
    assert "replay hardware mismatch" in result.stdout
