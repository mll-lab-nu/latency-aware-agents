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

from latency_bench.cross_device.runtime import (
    detect_runtime_env_raw,
    normalize_hardware_name,
    runtime_env_mismatch,
)


PROTOCOL_MODULE = "latency_bench.cross_device.protocol"
REAL_PROTOCOL = Path("configs/experiments/cross_device_protocol.yaml")


RUNTIME_SPEC = {"runtime_env": {"name": "baseline", "conda_env": "latency"}}


def _first_runtime_generated_config(out: Path, stage: str) -> Path:
    return next(
        path
        for path in (out / "generated_configs" / stage).rglob("*.yaml")
        if not path.name.endswith(".calibration.yaml")
    )


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


def _write_profile_manifest(path: Path, *, env_fps: float = 16.0, inference_worker_count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_id": "L_A100",
        "policy_id": "dry-policy",
        "env_id": "flappy",
        "target_device": "a100",
        "env_fps": env_fps,
        "frame_ms": 1000.0 / env_fps,
        "n_samples": 3,
        "trace_path": "e2e_latencies.jsonl",
        "latency_kind": "e2e_latency",
        "inference_trace_path": "infer_latencies.jsonl",
        "distribution_path": "latency_distribution.json",
        "plot_path": "latency_distribution.png",
        "summary": {"mean_ms": 125.0, "mean_effective_frames": 2.0, "n_samples": 3},
        "inference_summary": {"mean_ms": 62.5, "mean_effective_frames": 1.0, "n_samples": 3},
        "inference_worker_count": inference_worker_count,
    }
    path.write_text(
        json.dumps(
            payload,
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
        "env_name": "dummy",
        "target_latency_frames": 1,
        "base_config": str(first_config),
    }
    second = {
        "policy": "passing-policy",
        "env": "Dummy",
        "policy_type": "random",
        "model_type": "test_model",
        "env_name": "dummy",
        "target_latency_frames": 1,
        "base_config": str(second_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 1,
        "sim_worker_count": "auto",
        "seed_set": [0],
        "env_fps_calibration": {"worker_utilization_cap": 0.8},
        "runtime_env": {"name": "baseline", "conda_env": "latency"},
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            }
        ],
        "rows": [first, second],
        "experiments": {
            "exp2_same_device_calibration": {"panel": "b"},
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _write_spec(path: Path, base_config: Path) -> None:
    row = {
        "policy": "dry-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "env_name": "flappy",
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 2,
        "sim_worker_count": "auto",
        "seed_set": [0],
        "env_fps_calibration": {"worker_utilization_cap": 0.8},
        "runtime_env": {"name": "baseline", "conda_env": "latency"},
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            }
        ],
        "rows": [row],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a"},
            "exp2_same_device_calibration": {"panel": "b"},
            "exp3_cross_device_prediction": {
                "panel": "c",
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
        "env_name": "flappy",
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    vla = {
        "policy": "vla-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "env_name": "flappy",
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 2,
        "sim_worker_count": "auto",
        "seed_set": [0],
        "env_fps_calibration": {"worker_utilization_cap": 0.8},
        "runtime_env": {"name": "baseline", "conda_env": "latency"},
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
        "rows": [small, vla],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a"},
            "exp2_same_device_calibration": {"panel": "b"},
            "exp3_cross_device_prediction": {
                "panel": "c",
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
    run_env.setdefault("CROSS_DEVICE_INSTANCE_ID", "instance_test")
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


def test_runtime_env_detection_maps_conda_envs_and_override(monkeypatch):
    monkeypatch.delenv("CROSS_DEVICE_RUNTIME_ENV", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "latency")
    assert detect_runtime_env_raw() == "latency"
    assert runtime_env_mismatch(RUNTIME_SPEC) is None

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "starvla_rl_games_openvla")
    mismatch = runtime_env_mismatch(RUNTIME_SPEC)
    assert mismatch is not None
    assert "requires baseline (latency)" in mismatch
    assert "starvla_rl_games_openvla" in mismatch

    monkeypatch.setenv("CROSS_DEVICE_RUNTIME_ENV", "latency")
    assert detect_runtime_env_raw() == "latency"
    assert runtime_env_mismatch(RUNTIME_SPEC) is None


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
        {
            "id": "rtxpro6000blackwell",
            "display_name": "RTX PRO 6000",
            "profile_id": "L_RTX_PRO_6000_BLACKWELL",
            "aliases": ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"],
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
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "rtxpro6000blackwell",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition; NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "rtxpro6000blackwell",
    }
    for raw_name, expected_id in cases.items():
        assert normalize_hardware_name(raw_name, devices)["id"] == expected_id


def test_runtime_env_mismatch_blocks_execution_but_not_dry_run(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)

    mismatched = _run_cli(
        tmp_path,
        "measure",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
        check=False,
    )
    assert mismatched.returncode != 0
    assert "runtime env mismatch" in mismatched.stderr
    assert "requires baseline (latency)" in mismatched.stderr

    dry_run = _run_cli(
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
    assert "policy: dry-policy" in dry_run.stdout
    assert "runtime env mismatch" in dry_run.stderr


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
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    assert "policy: openvla-flappy" in result.stdout
    assert "policy: demon-attack-openvla" in result.stdout
    assert "policy: deadly-corridor-openvla" in result.stdout
    assert "openvla-flappy/flappy" in result.stdout
    assert "demon-attack-openvla/demon_attack" in result.stdout
    assert "deadly-corridor-openvla/deadly_corridor" in result.stdout


def test_real_protocol_rows_cover_flappy_demon_and_deadly():
    spec = yaml.safe_load(REAL_PROTOCOL.read_text(encoding="utf-8"))

    openvla_rows = {row["policy"]: row for row in spec["rows"]}

    assert set(openvla_rows) == {
        "openvla-flappy",
        "demon-attack-openvla",
        "deadly-corridor-openvla",
    }
    assert openvla_rows["openvla-flappy"]["policy_type"] == "starvla"
    assert openvla_rows["openvla-flappy"]["model_type"] == "openvla"
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
        {"target_device": "l40s", "replay_device": "b200"},
        {"target_device": "l40s", "replay_device": "h200"},
        {"target_device": "a100", "replay_device": "b200"},
        {"target_device": "a100", "replay_device": "h200"},
        {"target_device": "rtxpro6000blackwell", "replay_device": "b200"},
        {"target_device": "rtxpro6000blackwell", "replay_device": "h200"},
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
        "--dry-run",
        "--only-policy",
        "demon-attack-openvla",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = _first_runtime_generated_config(out, "measure")
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
    assert "profiles_smoke/openvla/demon_attack/a100/instance_test" in result.stdout
    generated = _first_runtime_generated_config(out, "measure")
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
        "--dry-run",
        "--only-policy",
        "openvla-flappy",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = _first_runtime_generated_config(out, "measure")
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["type"] == "starvla"
    assert "base_model_path" not in config["policy"]
    assert config["policy"]["checkpoint_path"] == (
        "/mnt/data/latency_data/openvla_flappy_fix_latency_2/checkpoints/steps_5000_pytorch_model.pt"
    )


def test_deadly_openvla_dry_run_writes_config(tmp_path: Path):
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
        "--dry-run",
        "--only-policy",
        "deadly-corridor-openvla",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
    )

    generated = _first_runtime_generated_config(out, "measure")
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["type"] == "starvla"
    assert config["env"]["obs_fps"] == 8.75
    assert "eval_suites" not in config["evaluation"]


def test_doctor_output_reports_runtime_match_and_skipped_rows(tmp_path: Path):
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
    assert "detected: latency" in result.stdout
    assert "required: baseline (latency)" in result.stdout
    assert "match: true" in result.stdout
    assert "Runnable stages:" in result.stdout
    assert "Skipped rows:" in result.stdout


def test_measure_plans_each_row_once_with_static_experiment_ids(tmp_path: Path):
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
    )

    assert result.stdout.count("policy: vla-policy") == 1
    assert result.stdout.count("policy: small-policy") == 1
    assert (
        "experiment_ids: exp1_latency_profile_summary,exp2_same_device_calibration,exp3_cross_device_prediction"
        in result.stdout
    )


def test_sim_same_dry_run_writes_guarded_generated_config_without_eval(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    profile_path = out / "profiles" / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"
    _write_profile_manifest(profile_path, env_fps=16.0, inference_worker_count=2)

    result = _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_INSTANCE_ID": "instance_local"},
    )

    assert "-m latency_bench.run" in result.stdout
    generated = _first_runtime_generated_config(out, "sim_same")
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["executor"]["mode"] == "simulated"
    assert config["executor"]["simulated_worker_capacity"] == 2
    assert config["env"]["env_fps"] == 16.0
    assert config["env"]["obs_fps"] == 16.0
    assert config["latency"]["type"] == "profile_sample"
    assert config["latency"]["profile_path"].endswith("profiles/test_model/flappy/a100/instance_local/profile.json")
    assert config["latency"]["sampler"] == "profile_default"
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1
    assert not (out / "runs").exists()


def test_sim_same_dry_run_passes_profile_sampler_to_generated_config(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    payload = yaml.safe_load(spec.read_text(encoding="utf-8"))
    payload["profile_sampler"] = "iid_inverse_cdf"
    spec.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    profile_path = out / "profiles" / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"
    _write_profile_manifest(profile_path, env_fps=16.0, inference_worker_count=2)

    _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        "--max-runs",
        "1",
        env={"CROSS_DEVICE_INSTANCE_ID": "instance_local"},
    )

    generated = _first_runtime_generated_config(out, "sim_same")
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["latency"]["sampler"] == "iid_inverse_cdf"


def test_sim_cross_dry_run_reads_profile_instance_from_target(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    profile_path = out / "profiles" / "test_model" / "flappy" / "a100" / "instance_remote" / "profile.json"
    _write_profile_manifest(profile_path, env_fps=16.0)

    result = _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--target",
        "a100/instance_remote",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        "--max-runs",
        "1",
    )

    assert "-m latency_bench.run" in result.stdout
    generated = _first_runtime_generated_config(out, "sim_cross")
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["latency"]["profile_path"].endswith("profiles/test_model/flappy/a100/instance_remote/profile.json")


def test_sim_same_uses_hf_profile_cache_before_skip(tmp_path: Path, monkeypatch):
    profile_resources = importlib.import_module("latency_bench.cross_device.profile_resources")
    planning = importlib.import_module("latency_bench.cross_device.planning")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    profiles_dir = out / "profiles"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
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
    monkeypatch.setattr(profile_resources, "ensure_profile_cached", ensure_profile_cached)

    plan = planning._build_plan(
        args,
        spec,
        "sim_same",
        out,
        profiles_dir,
        current_device,
        instance_id="instance_local",
        instance_origin="manual_override",
    )

    assert [item.runnable for item in plan] == [True]
    assert calls == [profiles_dir / "test_model" / "flappy" / "a100" / "instance_local" / "profile.json"]


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
        "--max-runs",
        "1",
    )

    assert "-m latency_bench.run" in result.stdout
    assert "-m latency_bench.experiments.build_latency_profile_from_trace" in result.stdout
    assert "--env-fps 30" in result.stdout
    assert "--metadata-json" in result.stdout
    assert '"policy_type": "random"' in result.stdout
    assert "--n-warmup" not in result.stdout
    assert "e2e_latencies.jsonl" in result.stdout
    assert "infer_latencies.jsonl" in result.stdout
    generated = _first_runtime_generated_config(out, "measure")
    calibration_generated = next((out / "generated_configs" / "measure").rglob("*.calibration.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    calibration_config = yaml.safe_load(calibration_generated.read_text(encoding="utf-8"))
    assert config["executor"]["mode"] == "realtime"
    assert calibration_config["executor"]["mode"] == "realtime"
    assert str(calibration_generated) in result.stdout
    assert "type" not in config["latency"]
    assert config["latency"]["device"] == "cuda:0"
    assert config["latency"]["sync_cuda"] is True
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1


def test_measure_dry_run_ignores_spec_n_samples(tmp_path: Path):
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
        "--max-runs",
        "1",
        "--n-episodes",
        "3",
    )

    assert "episodes: 3" in result.stdout
    assert "samples:" not in result.stdout
    assert "--n-samples" not in result.stdout
    assert "best_seed_0_n3_s5" not in result.stdout
    assert "best_seed_0_n3" in result.stdout


def test_measure_run_calibrates_env_fps_before_formal_eval(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("latency_bench.cross_device.planning")
    runner = importlib.import_module("latency_bench.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(
        args,
        spec,
        "measure",
        out,
        out / "profiles",
        current_device,
        instance_id="instance_test",
        instance_origin="manual_override",
    )[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_saturated_latency" in command:
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
            (task.output_dir / "infer_latencies.jsonl").write_text('{"latency_ms": 62.5}\n', encoding="utf-8")
            (task.output_dir / "e2e_latencies.jsonl").write_text('{"latency_ms": 125.0}\n', encoding="utf-8")
        elif "latency_bench.experiments.build_latency_profile_from_trace" in command:
            profile_output = Path(command[command.index("--output-dir") + 1])
            _write_profile_manifest(profile_output / "profile.json", env_fps=16.0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, args, out)

    assert [command[2] for command in commands] == [
        "latency_bench.experiments.calibrate_env_fps_from_saturated_latency",
        "latency_bench.run",
        "latency_bench.experiments.build_latency_profile_from_trace",
    ]
    final_config = yaml.safe_load(task.generated_config_path.read_text(encoding="utf-8"))
    assert final_config["env"]["env_fps"] == 16.0
    assert final_config["env"]["obs_fps"] == 16.0
    assert "--env-fps" in commands[2]
    assert commands[2][commands[2].index("--env-fps") + 1] == "16.0"


def test_measure_run_does_not_forward_n_samples_to_eval_or_profile(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("latency_bench.cross_device.planning")
    runner = importlib.import_module("latency_bench.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=3,
        only_device=None,
        only_env=None,
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(
        args,
        spec,
        "measure",
        out,
        out / "profiles",
        current_device,
        instance_id="instance_test",
        instance_origin="manual_override",
    )[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_saturated_latency" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"selected_env_fps": 16.0, "selected_obs_fps": 16.0}),
                encoding="utf-8",
            )
        elif "latency_bench.run" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("infer_latencies.jsonl").write_text(
                "".join(json.dumps({"latency_ms": sample}) + "\n" for sample in [10.0, 20.0]),
                encoding="utf-8",
            )
            output_dir.joinpath("e2e_latencies.jsonl").write_text(
                "".join(json.dumps({"latency_ms": sample}) + "\n" for sample in [20.0, 40.0]),
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

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, args, out)

    eval_commands = [command for command in commands if "latency_bench.run" in command]
    profile_command = commands[-1]
    assert len(eval_commands) == 1
    assert "--n-samples" not in eval_commands[0]
    assert "--n-samples" not in profile_command
    assert (task.output_dir / "infer_latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 2
    assert (task.output_dir / "e2e_latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 2
    assert (task.output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").count("episode_return_env") == 3


def test_measure_run_does_not_continue_after_episodes_until_n_samples(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("latency_bench.cross_device.planning")
    runner = importlib.import_module("latency_bench.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=2,
        only_device=None,
        only_env=None,
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(
        args,
        spec,
        "measure",
        out,
        out / "profiles",
        current_device,
        instance_id="instance_test",
        instance_origin="manual_override",
    )[0].task
    commands: list[list[str]] = []

    def fake_run(command, check):
        commands.append(list(command))
        if "latency_bench.experiments.calibrate_env_fps_from_saturated_latency" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"selected_env_fps": 16.0, "selected_obs_fps": 16.0}),
                encoding="utf-8",
            )
        elif "latency_bench.run" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("infer_latencies.jsonl").write_text(
                "".join(
                    json.dumps({"latency_ms": sample}) + "\n"
                    for sample in [10.0, 20.0, 30.0, 40.0, 50.0]
                ),
                encoding="utf-8",
            )
            output_dir.joinpath("e2e_latencies.jsonl").write_text(
                "".join(
                    json.dumps({"latency_ms": sample}) + "\n"
                    for sample in [20.0, 40.0, 60.0, 80.0, 100.0]
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

    runner._run_task(task, spec, {"raw": "a100", "device": current_device}, args, out)

    eval_commands = [command for command in commands if "latency_bench.run" in command]
    profile_command = commands[-1]
    assert len(eval_commands) == 1
    assert "--n-samples" not in eval_commands[0]
    assert "--n-samples" not in profile_command
    assert (task.output_dir / "infer_latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 5
    assert (task.output_dir / "e2e_latencies.jsonl").read_text(encoding="utf-8").count("latency_ms") == 5
    assert (task.output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").count("episode_return_env") == 2


def test_measure_run_records_failed_calibration_manifest_without_eval(tmp_path: Path, monkeypatch):
    planning = importlib.import_module("latency_bench.cross_device.planning")
    runner = importlib.import_module("latency_bench.cross_device.runner")
    base_config = tmp_path / "base.yaml"
    spec_path = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec_path, base_config)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    current_device = spec["devices"][0]
    args = argparse.Namespace(
        checkpoint_kind=None,
        dry_run=False,
        include_controls=False,
        max_runs=1,
        n_episodes=None,
        only_device=None,
        only_env=None,
        only_policy=None,
        overwrite=False,
        resume=False,
        seed_set=None,
        smoke=False,
        smoke_episodes=1,
        smoke_max_steps=None,
    )
    task = planning._build_plan(
        args,
        spec,
        "measure",
        out,
        out / "profiles",
        current_device,
        instance_id="instance_test",
        instance_origin="manual_override",
    )[0].task
    commands: list[list[str]] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    def fake_run(command, check):
        commands.append(list(command))
        raise subprocess.CalledProcessError(returncode=3, cmd=command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        runner._run_task(task, spec, {"raw": "a100", "device": current_device}, args, out)

    assert [command[2] for command in commands] == ["latency_bench.experiments.calibrate_env_fps_from_saturated_latency"]
    manifest = json.loads((task.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["python_executable"] == sys.executable
    assert manifest["cuda_visible_devices"] == "2,3"
    assert manifest["torch_version"]
    assert "torch_cuda_version" in manifest
    assert "latency_bench.experiments.calibrate_env_fps_from_saturated_latency" in manifest["calibration_command"]
    assert manifest["calibration_generated_config_path"].endswith(".calibration.yaml")
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
    row = spec_data["rows"][0]
    row.update(
        {
            "policy": "deadly-corridor-openvla",
            "env": "Deadly Corridor",
            "policy_type": "starvla",
            "model_type": "openvla",
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
        "--max-runs",
        "1",
    )

    generated = _first_runtime_generated_config(out, "measure")
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
    row = spec_data["rows"][0]
    row.update(
        {
            "policy": "deadly-corridor-openvla",
            "env": "Deadly Corridor",
            "policy_type": "starvla",
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
        "--max-runs",
        "1",
    )
    measure_config = yaml.safe_load(_first_runtime_generated_config(measure_out, "measure").read_text())
    assert measure_config["evaluation"]["eval_episodes"] == 1000
    assert "best_seed_0_n1000" in measure_config["logging"]["output_dir"]


def test_cli_n_episodes_overrides_stage_overrides(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    spec_data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    spec_data["rows"][0]["stage_overrides"] = {
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
        "--max-runs",
        "1",
        "--n-episodes",
        "20",
    )

    config = yaml.safe_load(_first_runtime_generated_config(out, "measure").read_text())
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
    _write_profile_manifest(
        out / "profiles" / "test_model" / "dummy" / "a100" / "instance_test" / "profile.json",
        env_fps=30.0,
    )

    result = _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--profiles-dir",
        str(out / "profiles"),
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
    _write_profile_manifest(
        out / "profiles" / "test_model" / "dummy" / "a100" / "instance_test" / "profile.json",
        env_fps=30.0,
    )

    completed_dir = out / "runs" / "sim_same" / "passing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "manifest.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    failed_dir = out / "runs" / "sim_same" / "failing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "manifest.json").write_text('{"status": "failed", "error": "old"}\n', encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--profiles-dir",
        str(out / "profiles"),
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


def test_sim_cross_target_requires_explicit_instance(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_mixed_spec(spec, base_config)

    result = _run_cli(
        tmp_path,
        "sim",
        "--spec",
        str(spec),
        "--hardware",
        "rtx4090",
        "--target",
        "a100",
        "--profiles-dir",
        str(out / "profiles"),
        "--out",
        str(out),
        "--dry-run",
        env={"CROSS_DEVICE_RUNTIME_ENV": "starvla_rl_games_openvla"},
        check=False,
    )

    assert result.returncode != 0
    assert "--target must be <device_id>/<instance_id>" in result.stderr
