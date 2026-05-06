from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path("scripts/experiments/run_cross_device_protocol.py")


RUNTIME_SPEC = {
    "runtime_envs": {
        "small_model": {"conda_env": "latency", "description": "small"},
        "openvla": {"conda_env": "latency-openvla", "description": "vla"},
    }
}


def _write_base_config(path: Path) -> None:
    config = {
        "experiment": {"name": "dry_run_policy_seed0", "seed": 0, "num_episodes": 2},
        "executor": {"mode": "simulated"},
        "env": {"name": "flappy", "fps_env": 30, "noop_action": "noop"},
        "control": {"frequency_hz": 30, "observe_every_n_frames": 1},
        "latency": {"type": "fixed", "fixed_latency_ms": 33.333333333333336, "train_values": [1]},
        "scheduler": {"persistence_mode": "hold_previous", "stale_action_mode": "latest_ready"},
        "policy": {"type": "random", "actions": ["noop", "flap"]},
        "logging": {"output_dir": "unused"},
        "evaluation": {"eval_episodes": 2, "eval_parallel_envs": 4, "eval_latency_values": [0, 1], "eval_max_steps": 10},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_spec(path: Path, base_config: Path) -> None:
    row = {
        "policy": "dry-policy",
        "env": "Flappy",
        "policy_type": "random",
        "runtime_env": "small_model",
        "env_name": "flappy",
        "fps": 30,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "latency_scope": "policy_predict_only",
        "trace_replay_order": "sequential",
        "loop_trace": True,
        "checkpoint_kind": "best",
        "n_warmup": 0,
        "n_episodes": 2,
        "seed_set": [0],
        "runtime_envs": {
            "small_model": {"conda_env": "latency", "description": "small"},
            "openvla": {"conda_env": "latency-openvla", "description": "vla"},
        },
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "speed_rank": 3,
                "aliases": ["NVIDIA A100"],
            }
        ],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a", "rows": [dict(row)]},
            "exp2_same_device_calibration": {"panel": "b", "rows": [dict(row)]},
            "exp3_cross_device_prediction": {
                "panel": "c",
                "rows": [dict(row)],
                "replay_devices": ["a100"],
                "main_claim_rule": "replay_device_speed_rank > target_device_speed_rank",
            },
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _write_mixed_spec(path: Path, base_config: Path) -> None:
    small = {
        "policy": "small-policy",
        "env": "Flappy",
        "policy_type": "random",
        "runtime_env": "small_model",
        "env_name": "flappy",
        "fps": 30,
        "base_config": str(base_config),
    }
    vla = {
        "policy": "vla-policy",
        "env": "Flappy",
        "policy_type": "random",
        "runtime_env": "openvla",
        "env_name": "flappy",
        "fps": 30,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "latency_scope": "policy_predict_only",
        "trace_replay_order": "sequential",
        "loop_trace": True,
        "checkpoint_kind": "best",
        "n_warmup": 0,
        "n_episodes": 2,
        "seed_set": [0],
        "runtime_envs": {
            "small_model": {"conda_env": "latency", "description": "small"},
            "openvla": {"conda_env": "latency-openvla", "description": "vla"},
        },
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "speed_rank": 3,
                "aliases": ["NVIDIA A100"],
            },
            {
                "id": "h200",
                "display_name": "H200",
                "profile_id": "L_H200",
                "speed_rank": 4,
                "aliases": ["NVIDIA H200"],
            },
            {
                "id": "rtx4090",
                "display_name": "RTX 4090",
                "profile_id": "L_4090",
                "speed_rank": 2,
                "aliases": ["NVIDIA GeForce RTX 4090"],
            },
        ],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a", "rows": [dict(vla)]},
            "exp2_same_device_calibration": {"panel": "b", "rows": [dict(small)]},
            "exp3_cross_device_prediction": {
                "panel": "c",
                "rows": [dict(vla)],
                "replay_devices": ["h200", "a100"],
                "main_claim_rule": "replay_device_speed_rank > target_device_speed_rank",
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
        [sys.executable, str(SCRIPT), *args],
        cwd=Path.cwd(),
        check=check,
        capture_output=True,
        text=True,
        env=run_env,
    )


def _load_protocol_module():
    spec = importlib.util.spec_from_file_location("run_cross_device_protocol_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_env_detection_maps_conda_envs_and_override(monkeypatch):
    module = _load_protocol_module()
    monkeypatch.delenv("CROSS_DEVICE_RUNTIME_ENV", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "latency")
    assert module.detect_runtime_env(RUNTIME_SPEC).matched_alias == "small_model"

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "latency-openvla")
    assert module.detect_runtime_env(RUNTIME_SPEC).matched_alias == "openvla"

    monkeypatch.setenv("CROSS_DEVICE_RUNTIME_ENV", "latency")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "latency-openvla")
    detected = module.detect_runtime_env(RUNTIME_SPEC)
    assert detected.detected_raw == "latency"
    assert detected.matched_alias == "small_model"


def test_hardware_detection_accepts_full_gpu_model_names_and_multi_gpu_text():
    module = _load_protocol_module()
    devices = [
        {
            "id": "rtx3090",
            "display_name": "RTX 3090",
            "profile_id": "L_3090",
            "speed_rank": 1,
            "aliases": ["NVIDIA GeForce RTX 3090"],
        },
        {
            "id": "rtx4090",
            "display_name": "RTX 4090",
            "profile_id": "L_4090",
            "speed_rank": 2,
            "aliases": ["NVIDIA GeForce RTX 4090"],
        },
        {
            "id": "l40s",
            "display_name": "L40S",
            "profile_id": "L_L40S",
            "speed_rank": 2,
            "aliases": ["NVIDIA L40S"],
        },
        {
            "id": "a100",
            "display_name": "A100",
            "profile_id": "L_A100",
            "speed_rank": 3,
            "aliases": ["NVIDIA A100"],
        },
        {
            "id": "h200",
            "display_name": "H200",
            "profile_id": "L_H200",
            "speed_rank": 4,
            "aliases": ["NVIDIA H200"],
        },
        {
            "id": "b200",
            "display_name": "B200",
            "profile_id": "L_B200",
            "speed_rank": 5,
            "aliases": ["NVIDIA B200"],
        },
    ]

    cases = {
        "NVIDIA A100-SXM4-80GB": "a100",
        "NVIDIA A100-SXM4-80GB; NVIDIA A100-SXM4-80GB": "a100",
        "NVIDIA L40S": "l40s",
        "NVIDIA GeForce RTX 3090": "rtx3090",
        "NVIDIA GeForce RTX 4090": "rtx4090",
        "NVIDIA H200": "h200",
        "NVIDIA B200": "b200",
    }
    for raw_name, expected_id in cases.items():
        assert module.normalize_hardware_name(raw_name, devices)["id"] == expected_id


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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )
    assert "policy: vla-policy" in openvla.stdout
    assert "skipped runtime_env mismatch: 1" in openvla.stdout


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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
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
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}\n", encoding="utf-8")

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
    assert config["latency"]["type"] == "empirical_trace_replay"
    assert config["latency"]["profile_path"].endswith("profiles/dry-policy/flappy/L_A100/profile.json")
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1
    assert not (out / "runs").exists()


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
    assert "latencies.jsonl" in result.stdout
    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["executor"]["mode"] == "realtime"
    assert config["latency"]["type"] == "measured"
    assert config["evaluation"]["eval_latency_values"] is None
    assert config["evaluation"]["eval_parallel_envs"] == 1


def test_measure_dry_run_preserves_integral_fps_as_int_for_deadly_corridor(tmp_path: Path):
    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config(base_config)
    _write_spec(spec, base_config)
    spec_data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    row = spec_data["experiments"]["exp2_same_device_calibration"]["rows"][0]
    row.update(
        {
            "policy": "deadly-corridor-sf-l1",
            "env": "Deadly Corridor",
            "policy_type": "deadly_corridor_sf",
            "env_name": "deadly_corridor",
            "fps": 35,
        }
    )
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
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["env"]["name"] == "deadly_corridor"
    assert config["env"]["fps_env"] == 35
    assert isinstance(config["env"]["fps_env"], int)
    assert config["control"]["frequency_hz"] == 35
    assert isinstance(config["control"]["frequency_hz"], int)


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


def test_sim_cross_skips_replay_hardware_not_configured_for_experiment(tmp_path: Path):
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    assert "runnable rows: 0" in result.stdout
    assert "replay hardware not configured for experiment" in result.stdout
