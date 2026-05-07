from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path("scripts/experiments/run_cross_device_protocol.py")
REAL_PROTOCOL = Path("configs/experiments/cross_device_protocol.yaml")


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


def _write_base_config_with_policy(path: Path, policy_type: str) -> None:
    config = {
        "experiment": {"name": f"{policy_type}_seed0", "seed": 0, "num_episodes": 1},
        "executor": {"mode": "simulated"},
        "env": {"name": "dummy", "fps_env": 30, "noop_action": "noop"},
        "control": {"frequency_hz": 30, "observe_every_n_frames": 1},
        "latency": {"type": "zero"},
        "scheduler": {"persistence_mode": "hold_previous", "stale_action_mode": "latest_ready"},
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
        "runtime_env": "small_model",
        "env_name": "dummy",
        "fps": 30,
        "base_config": str(first_config),
    }
    second = {
        "policy": "passing-policy",
        "env": "Dummy",
        "policy_type": "random",
        "runtime_env": "small_model",
        "env_name": "dummy",
        "fps": 30,
        "base_config": str(second_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "latency_scope": "policy_predict_only",
        "trace_replay_order": "sequential",
        "loop_trace": True,
        "checkpoint_kind": "best",
        "n_warmup": 0,
        "n_episodes": 1,
        "seed_set": [0],
        "runtime_envs": {"small_model": {"conda_env": "latency", "description": "small"}},
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
            "exp2_same_device_calibration": {"panel": "b", "rows": [first, second]},
        },
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


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


def _openvla_stats_json(key: str, action_dim: int) -> str:
    stats = {
        key: {
            "action": {
                "min": [0.0] * action_dim,
                "max": [1.0] * action_dim,
                "q01": [0.0] * action_dim,
                "q99": [1.0] * action_dim,
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
                "mask": [True] * action_dim,
            }
        }
    }
    return json.dumps(stats)


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
        "NVIDIA A100-SXM4-40GB": "a100",
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    assert "policy: OpenVLA-Flappy" in result.stdout
    assert "policy: demon-attack-openvla" in result.stdout
    assert "policy: deadly-corridor-openvla" in result.stdout
    assert "openvla-flappy/flappy" in result.stdout
    assert "demon-attack-openvla/demon_attack" in result.stdout
    assert "deadly-corridor-openvla/deadly_corridor" in result.stdout


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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )
    assert "policy: OpenVLA-Flappy" in flappy.stdout
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )
    assert "policy: OpenVLA-Flappy" not in demon.stdout
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )
    assert "policy: deadly-corridor-openvla" in slug.stdout
    assert "policy: OpenVLA-Flappy" not in slug.stdout
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )
    assert "policy: deadly-corridor-openvla" in raw.stdout
    assert "policy: OpenVLA-Flappy" not in raw.stdout
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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    assert result.stdout.count("policy: ") == 1


def test_openvla_env_var_overrides_are_written_and_printed_in_dry_run(tmp_path: Path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "demon_attack_dataset_statistics.json").write_text(
        _openvla_stats_json("demon_attack", 6),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "demon_checkpoint"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")
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
        "--only-policy",
        "demon-attack-openvla",
        env={
            "CROSS_DEVICE_RUNTIME_ENV": "latency-openvla",
            "OPENVLA_BASE_MODEL_PATH": str(base_model),
            "DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH": str(checkpoint),
        },
    )

    assert f"resolved_base_model_path: {base_model}" in result.stdout
    assert f"resolved_checkpoint_path: {checkpoint}" in result.stdout
    assert "openvla_dependency_status: ok" in result.stdout
    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["base_model_path"] == str(base_model)
    assert config["policy"]["checkpoint_path"] == str(checkpoint)


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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    assert "smoke: true" in result.stdout
    assert "episodes: 1" in result.stdout
    assert "eval_max_steps: 7" in result.stdout
    assert "smoke_seed_0_n1" in result.stdout
    assert "profiles_smoke/demon-attack-openvla/demon_attack" in result.stdout
    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["experiment"]["num_episodes"] == 1
    assert config["evaluation"]["eval_episodes"] == 1
    assert config["evaluation"]["eval_max_steps"] == 7
    assert "smoke_seed_0_n1" in config["logging"]["output_dir"]
    assert not (out / "runs").exists()


def test_openvla_check_deps_only_reports_missing_dependencies_without_eval_outputs(tmp_path: Path):
    out = tmp_path / "results"

    result = _run_cli(
        tmp_path,
        "check-deps",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--only-policy",
        "demon-attack-openvla",
        check=False,
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    assert result.returncode != 0
    assert "policy | env | base_model | stats_json | unnorm_key | checkpoint | weights | status" in result.stdout
    assert "demon-attack-openvla" in result.stdout
    assert "demon_attack" in result.stdout
    assert "missing" in result.stdout.lower()
    assert not (out / "runs").exists()


def test_openvla_check_deps_only_returns_zero_when_selected_dependencies_exist(tmp_path: Path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "demon_attack_dataset_statistics.json").write_text(
        _openvla_stats_json("demon_attack", 6),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "demon_checkpoint"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")
    out = tmp_path / "results"

    result = _run_cli(
        tmp_path,
        "check-deps",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(out),
        "--runtime-env",
        "openvla",
        "--only-policy",
        "demon-attack-openvla",
        env={
            "CROSS_DEVICE_RUNTIME_ENV": "latency-openvla",
            "OPENVLA_BASE_MODEL_PATH": str(base_model),
            "DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH": str(checkpoint),
        },
    )

    assert "demon-attack-openvla" in result.stdout
    assert "| ok" in result.stdout
    assert not (out / "runs").exists()


def test_flappy_openvla_checkpoint_env_override_is_written_in_dry_run(tmp_path: Path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "flappy_dataset_statistics.json").write_text(
        _openvla_stats_json("flappy_bird", 2),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "flappy_checkpoint"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")
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
        env={
            "CROSS_DEVICE_RUNTIME_ENV": "latency-openvla",
            "OPENVLA_BASE_MODEL_PATH": str(base_model),
            "FLAPPY_OPENVLA_CHECKPOINT_PATH": str(checkpoint),
        },
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["base_model_path"] == str(base_model)
    assert config["policy"]["checkpoint_path"] == str(checkpoint)


def test_deadly_openvla_check_deps_rejects_wrong_action_stats_length(tmp_path: Path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "deadly_corridor_dataset_statistics.json").write_text(
        _openvla_stats_json("deadly_corridor", 7),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "deadly_checkpoint"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")

    result = _run_cli(
        tmp_path,
        "check-deps",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(tmp_path / "results"),
        "--runtime-env",
        "openvla",
        "--only-policy",
        "deadly-corridor-openvla",
        check=False,
        env={
            "CROSS_DEVICE_RUNTIME_ENV": "latency-openvla",
            "OPENVLA_BASE_MODEL_PATH": str(base_model),
            "DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH": str(checkpoint),
        },
    )

    assert result.returncode != 0
    assert "deadly_corridor action stats q01 length 7 != policy.action_dim 11" in result.stdout


def test_deadly_openvla_check_deps_accepts_eleven_dim_action_stats(tmp_path: Path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "deadly_corridor_dataset_statistics.json").write_text(
        _openvla_stats_json("deadly_corridor", 11),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "deadly_checkpoint"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real checkpoint")

    result = _run_cli(
        tmp_path,
        "check-deps",
        "--spec",
        str(REAL_PROTOCOL),
        "--hardware",
        "a100",
        "--out",
        str(tmp_path / "results"),
        "--runtime-env",
        "openvla",
        "--only-policy",
        "deadly-corridor-openvla",
        env={
            "CROSS_DEVICE_RUNTIME_ENV": "latency-openvla",
            "OPENVLA_BASE_MODEL_PATH": str(base_model),
            "DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH": str(checkpoint),
        },
    )

    assert "deadly-corridor-openvla" in result.stdout
    assert "| ok" in result.stdout


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
        env={"CROSS_DEVICE_RUNTIME_ENV": "latency-openvla"},
    )

    generated = next((out / "generated_configs" / "measure").rglob("*.yaml"))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert config["policy"]["action_dim"] == 11
    assert config["policy"]["action_layout"] == "deadly_corridor_factorized_11"


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
    assert measure_config["experiment"]["num_episodes"] == 1000
    assert measure_config["evaluation"]["eval_episodes"] == 1000
    assert "best_seed_0_n1000" in measure_config["logging"]["output_dir"]

    realtime_out = tmp_path / "realtime_results"
    _run_cli(
        tmp_path,
        "realtime",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(realtime_out),
        "--dry-run",
        "--only-experiment",
        "exp2_same_device_calibration",
        "--max-runs",
        "1",
    )
    realtime_config = yaml.safe_load(next((realtime_out / "generated_configs" / "realtime").rglob("*.yaml")).read_text())
    assert realtime_config["experiment"]["num_episodes"] == 10
    assert realtime_config["evaluation"]["eval_episodes"] == 10
    assert "best_seed_0_n10" in realtime_config["logging"]["output_dir"]


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
    assert config["experiment"]["num_episodes"] == 20
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

    result = _run_cli(
        tmp_path,
        "realtime",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
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
    assert "stage=realtime" in combined
    assert "output_dir=" in combined
    assert "error=" in combined
    assert "passing-policy" in combined
    failed_manifest = next((out / "runs" / "realtime" / "failing-policy").rglob("manifest.json"))
    manifest = yaml.safe_load(failed_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert next((out / "runs" / "realtime" / "passing-policy").rglob("manifest.json")).exists()


def test_resume_skips_completed_manifest_but_retries_failed_run(tmp_path: Path):
    failing_config = tmp_path / "failing.yaml"
    passing_config = tmp_path / "passing.yaml"
    spec = tmp_path / "protocol.yaml"
    out = tmp_path / "results"
    _write_base_config_with_policy(failing_config, "unknown_policy")
    _write_base_config_with_policy(passing_config, "random")
    _write_two_row_realtime_spec(spec, failing_config, passing_config)

    completed_dir = out / "runs" / "realtime" / "passing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    completed_dir.mkdir(parents=True)
    (completed_dir / "manifest.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    failed_dir = out / "runs" / "realtime" / "failing-policy" / "dummy" / "a100" / "best_seed_0_n1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "manifest.json").write_text('{"status": "failed", "error": "old"}\n', encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "realtime",
        "--spec",
        str(spec),
        "--hardware",
        "a100",
        "--out",
        str(out),
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
