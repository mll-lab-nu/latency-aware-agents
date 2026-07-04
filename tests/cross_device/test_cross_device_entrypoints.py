from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def _write_base_config(path: Path) -> None:
    config = {
        "experiment": {"name": "entrypoint_smoke", "seed": 0},
        "executor": {"mode": "simulated"},
        "env": {"name": "flappy", "env_fps": 30, "obs_fps": 30, "noop_action": "noop"},
        "latency": {"type": "fixed", "fixed_latency_ms": 33.333333333333336},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": "random", "actions": ["noop", "flap"]},
        "logging": {"output_dir": "unused"},
        "evaluation": {
            "eval_episodes": 1,
            "eval_parallel_envs": 1,
            "eval_latency_values": None,
            "eval_max_steps": 1,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_protocol_spec(path: Path, base_config: Path) -> None:
    row = {
        "policy": "entry-policy",
        "env": "Flappy",
        "policy_type": "random",
        "model_type": "test_model",
        "runtime_env": "baseline",
        "env_name": "flappy",
        "fps": 30,
        "target_latency_frames": 1,
        "base_config": str(base_config),
    }
    spec = {
        "protocol_id": "cross_device_latency_v1",
        "output_root": str(path.parent / "results"),
        "checkpoint_kind": "best",
        "n_episodes": 1,
        "seed_set": [0],
        "env_fps_calibration": {"worker_utilization_cap": 0.8},
        "runtime_envs": {"baseline": {"conda_env": "latency", "description": "baseline"}},
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


def test_scripts_cross_device_modules_are_compatibility_aliases():
    module_pairs = {
        "scripts.cross_device.models": "latency_bench.cross_device.models",
        "scripts.cross_device.profile_layout": "latency_bench.cross_device.layout",
        "scripts.cross_device.instances": "latency_bench.cross_device.instances",
        "scripts.cross_device.runtime": "latency_bench.cross_device.runtime",
        "scripts.cross_device.planning": "latency_bench.cross_device.planning",
        "scripts.cross_device.config": "latency_bench.cross_device.configs",
        "scripts.cross_device.runner": "latency_bench.cross_device.runner",
        "scripts.cross_device.profile_cache": "latency_bench.cross_device.sync",
    }

    for old_name, new_name in module_pairs.items():
        assert importlib.import_module(old_name) is importlib.import_module(new_name)


def test_formal_protocol_entrypoint_runs_doctor_dry_run(tmp_path: Path, capsys):
    from latency_bench.cross_device.protocol import main

    base_config = tmp_path / "base.yaml"
    spec = tmp_path / "protocol.yaml"
    _write_base_config(base_config)
    _write_protocol_spec(spec, base_config)

    main(
        [
            "doctor",
            "--spec",
            str(spec),
            "--hardware",
            "a100",
            "--runtime-env",
            "baseline",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert "Runtime environment:" in out
    assert "matched device: a100" in out
    assert "policy=entry-policy" in out
