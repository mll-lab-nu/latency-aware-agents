from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "experiments" / "infra_acceleration_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("infra_acceleration_benchmark", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _base_config(env_name: str, tmp_path: Path) -> dict:
    config = {
        "experiment": {"name": f"{env_name}_base", "seed": 0},
        "backend": {
            "type": "sample_factory",
            "algo": "APPO",
            "device": "cpu",
            "train_dir": str(tmp_path / "checkpoints"),
            "restart_behavior": "resume",
            "run_mode": "train",
        },
        "executor": {"mode": "simulated"},
        "env": {
            "name": env_name,
            "env_fps": 30,
            "obs_fps": 30,
            "frame_stack": 4,
            "simulator": "cpu",
            "use_gpu_render": False,
            "gpu_render_device": "auto",
            "gpu_env_device": "auto",
        },
        "latency": {"type": "zero", "add_latency_info": False},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": "random", "actions": [0]},
        "training": {
            "train_for_env_steps": 25_000_000,
            "num_workers": 2,
            "num_envs_per_worker": 8,
            "worker_num_splits": 1,
            "batch_size": 1024,
            "rollout": 32,
            "recurrence": 1,
            "num_epochs": 2,
            "num_batches_per_epoch": 2,
        },
        "evaluation": {"eval_interval_steps": 1_000_000, "eval_episodes": 20, "eval_max_steps": 100},
        "logging": {
            "output_dir": str(tmp_path / "outputs"),
            "video": {"enabled": True, "num_bins": 10},
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
            "wandb_project": "latency-sensitive-bench",
        },
    }
    if env_name == "flappy":
        config["scheduler"]["hold_policy"] = "one_frame_then_noop"
        config["env"]["obs_resize"] = [84, 84]
    if env_name == "demon_attack":
        config["env"].update({"env_fps": 60, "obs_fps": 15, "env_id": "ALE/DemonAttack-v5"})
    if env_name == "deadly_corridor":
        config["env"].update({"env_fps": 35, "obs_fps": 8.75, "env_id": "doom_deadly_corridor"})
    return config


def test_paper_grade_dry_run_stages_real_baseline_and_accelerated_backends(tmp_path: Path):
    bench = _load_module()

    status = bench.main(["--dry-run", "--budget", "paper-grade", "--output-root", str(tmp_path)])

    configs = _read_jsonl(tmp_path / "staged_configs.jsonl")
    by_run = {config["experiment"]["name"]: config for config in configs}
    assert status == 0
    assert len(configs) == 18

    flappy_baseline = by_run["infra_accel_flappy_baseline_r0"]
    flappy_accelerated = by_run["infra_accel_flappy_accelerated_r0"]
    assert flappy_baseline["env"]["simulator"] == "cpu"
    assert flappy_baseline["env"]["use_gpu_render"] is False
    assert flappy_baseline["backend"]["device"] == "cpu"
    assert flappy_accelerated["env"]["simulator"] == "gpu"
    assert flappy_accelerated["env"]["use_gpu_render"] is True
    assert flappy_accelerated["backend"]["device"] == "cuda"

    assert by_run["infra_accel_demon_attack_baseline_r0"]["env"]["simulator"] == "cpu"
    assert by_run["infra_accel_demon_attack_accelerated_r0"]["env"]["simulator"] == "gpu"
    assert by_run["infra_accel_deadly_corridor_baseline_r0"]["env"]["simulator"] == "cpu"
    assert by_run["infra_accel_deadly_corridor_accelerated_r0"]["env"]["simulator"] == "envpool"
    for config in configs:
        extra_args = config["backend"]["extra_args"]
        assert extra_args[-4:] == [
            "--decorrelate_envs_on_one_worker",
            "False",
            "--heartbeat_reporting_interval",
            "900",
        ]


def test_paper_grade_train_steps_are_calibrated_from_baseline_runtime(monkeypatch, tmp_path: Path):
    bench = _load_module()
    budget = bench.BUDGETS["paper-grade"]

    def fake_run_training(config):
        return {
            "status": 0,
            "wall_sec": 30.0,
            "cpu_sec": 12.0,
            "env_steps": config["training"]["train_for_env_steps"],
        }

    monkeypatch.setattr(bench, "run_training", fake_run_training)

    resolved_budget, calibration_row = bench.calibrate_train_budget(
        _base_config("flappy", tmp_path),
        env_name="flappy",
        output_root=tmp_path / "bench",
        budget=budget,
    )

    assert resolved_budget.train_for_env_steps == 98_304
    assert resolved_budget.eval_episodes == 20
    assert resolved_budget.repeats == 3
    assert calibration_row["calibration_env_steps"] == 16_384
    assert calibration_row["calibration_fps"] == pytest.approx(546.1333333333333)
    assert calibration_row["target_train_seconds"] == pytest.approx(180.0)


def test_sample_factory_collected_env_steps_are_read_from_latest_checkpoint(tmp_path: Path):
    bench = _load_module()
    config = _base_config("flappy", tmp_path)
    config["experiment"]["name"] = "infra_accel_flappy_baseline_r0"
    config["backend"]["train_dir"] = str(tmp_path / "checkpoints")
    checkpoint_dir = tmp_path / "checkpoints" / "infra_accel_flappy_baseline_r0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_000000056_065536.pth").write_bytes(b"")
    (checkpoint_dir / "checkpoint_000000112_131072.pth").write_bytes(b"")

    assert bench.read_sample_factory_collected_env_steps(config) == 131_072


def test_calibration_training_failure_stops_without_resolved_budget(monkeypatch, tmp_path: Path):
    bench = _load_module()

    def fake_run_training(config):
        return {
            "status": 1,
            "wall_sec": 30.0,
            "cpu_sec": 12.0,
            "env_steps": config["training"]["train_for_env_steps"],
        }

    monkeypatch.setattr(bench, "run_training", fake_run_training)

    with pytest.raises(RuntimeError, match="training failed.*infra_accel_flappy_baseline_calibration_r0.*status=1"):
        bench.calibrate_train_budget(
            _base_config("flappy", tmp_path),
            env_name="flappy",
            output_root=tmp_path / "bench",
            budget=bench.BUDGETS["paper-grade"],
        )


def test_train_failure_stops_before_eval_and_summary(monkeypatch, tmp_path: Path):
    bench = _load_module()

    monkeypatch.setitem(bench.BASE_CONFIGS, "flappy", tmp_path / "flappy.yaml")

    def fake_load_config(path):
        return _base_config("flappy", tmp_path)

    def fake_run_training(config):
        return {
            "status": 1,
            "wall_sec": 3.0,
            "cpu_sec": 2.0,
            "env_steps": config["training"]["train_for_env_steps"],
        }

    def fake_run_eval(config, *, output_dir):
        raise AssertionError("eval should not run after failed training")

    monkeypatch.setattr(bench, "load_config", fake_load_config)
    monkeypatch.setattr(bench, "run_training", fake_run_training)
    monkeypatch.setattr(bench, "run_eval", fake_run_eval)

    with pytest.raises(RuntimeError, match="training failed.*infra_accel_flappy_baseline_r0.*status=1"):
        bench.main(
            [
                "--envs",
                "flappy",
                "--train-for-env-steps",
                "4096",
                "--output-root",
                str(tmp_path / "bench"),
            ]
        )

    assert not (tmp_path / "bench" / "summary.csv").exists()


def test_summary_reports_train_fps_wall_time_reduction_and_eval_stage_speed():
    bench = _load_module()
    rows = [
        {
            "env_name": "flappy",
            "phase": "train",
            "variant": "baseline",
            "status": 0,
            "wall_sec": 20.0,
            "env_steps": 1000,
            "episodes": "",
            "backend_label": "cpu",
        },
        {
            "env_name": "flappy",
            "phase": "train",
            "variant": "accelerated",
            "status": 0,
            "wall_sec": 5.0,
            "env_steps": 1000,
            "episodes": "",
            "backend_label": "cuda",
        },
        {
            "env_name": "flappy",
            "phase": "eval",
            "variant": "baseline",
            "status": 0,
            "wall_sec": 8.0,
            "env_steps": "",
            "episodes": 4,
            "episode_steps": 80,
            "backend_label": "serial",
        },
        {
            "env_name": "flappy",
            "phase": "eval",
            "variant": "accelerated",
            "status": 0,
            "wall_sec": 4.0,
            "env_steps": "",
            "episodes": 4,
            "episode_steps": 80,
            "backend_label": "flappy_gpu_batched",
        },
    ]

    summary = bench.summarize_results(rows)

    train = next(row for row in summary if row["phase"] == "train")
    eval_row = next(row for row in summary if row["phase"] == "eval")
    assert train["baseline_fps"] == pytest.approx(50.0)
    assert train["accelerated_fps"] == pytest.approx(200.0)
    assert train["speedup"] == pytest.approx(4.0)
    assert train["time_reduction_pct"] == pytest.approx(75.0)
    assert eval_row["baseline_episodes_per_sec"] == pytest.approx(0.5)
    assert eval_row["accelerated_episodes_per_sec"] == pytest.approx(1.0)
    assert eval_row["speedup"] == pytest.approx(2.0)
    assert eval_row["time_reduction_pct"] == pytest.approx(50.0)


def test_parity_timeout_is_reported_as_incomplete_behavior_proof(monkeypatch, tmp_path: Path):
    bench = _load_module()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial out", stderr="partial err")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    rows = bench.run_parity(["flappy"], output_root=tmp_path, timeout_sec=1.5)

    assert rows[0]["env_name"] == "flappy"
    assert rows[0]["parity_status"] == "timeout"
    assert rows[0]["status"] == 124
    assert rows[0]["behavior_changed"] == 1
    assert rows[0]["proof_complete"] == 0
    assert Path(rows[0]["log_stdout"]).read_text(encoding="utf-8") == "partial out"
    assert Path(rows[0]["log_stderr"]).read_text(encoding="utf-8") == "partial err"
