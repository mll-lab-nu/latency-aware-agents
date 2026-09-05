from __future__ import annotations

import sys

import pytest
import yaml

from scripts.tasks.humanoidbench_balance_simple import run_latency_pipeline
from training import train


def test_balance_latency_pipeline_passes_finite_vla_schedule(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    latency_config = tmp_path / "latency.yaml"
    latency_config.write_text(
        "latency:\n  method: temporal\n  seed: 17\n"
        "executor:\n  simulated_worker_capacity: 3\n"
        "scheduler:\n  hold_policy: hold\n  ordering_policy: latest_ready\n",
        encoding="utf-8",
    )
    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        "backend:\n  train_dir: native-checkpoints\nexperiment: {}\nlogging: {}\n",
        encoding="utf-8",
    )
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("{}\n", encoding="utf-8")
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(
        "\n".join(
            [
                "output_dir: pipeline-output",
                "profile_model: model",
                "profile_args: []",
                f"latency_config: {latency_config}",
                f"native_train_config: {train_config}",
                f"native_eval_config: {eval_config}",
                "native_train_run_name: native-run",
                "native_restart_behavior: overwrite",
                "native_checkpoint_kind: latest",
                "environment: {}",
                "vla_run_id: vla-run",
                "baseline_model_path: baseline-model",
                "collection:",
                "  max_episodes: 2",
                "  val_fraction: 0.5",
                "  num_envs: 1",
                "  replay_num_envs: 1",
                "  image_writer_workers: 0",
                "  row_buffer_size: 64",
                "  device: cuda",
                "vla_max_steps: 2",
                "vla_step_interval: 1",
                "vla_start_step: 1",
                "vla_num_gpus: 1",
                "vla_global_batch_size: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls = []

    def run(stage, command, *, output_dir, environment):
        calls.append((stage, command, output_dir, environment))
        return ""

    checkpoint = tmp_path / "teacher.pth"
    monkeypatch.setattr(run_latency_pipeline, "_run", run)
    monkeypatch.setattr(run_latency_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_latency_pipeline, "find_checkpoint", lambda *args, **kwargs: checkpoint
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_latency_pipeline.py",
            "--config",
            str(pipeline_config),
            "--profile-path",
            str(tmp_path / "profile.json"),
        ],
    )

    run_latency_pipeline.main()

    assert [call[0] for call in calls] == ["native_train", "vla"]
    environment = calls[0][3]
    assert environment["BALANCE_VLA_MAX_STEPS"] == "2"
    assert environment["BALANCE_VLA_STEP_INTERVAL"] == "1"
    assert environment["BALANCE_VLA_START_STEP"] == "1"
    assert environment["BALANCE_VLA_NUM_GPUS"] == "1"
    assert environment["BALANCE_VLA_GLOBAL_BATCH_SIZE"] == "1"
    assert environment["BALANCE_EXPORT_MAX_EPISODES"] == "2"
    assert environment["BALANCE_EXPORT_VAL_FRACTION"] == "0.5"
    assert environment["BALANCE_EXPORT_NUM_ENVS"] == "1"
    assert environment["BALANCE_EXPORT_REPLAY_NUM_ENVS"] == "1"
    assert environment["BALANCE_EXPORT_IMAGE_WRITER_WORKERS"] == "0"
    assert environment["BALANCE_EXPORT_ROW_BUFFER_SIZE"] == "64"
    assert environment["BALANCE_EXPORT_DEVICE"] == "cuda"
    assert environment["BALANCE_BASELINE_MODEL_PATH"] == str(
        run_latency_pipeline.ROOT / "baseline-model"
    )
    assert environment["BALANCE_TRAIN_DIR"].endswith("native-checkpoints")
    assert calls[1][1] == [
        "bash",
        run_latency_pipeline.VLA_ENTRY,
        str(checkpoint),
        str(tmp_path / "pipeline-output/latency.yaml"),
    ]
    captured = []
    monkeypatch.setattr(train, "run_from_config", lambda config: captured.append(config) or 0)
    with pytest.raises(SystemExit) as stopped:
        train.main([
            "--config", str(train_config),
            "--latency-config", str(tmp_path / "pipeline-output/latency.yaml"),
        ])
    assert stopped.value.code == 0
    generated = yaml.safe_load((tmp_path / "pipeline-output/latency.yaml").read_text())
    for section in ("executor", "latency", "scheduler"):
        assert captured[0][section] == generated[section]
