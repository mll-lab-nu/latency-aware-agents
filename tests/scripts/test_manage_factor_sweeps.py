from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml

from scripts.manage_factor_sweeps import collect_status_rows


def _write_train_config(
    path: Path,
    *,
    train_dir: Path,
    output_dir: Path,
    experiment_name: str = "flappy_frame_stack_fixed_l1_fs4_seed0",
    wandb_name: str = "flappy-fs4-fixed_l1-s0",
    wandb_entity: str | None = None,
) -> None:
    logging_cfg = {
        "output_dir": str(output_dir),
        "wandb_project": "latency-sensitive-bench",
        "wandb_group": "flappy-fs4-fixed_l1",
        "wandb_name": wandb_name,
    }
    if wandb_entity is not None:
        logging_cfg["wandb_user"] = wandb_entity
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": experiment_name, "seed": 0},
                "backend": {"train_dir": str(train_dir), "restart_behavior": "resume"},
                "training": {"train_for_env_steps": 1000},
                "logging": logging_cfg,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_commands(root: Path, train_config: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train_commands.txt").write_text(
        f"python -m training.train --config {train_config}\n",
        encoding="utf-8",
    )


def _write_checkpoint(train_dir: Path, experiment_name: str, env_steps: int) -> None:
    ckpt_dir = train_dir / experiment_name / "checkpoint_p0"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / f"checkpoint_000000001_{env_steps:012d}.pth").write_bytes(b"checkpoint")


def test_status_marks_train_complete_without_final_eval_as_eval_incomplete(tmp_path):
    generated_root = tmp_path / "generated"
    train_config = generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0" / "train.yaml"
    train_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs" / "train" / "frame_stack" / "fixed_l1" / "fs4" / "seed_0"
    _write_train_config(train_config, train_dir=train_dir, output_dir=output_dir)
    _write_commands(generated_root, train_config)
    _write_checkpoint(train_dir, "flappy_frame_stack_fixed_l1_fs4_seed0", env_steps=1024)

    rows = collect_status_rows(generated_root, active_grace_seconds=0)

    assert len(rows) == 1
    assert rows[0]["status"] == "eval_incomplete"
    assert rows[0]["latest_env_steps"] == "1024"
    assert rows[0]["wandb_name"] == "flappy-fs4-fixed_l1-s0"


def test_status_marks_final_eval_result_as_complete(tmp_path):
    generated_root = tmp_path / "generated"
    train_config = generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0" / "train.yaml"
    train_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs" / "train" / "frame_stack" / "fixed_l1" / "fs4" / "seed_0"
    _write_train_config(train_config, train_dir=train_dir, output_dir=output_dir)
    _write_commands(generated_root, train_config)
    _write_checkpoint(train_dir, "flappy_frame_stack_fixed_l1_fs4_seed0", env_steps=1024)
    result_path = output_dir / "latency_bench_train_eval" / "step_1024_final" / "eval_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"summaries": []}\n', encoding="utf-8")

    rows = collect_status_rows(generated_root, active_grace_seconds=0)

    assert rows[0]["status"] == "complete"
    assert rows[0]["final_eval_result"] == str(result_path)


def test_status_marks_existing_incomplete_training_as_train_corrupt(tmp_path):
    generated_root = tmp_path / "generated"
    train_config = generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0" / "train.yaml"
    train_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs" / "train" / "frame_stack" / "fixed_l1" / "fs4" / "seed_0"
    _write_train_config(train_config, train_dir=train_dir, output_dir=output_dir)
    _write_commands(generated_root, train_config)
    _write_checkpoint(train_dir, "flappy_frame_stack_fixed_l1_fs4_seed0", env_steps=512)

    rows = collect_status_rows(generated_root, active_grace_seconds=0)

    assert rows[0]["status"] == "train_corrupt"
    assert rows[0]["latest_env_steps"] == "512"


def test_repair_train_removes_local_outputs_and_wandb_run(tmp_path, monkeypatch):
    from scripts.manage_factor_sweeps import repair_train

    generated_root = tmp_path / "generated"
    train_config = generated_root / "frame_stack" / "fixed_l1" / "fs4" / "seed_0" / "train.yaml"
    train_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs" / "train" / "frame_stack" / "fixed_l1" / "fs4" / "seed_0"
    _write_train_config(
        train_config,
        train_dir=train_dir,
        output_dir=output_dir,
    )
    _write_commands(generated_root, train_config)
    _write_checkpoint(train_dir, "flappy_frame_stack_fixed_l1_fs4_seed0", env_steps=512)
    output_dir.mkdir(parents=True)
    (output_dir / "partial.txt").write_text("partial\n", encoding="utf-8")

    deleted_runs = []

    class FakeRun:
        def __init__(self, path: str):
            self.path = path

        def delete(self):
            deleted_runs.append(self.path)

    class FakeApi:
        def run(self, path: str):
            return FakeRun(path)

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Api = FakeApi
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rows = collect_status_rows(generated_root, active_grace_seconds=0)

    repair_train(rows, train_config=str(train_config), wandb_name=None, wandb_entity="entity")

    assert not (train_dir / "flappy_frame_stack_fixed_l1_fs4_seed0").exists()
    assert not output_dir.exists()
    assert deleted_runs == ["entity/latency-sensitive-bench/flappy-fs4-fixed_l1-s0"]
