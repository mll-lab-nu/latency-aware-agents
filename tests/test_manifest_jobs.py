from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

from scripts.experiment_runner.job_manifest import load_manifest, write_manifest
from scripts.experiment_runner.migrate_manifest_v2 import main as migrate_manifest_v2
from scripts.experiment_runner.run_manifest_jobs import run_jobs


def _write_config(path: Path, *, experiment_name: str, train_dir: Path, output_dir: Path, eval_output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_cfg = {
        "experiment": {"name": experiment_name, "seed": 7},
        "backend": {"train_dir": str(train_dir), "run_mode": "train"},
        "env": {"name": "flappy", "env_fps": 30, "obs_fps": 15, "frame_stack": 4},
        "latency": {"type": "fixed", "fixed_latency_ms": 100.0},
        "training": {"train_for_env_steps": 1000},
        "logging": {"output_dir": str(output_dir), "wandb_name": experiment_name, "wandb_group": "group"},
    }
    eval_cfg = {
        "experiment": {"name": experiment_name, "seed": 7},
        "backend": {"train_dir": str(train_dir), "run_mode": "eval"},
        "env": {"name": "flappy", "env_fps": 30, "obs_fps": 15, "frame_stack": 4},
        "latency": {"type": "fixed", "fixed_latency_ms": 100.0},
        "logging": {"output_dir": str(eval_output_dir)},
    }
    path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    path.with_name("latency_bench_eval.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")


def test_migrate_legacy_manifest_to_v2(tmp_path):
    generated_root = tmp_path / "generated"
    train_path = generated_root / "flappy" / "latency_3" / "seed_0" / "train.yaml"
    _write_config(
        train_path,
        experiment_name="flappy_sf_same_latency_l3_seed0",
        train_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs" / "train",
        eval_output_dir=tmp_path / "outputs" / "eval",
    )
    legacy_manifest = generated_root / "manifest.tsv"
    legacy_manifest.write_text(
        "\n".join(
            [
                "env\tlatency\tseed\ttrain_config\tlatency_bench_eval_config\ttrain_command\tlatency_bench_eval_command",
                "\t".join(
                    [
                        "flappy",
                        "3",
                        "0",
                        str(train_path),
                        str(train_path.with_name("latency_bench_eval.yaml")),
                        f"python -m training.train --config {train_path}",
                        f"python -m latency_bench.run --config {train_path.with_name('latency_bench_eval.yaml')}",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    migrate_manifest_v2(["--input", str(legacy_manifest)])

    rows = load_manifest(legacy_manifest)
    assert rows[0]["schema_version"] == "v2"
    assert rows[0]["experiment_family"] == "same_latency"
    assert rows[0]["sweep_name"] == "same_latency"
    assert rows[0]["observation_stride"] == "2"


def test_runner_uploads_in_background(tmp_path):
    marks = tmp_path / "marks"
    marks.mkdir()
    train_dir = tmp_path / "checkpoints"
    manifest_path = tmp_path / "manifest.tsv"
    job1_train = f"{sys.executable} -c \"from pathlib import Path; Path(r'{marks / 'job1_train'}').write_text('1')\""
    job1_eval = f"{sys.executable} -c \"from pathlib import Path; Path(r'{marks / 'job1_eval'}').write_text('1')\""
    job2_train = f"{sys.executable} -c \"from pathlib import Path; Path(r'{marks / 'job2_train'}').write_text('1')\""
    job2_eval = f"{sys.executable} -c \"from pathlib import Path; Path(r'{marks / 'job2_eval'}').write_text('1')\""
    rows = [
        {
            "schema_version": "v2",
            "job_id": "job1",
            "env": "flappy",
            "experiment_family": "same_latency",
            "sweep_name": "same_latency",
            "latency_label": "1",
            "seed": "1",
            "frame_stack": "4",
            "obs_fps": "15",
            "observation_stride": "2",
            "train_config": "a/train.yaml",
            "eval_kind": "latency_bench",
            "eval_config": "a/eval.yaml",
            "train_command": job1_train,
            "eval_command": job1_eval,
            "run_name": "job1",
            "wandb_group": "",
            "wandb_name": "job1",
            "train_dir": str(train_dir),
            "checkpoint_dir": str(train_dir / "job1" / "checkpoint_p0"),
            "train_output_dir": str(tmp_path / "out1"),
            "eval_output_dir": str(tmp_path / "eval1"),
            "train_artifact_dir": str(tmp_path / "art1"),
            "eval_artifact_dir": str(tmp_path / "eval_art1"),
            "upload_repo_id": "repo",
            "upload_repo_type": "model",
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
            "latency_kind": "fixed",
            "latency_fixed_frames": "1",
            "latency_mean_frames": "",
            "latency_std_frames": "",
            "latency_min_frames": "",
            "latency_max_frames": "",
        },
        {
            "schema_version": "v2",
            "job_id": "job2",
            "env": "flappy",
            "experiment_family": "same_latency",
            "sweep_name": "same_latency",
            "latency_label": "2",
            "seed": "2",
            "frame_stack": "4",
            "obs_fps": "15",
            "observation_stride": "2",
            "train_config": "b/train.yaml",
            "eval_kind": "latency_bench",
            "eval_config": "b/eval.yaml",
            "train_command": job2_train,
            "eval_command": job2_eval,
            "run_name": "job2",
            "wandb_group": "",
            "wandb_name": "job2",
            "train_dir": str(train_dir),
            "checkpoint_dir": str(train_dir / "job2" / "checkpoint_p0"),
            "train_output_dir": str(tmp_path / "out2"),
            "eval_output_dir": str(tmp_path / "eval2"),
            "train_artifact_dir": str(tmp_path / "art2"),
            "eval_artifact_dir": str(tmp_path / "eval_art2"),
            "upload_repo_id": "repo",
            "upload_repo_type": "model",
            "train_artifact_repo_path": "train/job2",
            "eval_artifact_repo_path": "eval/job2",
            "latency_kind": "fixed",
            "latency_fixed_frames": "2",
            "latency_mean_frames": "",
            "latency_std_frames": "",
            "latency_min_frames": "",
            "latency_max_frames": "",
        },
    ]
    write_manifest(manifest_path, rows)
    upload_started = tmp_path / "upload_started"
    upload_finished = tmp_path / "upload_finished"

    def uploader(job: dict[str, str]) -> None:
        Path(upload_started).write_text(job["job_id"], encoding="utf-8")
        time.sleep(0.5)
        Path(upload_finished).write_text(job["job_id"], encoding="utf-8")

    rc = run_jobs(
        manifest_path=manifest_path,
        gpus=["0"],
        upload_workers=1,
        max_upload_attempts=1,
        uploader=uploader,
    )

    assert rc == 0
    assert (marks / "job2_train").is_file()
    assert upload_started.is_file()
    assert upload_finished.is_file()
    assert (marks / "job2_train").stat().st_mtime < upload_finished.stat().st_mtime
