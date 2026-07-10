from __future__ import annotations

import os
import subprocess
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import scripts.experiment_runner.run_manifest_jobs as run_manifest_jobs
from scripts.experiment_runner.job_manifest import load_manifest, write_manifest
from scripts.experiment_runner.run_manifest_jobs import _default_uploader, run_jobs


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
            "train_upload_repo_id": "repo-model",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "repo-dataset",
            "eval_upload_repo_type": "dataset",
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
            "train_upload_repo_id": "repo-model",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "repo-dataset",
            "eval_upload_repo_type": "dataset",
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


def test_run_command_cleanup_terminates_child_process_group(tmp_path):
    pid_file = tmp_path / "pids.txt"
    command = (
        f"{sys.executable} -c "
        "\"import subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open(r'{pid_file}', 'w').write(str(child.pid)); "
        "time.sleep(60)\""
    )
    errors: list[BaseException] = []

    def run_command() -> None:
        try:
            run_manifest_jobs._run_command(command, gpu="0")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_command)
    thread.start()
    deadline = time.time() + 5.0
    while not pid_file.is_file() and time.time() < deadline:
        time.sleep(0.05)

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    run_manifest_jobs._terminate_active_processes(signal.SIGTERM)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert errors and isinstance(errors[0], subprocess.CalledProcessError)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_default_uploader_prefers_env_token_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=dotenv-token\n", encoding="utf-8")
    monkeypatch.setattr(run_manifest_jobs, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "env-token")

    calls: list[tuple[str, object]] = []

    class FakeApi:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs["repo_id"]))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs["path_in_repo"]))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _default_uploader(
        {
            "train_upload_repo_id": "latency-sensitive-bench/factor-sweeps-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/factor-sweeps-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert calls[0] == ("init", "env-token")
    assert calls[1] == ("create_repo", "latency-sensitive-bench/factor-sweeps-models")
    assert calls[2] == ("upload_folder", "train/job1")
    assert calls[3] == ("create_repo", "latency-sensitive-bench/factor-sweeps-evals")
    assert calls[4] == ("upload_folder", "eval/job1")


def test_default_uploader_falls_back_to_dotenv_token(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HUGGINGFACE_HUB_TOKEN=dotenv-token\n", encoding="utf-8")
    monkeypatch.setattr(run_manifest_jobs, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    init_tokens: list[object] = []

    class FakeApi:
        def __init__(self, *, token=None):
            init_tokens.append(token)

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return None

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _default_uploader(
        {
            "train_upload_repo_id": "latency-sensitive-bench/factor-sweeps-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/factor-sweeps-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert init_tokens == ["dotenv-token"]


def test_default_uploader_uses_cached_login_when_no_token_available(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_manifest_jobs, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    init_tokens: list[object] = []

    class FakeApi:
        def __init__(self, *, token=None):
            init_tokens.append(token)

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return None

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _default_uploader(
        {
            "train_upload_repo_id": "latency-sensitive-bench/factor-sweeps-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/factor-sweeps-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert init_tokens == [None]


def test_default_uploader_splits_model_and_dataset_repos_and_filters_noise(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_manifest_jobs, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    uploads: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token=None):
            return None

        def create_repo(self, **kwargs):
            uploads.append({"kind": "create_repo", "repo_id": kwargs["repo_id"], "repo_type": kwargs["repo_type"]})

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _default_uploader(
        {
            "train_upload_repo_id": "latency-sensitive-bench/factor-sweeps-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/factor-sweeps-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert uploads[0] == {
        "kind": "create_repo",
        "repo_id": "latency-sensitive-bench/factor-sweeps-models",
        "repo_type": "model",
    }
    assert uploads[1]["repo_id"] == "latency-sensitive-bench/factor-sweeps-models"
    assert uploads[1]["repo_type"] == "model"
    assert uploads[1]["path_in_repo"] == "train/job1"
    assert uploads[1]["ignore_patterns"] == run_manifest_jobs.UPLOAD_IGNORE_PATTERNS
    assert uploads[2] == {
        "kind": "create_repo",
        "repo_id": "latency-sensitive-bench/factor-sweeps-evals",
        "repo_type": "dataset",
    }
    assert uploads[3]["repo_id"] == "latency-sensitive-bench/factor-sweeps-evals"
    assert uploads[3]["repo_type"] == "dataset"
    assert uploads[3]["path_in_repo"] == "eval/job1"
    assert uploads[3]["ignore_patterns"] == run_manifest_jobs.UPLOAD_IGNORE_PATTERNS
