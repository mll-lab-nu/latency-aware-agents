from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from helpers import REPO_ROOT

import scripts.experiment_runner.run_manifest_jobs as run_manifest_jobs
from scripts.experiment_runner.job_manifest import (
    MANIFEST_V2_COLUMNS,
    append_event,
    event_log_path_for_manifest,
    load_events,
    load_manifest,
    write_manifest,
)
from scripts.experiment_runner.run_manifest_jobs import _default_uploader, run_jobs


def _write_single_job(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_mark = tmp_path / "train"
    eval_mark = tmp_path / "eval"
    manifest_path = tmp_path / "manifest.tsv"
    row = {column: "" for column in MANIFEST_V2_COLUMNS}
    row.update(
        {
            "schema_version": "v2",
            "job_id": "job1",
            "train_command": (
                f"{sys.executable} -c \"from pathlib import Path; "
                f"Path(r'{train_mark}').write_text('1')\""
            ),
            "eval_command": (
                f"{sys.executable} -c \"from pathlib import Path; "
                f"Path(r'{eval_mark}').write_text('1')\""
            ),
            "train_upload_repo_id": "repo-model",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "repo-dataset",
            "eval_upload_repo_type": "dataset",
        }
    )
    write_manifest(manifest_path, [row])
    return manifest_path, train_mark, eval_mark


def test_train_stage_only_trains_and_resumes_from_success(tmp_path):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)

    for _ in range(2):
        assert run_jobs(
            manifest_path=manifest_path,
            gpus=["0"],
            upload_workers=1,
            max_upload_attempts=1,
            stage="train",
        ) == 0

    assert train_mark.is_file()
    assert not eval_mark.exists()
    events = load_events(event_log_path_for_manifest(manifest_path))
    assert [event["stage"] for event in events] == ["train_started", "train_succeeded"]


@pytest.mark.parametrize(
    ("recorded_stage", "expected_stage"),
    [
        ("train_started", "train"),
        ("train_succeeded", "eval"),
        ("eval_started", "eval"),
        ("eval_failed", "eval"),
        ("eval_succeeded", "upload"),
        ("upload_started", "upload"),
    ],
)
def test_resume_stage_keeps_interrupted_job_at_its_current_pipeline_stage(
    tmp_path,
    recorded_stage,
    expected_stage,
):
    train_artifact_dir = tmp_path / "train_artifact"
    train_artifact_dir.mkdir()
    (train_artifact_dir / "config.json").write_text("{}")
    assert run_manifest_jobs._resume_stage(
        {"job_id": "job1", "train_artifact_dir": str(train_artifact_dir)},
        {"job1": recorded_stage},
    ) == expected_stage


@pytest.mark.parametrize("recorded_stage", ["train_succeeded", "eval_started", "eval_failed"])
def test_resume_stage_retrains_when_train_artifact_was_deleted(tmp_path, recorded_stage):
    assert run_manifest_jobs._resume_stage(
        {"job_id": "job1", "train_artifact_dir": str(tmp_path / "deleted_train_artifact")},
        {"job1": recorded_stage},
    ) == "train"


def test_eval_stage_does_not_train_or_upload(tmp_path):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)
    uploads: list[str] = []

    for _ in range(2):
        assert run_jobs(
            manifest_path=manifest_path,
            gpus=["0"],
            upload_workers=1,
            max_upload_attempts=1,
            stage="eval",
            uploader=lambda job: uploads.append(job["job_id"]),
        ) == 0

    assert not train_mark.exists()
    assert eval_mark.is_file()
    assert uploads == []
    events = load_events(event_log_path_for_manifest(manifest_path))
    assert [event["stage"] for event in events] == ["eval_started", "eval_succeeded"]


def test_upload_stage_needs_no_gpu_and_retries_failed_upload(tmp_path):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)
    event_log = event_log_path_for_manifest(manifest_path)
    append_event(event_log, {"job_id": "job1", "stage": "eval_succeeded", "ts": 1.0})

    def fail_upload(job: dict[str, str]) -> None:
        raise RuntimeError(job["job_id"])

    assert run_jobs(
        manifest_path=manifest_path,
        gpus=[],
        upload_workers=1,
        max_upload_attempts=1,
        stage="upload",
        uploader=fail_upload,
    ) == 1
    uploads: list[str] = []
    assert run_jobs(
        manifest_path=manifest_path,
        gpus=[],
        upload_workers=1,
        max_upload_attempts=1,
        stage="upload",
        uploader=lambda job: uploads.append(job["job_id"]),
    ) == 0

    assert uploads == ["job1"]
    assert run_jobs(
        manifest_path=manifest_path,
        gpus=[],
        upload_workers=1,
        max_upload_attempts=1,
        stage="upload",
        uploader=lambda job: uploads.append(job["job_id"]),
    ) == 0
    assert uploads == ["job1"]
    assert not train_mark.exists()
    assert not eval_mark.exists()
    assert [event["stage"] for event in load_events(event_log)][-3:] == [
        "upload_started",
        "upload_succeeded",
        "job_complete",
    ]


def test_upload_stage_resumes_interrupted_upload(tmp_path):
    manifest_path, _, _ = _write_single_job(tmp_path)
    append_event(
        event_log_path_for_manifest(manifest_path),
        {"job_id": "job1", "stage": "upload_started", "ts": 1.0},
    )
    uploads: list[str] = []

    assert run_jobs(
        manifest_path=manifest_path,
        gpus=[],
        upload_workers=1,
        max_upload_attempts=1,
        stage="upload",
        uploader=lambda job: uploads.append(job["job_id"]),
    ) == 0

    assert uploads == ["job1"]


def test_dry_run_does_not_execute_or_record_success(tmp_path, capsys):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)

    assert run_jobs(
        manifest_path=manifest_path,
        gpus=["0"],
        upload_workers=1,
        max_upload_attempts=1,
        stage="all",
        dry_run=True,
    ) == 0

    assert not train_mark.exists()
    assert not eval_mark.exists()
    assert load_events(event_log_path_for_manifest(manifest_path)) == []
    output = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert "upload job1: repo-dataset, repo-model" in output


def test_cli_requires_gpus_except_for_upload_stage():
    with pytest.raises(SystemExit):
        run_manifest_jobs.main(["--manifest", "manifest.tsv", "--stage", "train"])


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


def test_sigint_cancels_queue_and_kills_all_active_process_groups(tmp_path):
    helper_path = tmp_path / "ignore_term_job.py"
    helper_path.write_text(
        """from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ]
)
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "leader_pid": os.getpid(),
            "leader_pgid": os.getpgid(0),
            "child_pid": child.pid,
            "child_pgid": os.getpgid(child.pid),
        }
    ),
    encoding="utf-8",
)
time.sleep(60)
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.tsv"
    markers = [tmp_path / f"job_{idx}.json" for idx in range(4)]
    rows = []
    for idx, marker in enumerate(markers):
        row = {column: "" for column in MANIFEST_V2_COLUMNS}
        row.update(
            {
                "schema_version": "v2",
                "job_id": f"job{idx}",
                "train_command": f"exec {shlex.join([sys.executable, str(helper_path), str(marker)])}",
            }
        )
        rows.append(row)
    write_manifest(manifest_path, rows)

    runner = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.experiment_runner.run_manifest_jobs",
            "--manifest",
            str(manifest_path),
            "--stage",
            "train",
            "--gpus",
            "0,1",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process_groups: set[int] = set()
    try:
        deadline = time.time() + 10.0
        while sum(marker.is_file() for marker in markers) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert [marker.is_file() for marker in markers] == [True, True, False, False]
        for marker in markers[:2]:
            process_info = json.loads(marker.read_text(encoding="utf-8"))
            assert process_info["leader_pgid"] == process_info["child_pgid"]
            process_groups.add(int(process_info["leader_pgid"]))

        runner.send_signal(signal.SIGINT)
        time.sleep(0.1)
        runner.send_signal(signal.SIGINT)
        stdout, stderr = runner.communicate(timeout=10.0)
        assert runner.returncode != 0, (stdout, stderr)
        assert [marker.is_file() for marker in markers] == [True, True, False, False]

        deadline = time.time() + 5.0
        while process_groups and time.time() < deadline:
            for process_group in list(process_groups):
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    process_groups.remove(process_group)
            time.sleep(0.05)
        assert not process_groups

        events = load_events(event_log_path_for_manifest(manifest_path))
        assert [event["stage"] for event in events] == ["train_started", "train_started"]
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
        for process_group in process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigint_does_not_wait_for_in_flight_upload(tmp_path):
    manifest_path = tmp_path / "manifest.tsv"
    row = {column: "" for column in MANIFEST_V2_COLUMNS}
    row.update(
        {
            "schema_version": "v2",
            "job_id": "job0",
            "train_upload_repo_id": "repo-model",
            "train_upload_repo_type": "model",
        }
    )
    write_manifest(manifest_path, [row])
    append_event(
        event_log_path_for_manifest(manifest_path),
        {"job_id": "job0", "stage": "eval_succeeded"},
    )
    upload_started = tmp_path / "upload_started"
    helper_path = tmp_path / "blocking_upload_runner.py"
    helper_path.write_text(
        """from pathlib import Path
import sys
import time

from scripts.experiment_runner.run_manifest_jobs import run_jobs

manifest_path = Path(sys.argv[1])
upload_started = Path(sys.argv[2])

def blocking_uploader(job):
    del job
    upload_started.write_text("started", encoding="utf-8")
    time.sleep(60)

raise SystemExit(
    run_jobs(
        manifest_path=manifest_path,
        gpus=[],
        upload_workers=1,
        max_upload_attempts=1,
        stage="upload",
        uploader=blocking_uploader,
    )
)
""",
        encoding="utf-8",
    )
    runner = subprocess.Popen(
        [sys.executable, str(helper_path), str(manifest_path), str(upload_started)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 10.0
        while not upload_started.is_file() and time.time() < deadline:
            time.sleep(0.05)
        assert upload_started.is_file()

        started_at = time.monotonic()
        runner.send_signal(signal.SIGINT)
        stdout, stderr = runner.communicate(timeout=5.0)
        assert runner.returncode != 0, (stdout, stderr)
        assert time.monotonic() - started_at < 3.0
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


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
            "train_upload_repo_id": "latency-sensitive-bench/sweep-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/sweep-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert calls[0] == ("init", "env-token")
    assert calls[1] == ("create_repo", "latency-sensitive-bench/sweep-evals")
    assert calls[2] == ("upload_folder", "eval/job1")
    assert calls[3] == ("create_repo", "latency-sensitive-bench/sweep-models")
    assert calls[4] == ("upload_folder", "train/job1")


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
            "train_upload_repo_id": "latency-sensitive-bench/sweep-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/sweep-evals",
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
            "train_upload_repo_id": "latency-sensitive-bench/sweep-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/sweep-evals",
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
            "train_upload_repo_id": "latency-sensitive-bench/sweep-models",
            "train_upload_repo_type": "model",
            "eval_upload_repo_id": "latency-sensitive-bench/sweep-evals",
            "eval_upload_repo_type": "dataset",
            "train_artifact_dir": str(tmp_path / "train_artifacts"),
            "eval_artifact_dir": str(tmp_path / "eval_artifacts"),
            "train_artifact_repo_path": "train/job1",
            "eval_artifact_repo_path": "eval/job1",
        }
    )

    assert uploads[0] == {
        "kind": "create_repo",
        "repo_id": "latency-sensitive-bench/sweep-evals",
        "repo_type": "dataset",
    }
    assert uploads[1]["repo_id"] == "latency-sensitive-bench/sweep-evals"
    assert uploads[1]["repo_type"] == "dataset"
    assert uploads[1]["path_in_repo"] == "eval/job1"
    assert uploads[1]["ignore_patterns"] == run_manifest_jobs.UPLOAD_IGNORE_PATTERNS
    assert uploads[2] == {
        "kind": "create_repo",
        "repo_id": "latency-sensitive-bench/sweep-models",
        "repo_type": "model",
    }
    assert uploads[3]["repo_id"] == "latency-sensitive-bench/sweep-models"
    assert uploads[3]["repo_type"] == "model"
    assert uploads[3]["path_in_repo"] == "train/job1"
    assert uploads[3]["ignore_patterns"] == run_manifest_jobs.UPLOAD_IGNORE_PATTERNS
