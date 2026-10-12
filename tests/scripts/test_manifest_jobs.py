from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from helpers import REPO_ROOT

import scripts.experiment_runner.run_manifest_jobs as run_manifest_jobs
from scripts.experiment_runner.job_manifest import (
    MANIFEST_V2_COLUMNS,
    event_log_path_for_manifest,
    load_events,
    write_manifest,
)
from scripts.experiment_runner.run_manifest_jobs import run_jobs


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
        ("eval_succeeded", "complete"),
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


def test_eval_stage_only_evaluates(tmp_path):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)

    for _ in range(2):
        assert run_jobs(
            manifest_path=manifest_path,
            gpus=["0"],
            stage="eval",
        ) == 0

    assert not train_mark.exists()
    assert eval_mark.is_file()
    events = load_events(event_log_path_for_manifest(manifest_path))
    assert [event["stage"] for event in events] == ["eval_started", "eval_succeeded"]


def test_dry_run_does_not_execute_or_record_success(tmp_path, capsys):
    manifest_path, train_mark, eval_mark = _write_single_job(tmp_path)

    assert run_jobs(
        manifest_path=manifest_path,
        gpus=["0"],
        stage="all",
        dry_run=True,
    ) == 0

    assert not train_mark.exists()
    assert not eval_mark.exists()
    assert load_events(event_log_path_for_manifest(manifest_path)) == []
    output = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert str(train_mark) in output
    assert str(eval_mark) in output


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
