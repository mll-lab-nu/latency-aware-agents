from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.experiment_runner.job_manifest import (
    append_event,
    event_log_path_for_manifest,
    load_events,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
UPLOAD_IGNORE_PATTERNS = [
    "wandb/**",
    "wandb_logs/**",
    "**/wandb/**",
    "**/wandb_logs/**",
    "**/events.out.tfevents.*",
    "**/*_state/**",
    "**/optimizer/**",
    "**/optimizers/**",
    "**/.cache/**",
    "**/__pycache__/**",
]
PROCESS_GROUP_TERM_TIMEOUT_SEC = 2.0
PROCESS_GROUP_KILL_TIMEOUT_SEC = 1.0
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()


@dataclass(frozen=True)
class UploadTask:
    job: dict[str, str]


def _now() -> float:
    return time.time()


def _job_states(events: list[dict[str, Any]]) -> dict[str, str]:
    states: dict[str, str] = {}
    for event in events:
        states[str(event["job_id"])] = str(event["stage"])
    return states


def _successful_jobs(events: list[dict[str, Any]], stage: str) -> set[str]:
    return {
        str(event["job_id"])
        for event in events
        if str(event["stage"]) == stage
    }


def _event_writer(event_log: Path) -> Callable[[dict[str, Any]], None]:
    lock = threading.Lock()

    def write(event: dict[str, Any]) -> None:
        with lock:
            append_event(event_log, event)

    return write


def _forget_process(process: subprocess.Popen) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.discard(process)


def _active_processes() -> list[subprocess.Popen]:
    with _ACTIVE_PROCESSES_LOCK:
        return list(_ACTIVE_PROCESSES)


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _terminate_active_processes(sig: int = signal.SIGTERM) -> None:
    processes = _active_processes()
    for process in processes:
        _signal_process_group(process, sig)
    deadline = time.time() + PROCESS_GROUP_TERM_TIMEOUT_SEC
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        _signal_process_group(process, signal.SIGKILL)
    kill_deadline = time.time() + PROCESS_GROUP_KILL_TIMEOUT_SEC
    for process in processes:
        remaining = max(0.0, kill_deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass


def _run_command(
    command: str,
    *,
    gpu: str,
    cancel_event: threading.Event | None = None,
) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with _ACTIVE_PROCESSES_LOCK:
        if cancel_event is not None and cancel_event.is_set():
            return False
        process = subprocess.Popen(command, shell=True, env=env, start_new_session=True)
        _ACTIVE_PROCESSES.add(process)
    try:
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        return True
    finally:
        _forget_process(process)


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _resolve_hf_token() -> str | None:
    for env_var in HF_TOKEN_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    dotenv_values = _load_dotenv(REPO_ROOT / ".env")
    for env_var in HF_TOKEN_ENV_VARS:
        value = dotenv_values.get(env_var)
        if value:
            return value
    return None


def _default_uploader(job: dict[str, str]) -> None:
    train_repo_id = str(job["train_upload_repo_id"])
    eval_repo_id = str(job["eval_upload_repo_id"])
    if not train_repo_id and not eval_repo_id:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_hf_token())
    if eval_repo_id:
        eval_artifact_dir = Path(str(job["eval_artifact_dir"]))
        if eval_artifact_dir.is_dir():
            api.create_repo(
                repo_id=eval_repo_id,
                repo_type=str(job["eval_upload_repo_type"]),
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=eval_repo_id,
                repo_type=str(job["eval_upload_repo_type"]),
                folder_path=str(eval_artifact_dir),
                path_in_repo=str(job["eval_artifact_repo_path"]),
                ignore_patterns=UPLOAD_IGNORE_PATTERNS,
            )
        else:
            print(
                f"[upload] skipping missing eval artifact dir: {eval_artifact_dir}",
                file=sys.stderr,
            )
    if train_repo_id:
        train_artifact_dir = Path(str(job["train_artifact_dir"]))
        if not train_artifact_dir.is_dir():
            raise ValueError(f"train artifact dir does not exist: {train_artifact_dir}")
        api.create_repo(
            repo_id=train_repo_id,
            repo_type=str(job["train_upload_repo_type"]),
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=train_repo_id,
            repo_type=str(job["train_upload_repo_type"]),
            folder_path=str(train_artifact_dir),
            path_in_repo=str(job["train_artifact_repo_path"]),
            ignore_patterns=UPLOAD_IGNORE_PATTERNS,
        )


def _resume_stage(job: dict[str, str], states: dict[str, str]) -> str:
    stage = states.get(str(job["job_id"]), "")
    if stage in {"job_complete", "upload_skipped", "upload_succeeded"}:
        return "complete"
    if stage in {"eval_succeeded", "upload_started", "upload_failed"}:
        return "upload"
    if stage in {"train_succeeded", "eval_started", "eval_failed"}:
        train_config = Path(job["train_artifact_dir"]) / "config.json"
        return "eval" if train_config.is_file() else "train"
    return "train"


def run_jobs(
    *,
    manifest_path: str | Path,
    gpus: list[str],
    upload_workers: int,
    max_upload_attempts: int,
    stage: str = "all",
    match: str = "",
    dry_run: bool = False,
    uploader: Callable[[dict[str, str]], None] | None = None,
) -> int:
    manifest_file = Path(manifest_path).expanduser().resolve()
    event_log = event_log_path_for_manifest(manifest_file)
    write_event = _event_writer(event_log)
    rows = load_manifest(manifest_file)
    if match:
        rows = [row for row in rows if match in row["job_id"]]
    events = load_events(event_log)
    states = _job_states(events)
    if stage == "all":
        jobs = [row for row in rows if _resume_stage(row, states) != "complete"]
    elif stage == "train":
        succeeded = _successful_jobs(events, "train_succeeded")
        jobs = [row for row in rows if row["job_id"] not in succeeded]
    elif stage == "eval":
        succeeded = _successful_jobs(events, "eval_succeeded")
        jobs = [row for row in rows if row["job_id"] not in succeeded]
    else:
        jobs = [
            row
            for row in rows
            if states.get(row["job_id"], "")
            in {"train_succeeded", "eval_succeeded", "upload_started", "upload_failed"}
        ]
    job_queue: queue.Queue[dict[str, str] | None] = queue.Queue()
    upload_queue: queue.Queue[UploadTask | None] = queue.Queue()
    failures: list[tuple[str, str]] = []
    failure_lock = threading.Lock()
    cancel_event = threading.Event()
    signal_event = threading.Event()
    cleanup_started = False
    uploader_fn = _default_uploader if uploader is None else uploader

    if stage != "upload":
        for job in jobs:
            job_queue.put(job)
        for _ in gpus:
            job_queue.put(None)
    else:
        for job in jobs:
            upload_queue.put(UploadTask(job))

    def record_failure(job_id: str, stage: str) -> None:
        with failure_lock:
            failures.append((job_id, stage))

    def train_worker(gpu: str) -> None:
        while True:
            job = job_queue.get()
            if job is None:
                job_queue.task_done()
                break
            if cancel_event.is_set():
                job_queue.task_done()
                continue
            next_stage = _resume_stage(job, states) if stage == "all" else stage
            job_id = str(job["job_id"])
            try:
                if next_stage == "train":
                    if cancel_event.is_set():
                        continue
                    if dry_run:
                        print(f"[dry-run] CUDA_VISIBLE_DEVICES={gpu} {job['train_command']}", flush=True)
                    else:
                        write_event({"job_id": job_id, "stage": "train_started", "gpu": gpu, "ts": _now()})
                        started = _run_command(
                            str(job["train_command"]),
                            gpu=gpu,
                            cancel_event=cancel_event,
                        )
                        if not started or cancel_event.is_set():
                            continue
                        write_event({"job_id": job_id, "stage": "train_succeeded", "gpu": gpu, "ts": _now()})
                    next_stage = "eval" if stage == "all" else "complete"
                if next_stage == "eval":
                    if cancel_event.is_set():
                        continue
                    if dry_run:
                        print(f"[dry-run] CUDA_VISIBLE_DEVICES={gpu} {job['eval_command']}", flush=True)
                    else:
                        write_event({"job_id": job_id, "stage": "eval_started", "gpu": gpu, "ts": _now()})
                        started = _run_command(
                            str(job["eval_command"]),
                            gpu=gpu,
                            cancel_event=cancel_event,
                        )
                        if not started or cancel_event.is_set():
                            continue
                        write_event({"job_id": job_id, "stage": "eval_succeeded", "gpu": gpu, "ts": _now()})
                    next_stage = "upload" if stage == "all" else "complete"
                if next_stage == "upload" and not cancel_event.is_set():
                    upload_queue.put(UploadTask(job))
            except subprocess.CalledProcessError:
                if cancel_event.is_set():
                    continue
                failed_stage = "train_failed" if next_stage == "train" else "eval_failed"
                write_event({"job_id": job_id, "stage": failed_stage, "gpu": gpu, "ts": _now()})
                record_failure(job_id, failed_stage)
            finally:
                job_queue.task_done()

    def upload_worker() -> None:
        while True:
            task = upload_queue.get()
            try:
                if task is None:
                    break
                if cancel_event.is_set():
                    continue
                job = task.job
                job_id = str(job["job_id"])
                train_repo_id = str(job["train_upload_repo_id"])
                eval_repo_id = str(job["eval_upload_repo_id"])
                if dry_run:
                    targets = [repo_id for repo_id in (eval_repo_id, train_repo_id) if repo_id]
                    print(f"[dry-run] upload {job_id}: {', '.join(targets) or 'no upload target'}", flush=True)
                    continue
                if not train_repo_id and not eval_repo_id:
                    write_event({"job_id": job_id, "stage": "upload_skipped", "ts": _now()})
                    write_event({"job_id": job_id, "stage": "job_complete", "ts": _now()})
                    continue
                attempt = 0
                success = False
                while attempt < max_upload_attempts and not success and not cancel_event.is_set():
                    attempt += 1
                    write_event({"job_id": job_id, "stage": "upload_started", "attempt": attempt, "ts": _now()})
                    try:
                        uploader_fn(job)
                        if cancel_event.is_set():
                            continue
                        write_event({"job_id": job_id, "stage": "upload_succeeded", "attempt": attempt, "ts": _now()})
                        write_event({"job_id": job_id, "stage": "job_complete", "ts": _now()})
                        success = True
                    except Exception as error:
                        if cancel_event.is_set():
                            continue
                        write_event(
                            {
                                "job_id": job_id,
                                "stage": "upload_failed",
                                "attempt": attempt,
                                "error": repr(error),
                                "ts": _now(),
                            }
                        )
                        print(f"upload failed for {job_id} (attempt {attempt}): {error!r}", file=sys.stderr)
                if not success and not cancel_event.is_set():
                    record_failure(job_id, "upload_failed")
            finally:
                upload_queue.task_done()

    upload_threads = [
        threading.Thread(target=upload_worker, name=f"upload-{idx}", daemon=True)
        for idx in range(int(upload_workers))
    ]
    train_threads = [
        threading.Thread(target=train_worker, args=(gpu,), name=f"train-{gpu}", daemon=True)
        for gpu in gpus if stage != "upload"
    ]
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_active_processes(signum, frame):
        nonlocal cleanup_started
        del frame
        if signal_event.is_set():
            return
        signal_event.set()
        cancel_event.set()
        if cleanup_started:
            return
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGINT, stop_active_processes)
    signal.signal(signal.SIGTERM, stop_active_processes)
    try:
        for thread in upload_threads + train_threads:
            thread.start()
        job_queue.join()
        upload_queue.join()
        return 1 if failures else 0
    finally:
        cleanup_started = True
        cancel_event.set()
        _terminate_active_processes(signal.SIGTERM)
        for thread in train_threads:
            thread.join()
        _terminate_active_processes(signal.SIGTERM)
        if signal_event.is_set():
            for _ in upload_threads:
                upload_queue.put(None)
        else:
            upload_queue.join()
            for _ in upload_threads:
                upload_queue.put(None)
            for thread in upload_threads:
                thread.join()
        _terminate_active_processes(signal.SIGTERM)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manifest-defined train/eval/upload jobs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "train", "eval", "upload"), default="all")
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--max-upload-attempts", type=int, default=3)
    parser.add_argument("--match", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage != "upload" and args.gpus is None:
        parser.error("--gpus is required for all, train, and eval stages")
    gpus = [] if args.gpus is None else [item.strip() for item in args.gpus.split(",") if item.strip()]
    return run_jobs(
        manifest_path=args.manifest,
        gpus=gpus,
        upload_workers=int(args.upload_workers),
        max_upload_attempts=int(args.max_upload_attempts),
        stage=str(args.stage),
        match=str(args.match),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
