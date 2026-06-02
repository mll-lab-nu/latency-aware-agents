from __future__ import annotations

import argparse
import os
import queue
import subprocess
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


def _event_writer(event_log: Path) -> Callable[[dict[str, Any]], None]:
    lock = threading.Lock()

    def write(event: dict[str, Any]) -> None:
        with lock:
            append_event(event_log, event)

    return write


def _run_command(command: str, *, gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, shell=True, check=True, env=env)


def _default_uploader(job: dict[str, str]) -> None:
    repo_id = str(job["upload_repo_id"])
    if not repo_id:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=str(job["upload_repo_type"]),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type=str(job["upload_repo_type"]),
        folder_path=str(job["train_artifact_dir"]),
        path_in_repo=str(job["train_artifact_repo_path"]),
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type=str(job["upload_repo_type"]),
        folder_path=str(job["eval_artifact_dir"]),
        path_in_repo=str(job["eval_artifact_repo_path"]),
    )


def _resume_stage(job: dict[str, str], states: dict[str, str]) -> str:
    stage = states.get(str(job["job_id"]), "")
    if stage in {"job_complete", "upload_skipped", "upload_succeeded"}:
        return "complete"
    if stage == "eval_succeeded":
        return "upload"
    if stage == "train_succeeded":
        return "eval"
    return "train"


def run_jobs(
    *,
    manifest_path: str | Path,
    gpus: list[str],
    upload_workers: int,
    max_upload_attempts: int,
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
    states = _job_states(load_events(event_log))
    jobs = [row for row in rows if _resume_stage(row, states) != "complete"]
    job_queue: queue.Queue[dict[str, str] | None] = queue.Queue()
    upload_queue: queue.Queue[UploadTask | None] = queue.Queue()
    failures: list[tuple[str, str]] = []
    failure_lock = threading.Lock()
    uploader_fn = _default_uploader if uploader is None else uploader

    for job in jobs:
        job_queue.put(job)
    for _ in gpus:
        job_queue.put(None)

    def record_failure(job_id: str, stage: str) -> None:
        with failure_lock:
            failures.append((job_id, stage))

    def train_worker(gpu: str) -> None:
        while True:
            job = job_queue.get()
            if job is None:
                job_queue.task_done()
                break
            stage = _resume_stage(job, states)
            job_id = str(job["job_id"])
            try:
                if stage == "train":
                    write_event({"job_id": job_id, "stage": "train_started", "gpu": gpu, "ts": _now()})
                    if not dry_run:
                        _run_command(str(job["train_command"]), gpu=gpu)
                    write_event({"job_id": job_id, "stage": "train_succeeded", "gpu": gpu, "ts": _now()})
                    stage = "eval"
                if stage == "eval":
                    write_event({"job_id": job_id, "stage": "eval_started", "gpu": gpu, "ts": _now()})
                    if not dry_run:
                        _run_command(str(job["eval_command"]), gpu=gpu)
                    write_event({"job_id": job_id, "stage": "eval_succeeded", "gpu": gpu, "ts": _now()})
                    stage = "upload"
                if stage == "upload":
                    upload_queue.put(UploadTask(job))
            except subprocess.CalledProcessError:
                failed_stage = "train_failed" if stage == "train" else "eval_failed"
                write_event({"job_id": job_id, "stage": failed_stage, "gpu": gpu, "ts": _now()})
                record_failure(job_id, failed_stage)
            finally:
                job_queue.task_done()

    def upload_worker() -> None:
        while True:
            task = upload_queue.get()
            if task is None:
                upload_queue.task_done()
                break
            job = task.job
            job_id = str(job["job_id"])
            repo_id = str(job["upload_repo_id"])
            if not repo_id:
                write_event({"job_id": job_id, "stage": "upload_skipped", "ts": _now()})
                write_event({"job_id": job_id, "stage": "job_complete", "ts": _now()})
                upload_queue.task_done()
                continue
            attempt = 0
            success = False
            while attempt < max_upload_attempts and not success:
                attempt += 1
                write_event({"job_id": job_id, "stage": "upload_started", "attempt": attempt, "ts": _now()})
                try:
                    if not dry_run:
                        uploader_fn(job)
                    write_event({"job_id": job_id, "stage": "upload_succeeded", "attempt": attempt, "ts": _now()})
                    write_event({"job_id": job_id, "stage": "job_complete", "ts": _now()})
                    success = True
                except Exception:
                    write_event({"job_id": job_id, "stage": "upload_failed", "attempt": attempt, "ts": _now()})
            if not success:
                record_failure(job_id, "upload_failed")
            upload_queue.task_done()

    upload_threads = [
        threading.Thread(target=upload_worker, name=f"upload-{idx}", daemon=True)
        for idx in range(int(upload_workers))
    ]
    train_threads = [
        threading.Thread(target=train_worker, args=(gpu,), name=f"train-{gpu}", daemon=True)
        for gpu in gpus
    ]
    for thread in upload_threads + train_threads:
        thread.start()
    job_queue.join()
    for _ in upload_threads:
        upload_queue.put(None)
    upload_queue.join()
    for thread in train_threads + upload_threads:
        thread.join()
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manifest-defined train/eval/upload jobs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--max-upload-attempts", type=int, default=3)
    parser.add_argument("--match", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    return run_jobs(
        manifest_path=args.manifest,
        gpus=gpus,
        upload_workers=int(args.upload_workers),
        max_upload_attempts=int(args.max_upload_attempts),
        match=str(args.match),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
