from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from scripts.experiment_runner.job_manifest import (
    append_event,
    event_log_path_for_manifest,
    load_events,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESS_GROUP_TERM_TIMEOUT_SEC = 2.0
PROCESS_GROUP_KILL_TIMEOUT_SEC = 1.0
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()


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
        remaining = deadline - time.time()
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        _signal_process_group(process, signal.SIGKILL)
    kill_deadline = time.time() + PROCESS_GROUP_KILL_TIMEOUT_SEC
    for process in processes:
        remaining = kill_deadline - time.time()
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


def _resume_stage(job: dict[str, str], states: dict[str, str]) -> str:
    stage = states.get(str(job["job_id"]), "")
    if stage in {"job_complete", "eval_succeeded"}:
        return "complete"
    if stage in {"train_succeeded", "eval_started", "eval_failed"}:
        train_config = Path(job["train_artifact_dir"]) / "config.json"
        return "eval" if train_config.is_file() else "train"
    return "train"


def run_jobs(
    *,
    manifest_path: str | Path,
    gpus: list[str],
    stage: str = "all",
    match: str = "",
    dry_run: bool = False,
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
    else:
        succeeded = _successful_jobs(events, stage + "_succeeded")
        jobs = [row for row in rows if row["job_id"] not in succeeded]
    job_queue: queue.Queue[dict[str, str] | None] = queue.Queue()
    failures: list[tuple[str, str]] = []
    failure_lock = threading.Lock()
    cancel_event = threading.Event()
    signal_event = threading.Event()
    cleanup_started = False
    for job in jobs:
        job_queue.put(job)
    for _ in gpus:
        job_queue.put(None)

    def train_worker(gpu: str) -> None:
        while True:
            job = job_queue.get()
            try:
                if job is None:
                    break
                if cancel_event.is_set():
                    continue
                next_stage = _resume_stage(job, states) if stage == "all" else stage
                job_id = str(job["job_id"])
                for current_stage in ("train", "eval"):
                    if next_stage != current_stage:
                        continue
                    if cancel_event.is_set():
                        break
                    command = str(job[current_stage + "_command"])
                    if dry_run:
                        print(f"[dry-run] CUDA_VISIBLE_DEVICES={gpu} {command}", flush=True)
                    else:
                        write_event({"job_id": job_id, "stage": current_stage + "_started", "gpu": gpu, "ts": _now()})
                        try:
                            started = _run_command(command, gpu=gpu, cancel_event=cancel_event)
                        except subprocess.CalledProcessError:
                            if not cancel_event.is_set():
                                failed_stage = current_stage + "_failed"
                                write_event({"job_id": job_id, "stage": failed_stage, "gpu": gpu, "ts": _now()})
                                with failure_lock:
                                    failures.append((job_id, failed_stage))
                            break
                        if not started or cancel_event.is_set():
                            break
                        write_event({"job_id": job_id, "stage": current_stage + "_succeeded", "gpu": gpu, "ts": _now()})
                    next_stage = "eval" if stage == "all" and current_stage == "train" else "complete"
                if stage == "all" and next_stage == "complete" and not dry_run and not cancel_event.is_set():
                    write_event({"job_id": job_id, "stage": "job_complete", "ts": _now()})
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=train_worker, args=(gpu,), name=f"train-{gpu}", daemon=True)
               for gpu in gpus]
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_active_processes(signum, frame):
        del frame
        if signal_event.is_set():
            return
        signal_event.set()
        cancel_event.set()
        if not cleanup_started:
            raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGINT, stop_active_processes)
    signal.signal(signal.SIGTERM, stop_active_processes)
    try:
        for thread in threads:
            thread.start()
        job_queue.join()
        return 1 if failures else 0
    finally:
        cleanup_started = True
        cancel_event.set()
        _terminate_active_processes(signal.SIGTERM)
        for thread in threads:
            thread.join()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manifest-defined train/eval jobs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--match", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    gpus = [] if args.gpus is None else [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one device")
    return run_jobs(
        manifest_path=args.manifest,
        gpus=gpus,
        stage=str(args.stage),
        match=str(args.match),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
