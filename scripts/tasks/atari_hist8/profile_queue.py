"""Run the six published H8 workloads on two independent single-GPU lanes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.run_hist8_v2 import process_running


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/tasks/atari_hist8_v1"
MODELS = ("qwenoft", "qwengr00t", "qwenpi_v3")
TASKS = ("atlantis", "asterix")
REPO = "latency-sensitive-bench/benchmark-models"


def main() -> None:
    queue = OUTPUT / "profile_queue"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "controller.pid").write_text(str(os.getpid()) + "\n")
    state_path = queue / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        f"{task}-{model}": {"task": task, "model": model, "status": "pending", "attempt": 0, "gpus": []}
        for model in MODELS for task in TASKS
    }
    api = HfApi(token=os.environ["HF_TOKEN"])
    children = {}
    next_poll = 0.0
    for_poll = {}
    while True:
        for name, job in state.items():
            if job["status"] != "running":
                continue
            receipt = OUTPUT / "profile_receipts" / f"{name}.json"
            code = children[name].poll() if name in children else (
                None if process_running(job["pid"], Path(job["publication"])) else (0 if receipt.exists() else -1)
            )
            if code is not None:
                job.update(status="completed" if code == 0 else "failed",
                           returncode=code, finished=time.time())
                print(json.dumps({"job": name, "status": job["status"]}), flush=True)
        occupied = {gpu for job in state.values() if job["status"] == "running" for gpu in job["gpus"]}
        free = [gpu for gpu in (0, 1) if gpu not in occupied]
        if free and any(job["status"] == "pending" for job in state.values()) and time.time() >= next_poll:
            revision = api.model_info(REPO).sha
            files = set(api.list_repo_files(REPO, revision=revision))
            for_poll = {}
            for name, job in state.items():
                run_id = f"{job['task']}-{job['model']}-hist8-zero-s42-v1"
                path = f"zero-latency/{job['task']}/vla/starvla-{job['model']}-h1/{run_id}"
                if job["status"] == "pending" and path + "/publication_status.json" in files:
                    status = json.loads(Path(hf_hub_download(REPO, path + "/publication_status.json", revision=revision)).read_text())
                    provenance = json.loads(Path(hf_hub_download(REPO, path + "/provenance.json", revision=revision)).read_text())
                    for_poll[name] = {"repo_id": REPO, "revision": revision, "task": job["task"],
                                      "model": job["model"], "run_id": run_id, "path": path,
                                      "checkpoint_sha256": provenance["checkpoint_sha256"], **status}
            next_poll = time.time() + 60
        for name, job in state.items():
            if not free:
                break
            if job["status"] != "pending" or name not in for_poll:
                continue
            gpu = free.pop(0)
            job["attempt"] += 1
            directory = queue / "jobs" / name / f"attempt_{job['attempt']}"
            directory.mkdir(parents=True, exist_ok=True)
            publication = directory / "publication.json"
            write_json_atomic(publication, for_poll[name])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with (directory / "output.log").open("w") as log:
                child = subprocess.Popen([
                    sys.executable, "-u", "-m", "scripts.tasks.atari_hist8.profile_published",
                    "--publication", str(publication),
                ], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            children[name] = child
            job.update(status="running", pid=child.pid, gpus=[gpu], started=time.time(),
                       publication=str(publication), output_log=str(directory / "output.log"))
            print(json.dumps({"job": name, "status": "running", "gpu": gpu}), flush=True)
        write_json_atomic(state_path, state)
        if all(job["status"] == "completed" for job in state.values()):
            write_json_atomic(queue / "complete.json", {"completed": time.time(), "profiles": 6})
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
