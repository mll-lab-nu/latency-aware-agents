"""Run the per-host hist8 training, dataset, and evaluation jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import sys
import time

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.eval_vla_hist8 import package_bundle, file_sha256, EVAL_PROTOCOL


ROOT = Path(__file__).resolve().parents[3]


def training_command(job: dict, *, resume: bool, stop_after_steps: int | None) -> list[str]:
    command = [
        job["python"], "-m", "scripts.tasks.decision_history.train_vla_hist8",
        "--task", job["task"], "--model", job["model"],
        "--dataset-dir", job["dataset_dir"], "--base-model-dir", job["backbone"],
        "--output-root", job["train_root"], "--run-id", job["run_id"],
        "--vlm-lr", str(job["vlm_lr"]), "--port", str(job["port"]),
        "--training-condition", job["training_condition"],
    ]
    if resume:
        command.append("--resume")
    if stop_after_steps is not None:
        command.extend(["--stop-after-steps", str(stop_after_steps)])
    return command


def selected_job(job: dict) -> dict:
    """Read the completed PI decision used by existing live queue manifests."""
    selection = json.loads(Path(job["selection_path"]).read_text())
    return job["candidates"][selection["winner"]]


def prepare_bundle(job: dict) -> Path:
    """Resolve the selected checkpoint and package the artifacts consumed by evaluation."""
    if job["kind"] in {"eval_selected", "stage_selected"}:
        source = selected_job(job)
        run_dir = Path(source["train_root"]) / source["run_id"]
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        bundle = Path(job["bundle_root"]) / source["run_id"]
        package_bundle(run_dir, Path(source["dataset_dir"]), Path(source["collection_provenance"]),
                       bundle, step=5000, wandb_run_id=metadata["wandb_run_id"])
    elif job["kind"] in {"eval_checkpoint", "stage_checkpoint", "package_checkpoint"}:
        bundle = Path(job["bundle_dir"])
        package_bundle(Path(job["run_dir"]), Path(job["dataset_dir"]), Path(job["collection_provenance"]),
                       bundle, step=job["step"], wandb_run_id=job["wandb_run_id"])
        if job["source"] == "legacy":
            provenance = json.loads((bundle / "provenance.json").read_text())
            assert provenance["checkpoint_sha256"] == job["checkpoint_sha256"]
    elif job["kind"] == "eval_existing":
        bundle = Path(job["bundle_dir"])
        source = Path(job["source_bundle_dir"])
        sha = file_sha256(source / "checkpoints/model.pt")
        assert sha == job["checkpoint_sha256"]
        (bundle / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json", "manifest.json",
                     "provenance.json", "checkpoint.sha256", "collection_quality_report.json", "rollout_metadata.json"):
            shutil.copyfile(source / name, bundle / name)
        checkpoint = bundle / "checkpoints/model.pt"
        if checkpoint.exists():
            checkpoint.unlink()
        os.link(source / "checkpoints/model.pt", checkpoint)
        provenance = json.loads((bundle / "provenance.json").read_text())
        provenance.update(checkpoint_sha256=sha, eval_protocol_version=EVAL_PROTOCOL, history_encoding="binary_0_1")
        write_json_atomic(bundle / "provenance.json", provenance)
    elif job["kind"] == "eval_ready":
        return Path(job["bundle_dir"])
    else:
        raise KeyError(job["kind"])
    snapshot = json.loads((ROOT / "code_snapshot.json").read_text())
    write_json_atomic(bundle / "eval_code_snapshot.json", snapshot)
    return bundle


def execute(job: dict) -> int:
    if job["kind"] == "atari_teacher":
        return subprocess.run([
            job["python"], "-m", "scripts.tasks.atari_hist8.profile_teacher",
            "--profile-receipt", job["profile_receipt"], "--stage", job["stage"],
            "--output-root", job["output_root"],
        ], cwd=ROOT, check=False).returncode
    if job["kind"] == "collect_profile":
        return subprocess.run([
            job["python"], "-m", "scripts.tasks.decision_history.collect_vla_hist8",
            "--task", job["task"], "--model", job["model"], "--teacher-catalog", job["teacher_catalog"],
            "--output-root", job["output_root"], "--profile-root", str(ROOT),
        ], cwd=ROOT, check=False).returncode
    if job["kind"] == "publish_vla":
        command = [
            job["python"], "-m", "scripts.tasks.atari_hist8.publish_zero",
            "--task", job["task"], "--model", job["model"], "--stage", job["stage"],
            "--output-root", job["output_root"],
        ]
        if "training_condition" in job:
            command.extend(["--training-condition", job["training_condition"],
                            "--eval-condition", job["env_condition"]])
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    if job["kind"] == "collect_zero":
        command = [job["python"], "-m", "scripts.tasks.decision_history.collect_zero_vla_hist8",
                   "--task", job["task"], "--teacher-catalog", job["teacher_catalog"],
                   "--output-root", job["output_root"], "--stage", job["stage"]]
        if job["mirror_host"] is not None:
            command.extend(["--mirror-host", job["mirror_host"]])
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    if job["kind"] in {"train", "resume_selected", "train_selected"}:
        selected = job if job["kind"] == "train" else selected_job(job)
        command = training_command(
            selected, resume=job["kind"] == "resume_selected" or selected["resume"],
            stop_after_steps=None if job["kind"] == "resume_selected" else selected["stop_after_steps"],
        )
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    bundle = prepare_bundle(job)
    if job["kind"] == "package_checkpoint":
        provenance = json.loads((bundle / "provenance.json").read_text())
        write_json_atomic(bundle / "bundle.complete.json", {
            "checkpoint_sha256": provenance["checkpoint_sha256"], "training_step": provenance["training_step"],
        })
        return 0
    if job["kind"] in {"stage_checkpoint", "stage_selected"}:
        target = f"ubuntu@{job['target_host']}"
        remote = Path(job["target_bundle_root"]) / bundle.name
        ssh = ["ssh", "-i", job["transfer_key"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes"]
        subprocess.run([*ssh, target, f"mkdir -p {shlex.quote(str(remote))}"], check=True)
        subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(ssh),
                        str(bundle) + "/", f"{target}:{remote}/"], check=True)
        remote_sha = subprocess.check_output(
            [*ssh, target,
             f"sha256sum {shlex.quote(str(remote / 'checkpoints/model.pt'))}"], text=True,
        ).split()[0]
        provenance = json.loads((bundle / "provenance.json").read_text())
        assert remote_sha == provenance["checkpoint_sha256"]
        marker = bundle / "bundle.complete.json"
        write_json_atomic(marker, {"checkpoint_sha256": remote_sha, "training_step": provenance["training_step"]})
        subprocess.run(["rsync", "-a", "-e", shlex.join(ssh), str(marker), f"{target}:{remote}/"], check=True)
        return 0
    command = [
        job["python"], "-m", "scripts.tasks.decision_history.eval_vla_hist8", "eval",
        "--task", job["task"], "--model", job["model"], "--bundle-dir", str(bundle),
        "--profile-path", job["profile_path"], "--backbone-path", job["backbone"],
        "--vla-python", job["python"], "--output-dir", job["eval_dir"],
        "--device", "cuda:0", "--episodes", "100",
        "--env-condition", job["env_condition"], "--parallel-envs", str(job["parallel_envs"]),
        "--metric-prefix", job["metric_prefix"],
        "--inference-devices", *[f"cuda:{index}" for index in range(job["gpus_required"])],
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def worker(specification: Path, result_path: Path) -> None:
    job = json.loads(specification.read_text())
    environment_root = Path(sys.executable).parents[1]
    os.environ["CUDA_HOME"] = str(environment_root)
    os.environ["CXX"] = str(environment_root / "bin/x86_64-conda-linux-gnu-g++")
    os.environ["NVCC_PREPEND_FLAGS"] = f"-ccbin={os.environ['CXX']}"
    record = {"id": job["id"], "status": "running", "pid": os.getpid(), "started": time.time(),
              "gpus": os.environ["CUDA_VISIBLE_DEVICES"], "specification": str(specification),
              "hostname": socket.gethostname()}
    if job["gpus_required"]:
        record["hardware"] = subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=index,name,uuid", "--format=csv,noheader",
        ], text=True).splitlines()
    write_json_atomic(result_path, record)
    code = execute(job)
    record.update(status="completed" if code == 0 else "needs_diagnosis" if code == 2 else "failed",
                  returncode=code, finished=time.time())
    write_json_atomic(result_path, record)
    raise SystemExit(code)


def available_gpus(reserved: set[int]) -> list[int]:
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True,
    ).splitlines()
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True,
    ).splitlines()
    busy = set(line.strip() for line in applications)
    return [int(index.strip()) for index, uuid, memory in (line.split(",") for line in devices)
            if int(index.strip()) not in reserved and uuid.strip() not in busy and int(memory.strip()) < 1024]


def process_running(pid: int, specification: Path) -> bool:
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return str(specification).encode() in arguments


def run_queue(manifest_path: Path, retry: list[str]) -> None:
    manifest = json.loads(manifest_path.read_text())
    queue = manifest_path.parent
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (queue / "controller.pid").write_text(f"{os.getpid()}\n")
    jobs = {job["id"]: job for job in manifest["jobs"]}
    state_path = queue / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {job_id: {"status": "pending", "attempt": 0, "gpus": []} for job_id in jobs}
    for job_id in retry:
        assert state[job_id]["status"] in {"failed", "needs_diagnosis"}
        state[job_id]["status"] = "pending"
    children = {}
    while True:
        for child in children.values():
            child.poll()
        for job_id, status in state.items():
            if status["status"] != "running":
                continue
            directory = queue / "jobs" / job_id / f"attempt_{status['attempt']}"
            result_path = directory / "result.json"
            if job_id in children:
                code = children[job_id].poll()
            else:
                code = None if process_running(status["pid"], directory / "job.json") else -1
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["status"] != "running":
                    status.update(status=result["status"], returncode=result["returncode"], finished=result["finished"])
                    print(json.dumps({"event": result["status"], "job": job_id, "returncode": result["returncode"]}), flush=True)
                    continue
            if code is not None:
                status.update(status="failed", returncode=code, finished=time.time())
        reserved = {gpu for status in state.values() if status["status"] == "running" for gpu in status["gpus"]}
        free = available_gpus(reserved)
        for job_id, job in sorted(jobs.items(), key=lambda item: item[1]["priority"]):
            status = state[job_id]
            if status["status"] != "pending":
                continue
            if not all(state[parent]["status"] == "completed" for parent in job["depends_on"]):
                continue
            if "ready_path" in job and not Path(job["ready_path"]).exists():
                continue
            eligible = [gpu for gpu in free if gpu in job["gpu_pool"]] if "gpu_pool" in job else free
            if len(eligible) < job["gpus_required"]:
                if "aux_gpu" in job and job["aux_gpu"] in eligible:
                    selected = [job["aux_gpu"]]
                else:
                    continue
            else:
                selected = eligible[:job["gpus_required"]]
            free = [gpu for gpu in free if gpu not in selected]
            status["attempt"] += 1
            directory = queue / "jobs" / job_id / f"attempt_{status['attempt']}"
            directory.mkdir(parents=True, exist_ok=True)
            specification = directory / "job.json"
            write_json_atomic(specification, job | {"gpus_required": len(selected)})
            env = os.environ.copy()
            runtime_root = Path(job["runtime_root"])
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
            env["PYTHONPATH"] = os.pathsep.join(str(runtime_root / path) for path in (
                ".", "third_party/starVLA", "third_party/sample-factory", "third_party/flappy-bird-gymnasium", "third_party/cule"))
            env["NO_ALBUMENTATIONS_UPDATE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            with (directory / "output.log").open("w") as output:
                child = subprocess.Popen(
                    [job["python"], str(runtime_root / "scripts/tasks/decision_history/run_hist8_v2.py"), "worker", "--job", str(specification),
                     "--result", str(directory / "result.json")],
                    cwd=runtime_root, env=env, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            children[job_id] = child
            status.update(status="running", pid=child.pid, gpus=selected, started=time.time())
            print(json.dumps({"event": "started", "job": job_id, "pid": child.pid, "gpus": selected}), flush=True)
        write_json_atomic(state_path, state)
        if all(status["status"] == "completed" for status in state.values()):
            write_json_atomic(queue / "completed.json", {"completed": time.time(), "jobs": len(state)})
            return
        running = any(status["status"] == "running" for status in state.values())
        runnable = any(status["status"] == "pending" and all(
            state[parent]["status"] == "completed" for parent in jobs[job_id]["depends_on"])
            for job_id, status in state.items())
        if not running and not runnable:
            write_json_atomic(queue / "attention.json", {"jobs": state})
            raise SystemExit(2)
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--retry", action="append", default=[])
    task = subparsers.add_parser("worker")
    task.add_argument("--job", type=Path, required=True)
    task.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.job, args.result)
    else:
        run_queue(args.manifest, args.retry)


if __name__ == "__main__":
    main()
