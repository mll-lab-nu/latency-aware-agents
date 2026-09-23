"""Start queued naive H8 DAgger arms when an H200 training pair is free."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RELATIVE_ROOT = Path("outputs/starvla/runs/mikasa_intercept_grab_fast")
RUN_NAME = "h8-naive-dagger-20260923"
REMOTE_ROOT = Path("/home/ubuntu/codex/mikasa-h8")
REMOTE_SCRIPT = REMOTE_ROOT / "scripts/tasks/mikasa_intercept_grab_fast/dispatch_h8_naive_dagger_queue.py"
HOSTS = ("34.131.223.74", "34.131.91.104")
MODELS = ("qwenoft", "qwenpi_v3", "qwengr00t")
OLD_ARMS = {
    HOSTS[0]: (("profile-latency", "qwenoft", (0, 1), 6),
               ("fixed-2", "qwenoft", (2, 3), 7),
               ("fixed-2", "qwengr00t", (4, 5), 6)),
    HOSTS[1]: (("profile-latency", "qwenpi_v3", (0, 1), 6),
               ("profile-latency", "qwengr00t", (2, 3), 7),
               ("fixed-2", "qwenpi_v3", (4, 5), 6)),
}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _alive(output: Path) -> bool:
    marker = output / "controller.pid"
    if not marker.is_file():
        return False
    pid = int(marker.read_text())
    command = Path(f"/proc/{pid}/cmdline")
    return command.is_file() and str(output).encode() in command.read_bytes()


def _ready(model: str) -> bool:
    base = REMOTE_ROOT / RELATIVE_ROOT
    source = base / "h8-zero-latency" / model
    dataset = REMOTE_ROOT / "outputs/starvla/datasets/mikasa_intercept_grab_fast/teacher-rollouts-h8-state-v1"
    required = (
        source / "run.json", source / "bootstrap.yaml",
        source / "training/bootstrap/config.full.yaml",
        source / "training/bootstrap/dataset_statistics.json",
        source / "training/bootstrap/checkpoints/steps_5000_pytorch_model.pt",
        source / "evals/final-zero/eval_config.yaml",
        source / "evals/final-profile-no-startup-wait/eval_config.yaml",
        source / "evals/final-fixed-2/eval_config.yaml",
        dataset / "lerobot/task_contract.json",
        dataset / "lerobot__val/task_contract.json",
    )
    return all(path.is_file() for path in required)


def _snapshot(host: str) -> dict:
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    devices = {int(index.strip()): (uuid.strip(), int(used.strip()))
               for index, uuid, used in (line.split(",") for line in lines)}
    memory = {index: used for index, (_, used) in devices.items()}
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    applications = {}
    for line in processes:
        uuid, pid = line.split(",")
        applications.setdefault(uuid.strip(), []).append(int(pid.strip()))
    active_uuids = set(applications)
    busy_gpus = [index for index, (uuid, _) in devices.items() if uuid in active_uuids]
    old_root = REMOTE_ROOT / RELATIVE_ROOT / "h8-conditioned-dagger-20260923"
    claimed = set()
    simulator_claims = set()
    for condition, model, pair, simulator in OLD_ARMS[host]:
        output = old_root / condition / model
        if not (output / "result.json").is_file() or _alive(output):
            claimed.update(pair)
            simulator_claims.add(simulator)
    shareable_simulators = []
    for gpu in simulator_claims:
        uuid = devices[gpu][0]
        if memory[gpu] >= 8192:
            continue
        commands = [Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    for pid in applications.get(uuid, [])]
        if all(b"dagger_rollout" in b" ".join(command)
               and b"h8-conditioned-dagger-20260923" in b" ".join(command)
               and b"--simulator-gpu" in command
               and command[command.index(b"--simulator-gpu") + 1] == str(gpu).encode()
               for command in commands):
            shareable_simulators.append(gpu)
    naive_root = REMOTE_ROOT / RELATIVE_ROOT / RUN_NAME
    jobs = {}
    for model in MODELS:
        output = naive_root / model
        record = output / "run.json"
        alive = _alive(output)
        jobs[model] = {
            "alive": alive,
            "complete": (output / "result.json").is_file() and not alive,
            "run": json.loads(record.read_text()) if record.is_file() else None,
        }
    return {
        "host": host, "memory": memory, "busy_gpus": busy_gpus,
        "old_training_claims": sorted(claimed),
        "old_simulator_claims": sorted(simulator_claims),
        "shareable_simulators": sorted(shareable_simulators),
        "ready_models": [model for model in MODELS if _ready(model)],
        "jobs": jobs,
    }


def _remote(host: str, *args: str) -> dict:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               f"ubuntu@{host}", "python3", str(REMOTE_SCRIPT), *args]
    result = json.loads(subprocess.check_output(command, text=True))
    if args[0] == "snapshot":
        result["memory"] = {int(gpu): used for gpu, used in result["memory"].items()}
    return result


def _launch(model: str, pair: tuple[int, int], simulator: int, port: int) -> dict:
    output = REMOTE_ROOT / RELATIVE_ROOT / RUN_NAME / model
    output.mkdir(parents=True, exist_ok=True)
    if _alive(output):
        return {"pid": int((output / "controller.pid").read_text()), "already_running": True}
    if (output / "controller.pid").exists():
        raise RuntimeError(f"{model} controller stopped before result: {output}")
    launcher = REMOTE_ROOT / "scripts/tasks/mikasa_intercept_grab_fast/launch_h8_naive_dagger_arm.sh"
    uv = ("/home/ubuntu/.local/bin/uv" if os.uname().nodename == "h200-8-asia-test-1"
          else "/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/uv")
    command = ["bash", str(launcher), model, f"{pair[0]},{pair[1]}",
               str(simulator), str(port), str(output)]
    with (output / "controller.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REMOTE_ROOT, env=os.environ | {"UV": uv},
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (output / "controller.pid").write_text(f"{process.pid}\n")
    return {"pid": process.pid, "already_running": False}


def _slot(snapshot: dict, records: dict, host: str) -> tuple[tuple[int, int], int] | None:
    claimed = set(snapshot["old_training_claims"])
    old_simulators = set(snapshot["old_simulator_claims"])
    shareable_simulators = set(snapshot["shareable_simulators"])
    new_simulators = set()
    for record in records.values():
        if record["host"] == host and record["status"] != "complete":
            claimed.update(int(gpu) for gpu in record["train_devices"].split(","))
            new_simulators.add(record["simulator_gpu"])
    free = [gpu for gpu in sorted(snapshot["memory"])
            if (snapshot["memory"][gpu] < 1024 or gpu in shareable_simulators)
            and (gpu not in snapshot["busy_gpus"] or gpu in shareable_simulators)
            and gpu not in claimed and gpu not in new_simulators
            and (gpu not in old_simulators or gpu in shareable_simulators)]
    if len(free) < 2:
        return None
    pair = tuple(free[:2])
    if 6 in pair:
        return pair, 6
    if 7 in pair:
        return pair, 7
    simulator = next((gpu for gpu in (6, 7) if gpu not in pair
                      and gpu not in claimed and gpu not in new_simulators
                      and (gpu in old_simulators or
                           (snapshot["memory"][gpu] < 1024
                            and gpu not in snapshot["busy_gpus"]))), None)
    if simulator is None:
        simulator = next((gpu for gpu in free[2:] if gpu not in pair), None)
    return (pair, simulator) if simulator is not None else None


def _dispatch() -> dict:
    queue = ROOT / RELATIVE_ROOT / RUN_NAME / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = queue / "manifest.json"
        if not manifest.is_file():
            _write_json(manifest, {"run": RUN_NAME, "models": list(MODELS),
                                   "source": "h8-zero-latency step-5000",
                                   "dagger_steps": 5000})
        state_file = queue / "jobs.json"
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        if not state_file.is_file():
            _write_json(state_file, state)
        snapshots = {host: _remote(host, "snapshot", "--host", host) for host in HOSTS}
        for model in MODELS:
            for host in HOSTS:
                remote_job = snapshots[host]["jobs"][model]
                if remote_job["run"] is not None and model not in state:
                    run = remote_job["run"]
                    state[model] = {"host": host, "train_devices": run["train_devices"],
                                    "simulator_gpu": run["simulator_gpu"],
                                    "port": run["port"], "status": "running"}
            if model in state:
                host = state[model]["host"]
                job = snapshots[host]["jobs"][model]
                if state[model]["status"] == "launching" and not job["alive"] and not job["complete"]:
                    pair = tuple(int(gpu) for gpu in state[model]["train_devices"].split(","))
                    result = _remote(host, "launch", "--model", model,
                                     "--pair", f"{pair[0]},{pair[1]}",
                                     "--simulator", str(state[model]["simulator_gpu"]),
                                     "--port", str(state[model]["port"]))
                    state[model]["pid"] = result["pid"]
                    job["alive"] = True
                state[model]["status"] = ("complete" if job["complete"] else
                                           "running" if job["alive"] else "needs_attention")
        _write_json(state_file, state)
        started = []
        for model in MODELS:
            if model in state:
                continue
            for host in HOSTS:
                snapshot = snapshots[host]
                if model not in snapshot["ready_models"]:
                    continue
                slot = _slot(snapshot, state, host)
                if slot is None:
                    continue
                pair, simulator = slot
                port = 29920 + MODELS.index(model) * 2
                state[model] = {"host": host, "train_devices": f"{pair[0]},{pair[1]}",
                                "simulator_gpu": simulator, "port": port,
                                "status": "launching"}
                _write_json(state_file, state)
                result = _remote(host, "launch", "--model", model,
                                 "--pair", f"{pair[0]},{pair[1]}",
                                 "--simulator", str(simulator), "--port", str(port))
                state[model].update({"status": "running", "pid": result["pid"]})
                _write_json(state_file, state)
                snapshot["memory"][pair[0]] = 999999
                snapshot["memory"][pair[1]] = 999999
                started.append(model)
                break
        return {"jobs": state, "started": started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dispatch", "snapshot", "launch"))
    parser.add_argument("--host", choices=HOSTS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--pair")
    parser.add_argument("--simulator", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.mode == "dispatch":
        result = _dispatch()
    elif args.mode == "snapshot":
        result = _snapshot(args.host)
    else:
        left, right = (int(gpu) for gpu in args.pair.split(","))
        result = _launch(args.model, (left, right), args.simulator, args.port)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
