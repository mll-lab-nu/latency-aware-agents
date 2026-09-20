"""Run the zero-latency and conditioned-profile QwenPI H16 DAgger recipes."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from scripts.starvla.train_task import environment


ROOT = Path(__file__).resolve().parents[3]
JOBS = (("zero_full", "2,3"), ("profile_conditioned", "4,5"))


def reserved_gpus(gpu_xml, environments):
    """Include live controllers with no CUDA context during phase transitions."""
    gpus = ET.fromstring(gpu_xml).findall("gpu")
    busy = {str(index) for index, gpu in enumerate(gpus) if gpu.findall("processes/process_info")}
    ids = {gpu.findtext("uuid"): str(index) for index, gpu in enumerate(gpus)}
    ids.update({str(index): str(index) for index in range(len(gpus))})
    for environment in environments:
        for field in environment.split(b"\0"):
            if field.startswith(b"CUDA_VISIBLE_DEVICES="):
                busy.update(ids[value] for value in field.split(b"=", 1)[1].decode().split(",") if value in ids)
    return busy


def busy_gpus():
    environments = []
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            environments.append(path.read_bytes())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # /proc races with exiting processes; inaccessible GPU contexts remain in NVML.
            continue
    return reserved_gpus(subprocess.check_output(["nvidia-smi", "-q", "-x"]), environments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vla-python", required=True)
    parser.add_argument("--qwen-backbone", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = {mode: {"status": "queued", "pair": pair, "model": "qwenpi_v3", "horizon": 16}
               for mode, pair in JOBS}
    def launch(mode, pair, stage):
        output = args.output / f"qwenpi_v3_h16_{mode}_{stage}"
        command = [sys.executable, "-u", "-m",
                   "scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger",
                   "--model", "qwenpi_v3", "--horizon", "16", "--label-mode", mode,
                   "--output", str(output), "--vla-python", args.vla_python,
                   "--backbone", args.qwen_backbone,
                   "--port", str(29772 + [pair for _, pair in JOBS].index(pair)),
                   "--rollout-workers", "16", "--inference-batch-size", "8",
                   "--rollout-overlap", "--no-gradient-checkpointing"]
        if stage == "smoke":
            command.append("--smoke")
        env = environment() | {"CUDA_VISIBLE_DEVICES": pair,
                               "MUJOCO_EGL_DEVICE_ID": pair.split(",")[0],
                               "WANDB_MODE": "disabled" if stage == "smoke" else "online"}
        log_name = f"{mode}_{stage}.log"
        with (args.output / log_name).open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        records[mode].update(status=stage, pid=process.pid, output=str(output))
        print(json.dumps({"label_mode": mode, **records[mode]}), flush=True)
        return process

    def save_status():
        status = {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "busy_gpus": sorted(busy_gpus()), "jobs": records}
        temporary = args.output / "queue.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output / "queue.json")

    def wait_all(processes, stage):
        while processes:
            for mode, process in list(processes.items()):
                code = process.poll()
                if code is not None:
                    records[mode].update(status=f"{stage}_passed" if code == 0 else "failed",
                                         exit_code=code)
                    del processes[mode]
            save_status()
            if processes:
                time.sleep(30)

    while any(not busy_gpus().isdisjoint(pair.split(",")) for _, pair in JOBS):
        save_status()
        time.sleep(30)

    smokes = {mode: launch(mode, pair, "smoke") for mode, pair in JOBS}
    wait_all(smokes, "smoke")

    active = {}
    if records["zero_full"]["status"] == "smoke_passed":
        active["zero_full"] = launch("zero_full", "2,3", "formal")
    if records["profile_conditioned"]["status"] == "smoke_passed":
        active["profile_conditioned"] = launch("profile_conditioned", "4,5", "formal")
    wait_all(active, "formal")
    for record in records.values():
        if record["status"] == "formal_passed":
            record["status"] = "completed"
    save_status()
    return int(any(record["status"] == "failed" for record in records.values()))


if __name__ == "__main__":
    raise SystemExit(main())
