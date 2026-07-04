"""Multi-GPU worker planning for cross-device simulated replay stages."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import torch

from latency_bench.cross_device.configs import _eval_command
from latency_bench.cross_device.models import ProtocolTask
from latency_bench.utils.io import load_jsonl_records, write_jsonl


def build_sim_worker_plan(task: ProtocolTask, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if task.stage not in {"sim_same", "sim_cross"}:
        return []
    devices = sim_worker_devices(spec)
    worker_count = min(len(devices), int(task.n_episodes))
    if worker_count <= 1:
        return []
    ranges = episode_ranges(int(task.n_episodes), worker_count)
    plan: list[dict[str, Any]] = []
    for worker_index, ((episode_start, episode_count), device) in enumerate(zip(ranges, devices[:worker_count])):
        output_dir = task.output_dir / "workers" / f"worker_{worker_index}"
        plan.append(
            {
                "worker_index": worker_index,
                "cuda_visible_devices": device,
                "episode_start": episode_start,
                "episode_count": episode_count,
                "output_dir": str(output_dir),
                "command": shlex.join(worker_eval_command(task, output_dir, episode_start, episode_count)),
            }
        )
    return plan


def sim_worker_devices(spec: dict[str, Any]) -> list[str]:
    devices = visible_cuda_devices()
    worker_count = spec["sim_worker_count"]
    if str(worker_count).strip().lower() == "auto":
        return devices
    return devices[: int(worker_count)]


def visible_cuda_devices() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def episode_ranges(n_episodes: int, worker_count: int) -> list[tuple[int, int]]:
    base_count = n_episodes // worker_count
    remainder = n_episodes % worker_count
    ranges = []
    episode_start = 0
    for worker_index in range(worker_count):
        episode_count = base_count + (1 if worker_index < remainder else 0)
        ranges.append((episode_start, episode_count))
        episode_start += episode_count
    return ranges


def worker_eval_command(
    task: ProtocolTask,
    output_dir: Path,
    episode_start: int,
    episode_count: int,
) -> list[str]:
    cmd = _eval_command(task)
    cmd[cmd.index("--output-dir") + 1] = str(output_dir)
    cmd.extend(["--episode-start", str(episode_start), "--episode-count", str(episode_count)])
    return cmd


def add_sim_worker_manifest_fields(
    manifest: dict[str, Any],
    sim_worker_plan: list[dict[str, Any]],
) -> None:
    manifest["sim_worker_count"] = len(sim_worker_plan)
    manifest["sim_worker_devices"] = [item["cuda_visible_devices"] for item in sim_worker_plan]
    manifest["sim_worker_episode_ranges"] = [
        {
            "worker_index": item["worker_index"],
            "episode_start": item["episode_start"],
            "episode_count": item["episode_count"],
        }
        for item in sim_worker_plan
    ]
    manifest["sim_worker_commands"] = [item["command"] for item in sim_worker_plan]


def run_sim_worker_plan(task: ProtocolTask, sim_worker_plan: list[dict[str, Any]]) -> None:
    processes: list[tuple[dict[str, Any], subprocess.Popen]] = []
    for worker in sim_worker_plan:
        command = worker_eval_command(
            task,
            Path(worker["output_dir"]),
            int(worker["episode_start"]),
            int(worker["episode_count"]),
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker["cuda_visible_devices"])
        print(
            "running sim worker "
            f"worker={worker['worker_index']} cuda={worker['cuda_visible_devices']} "
            f"episodes={worker['episode_start']}..{int(worker['episode_start']) + int(worker['episode_count']) - 1} "
            f"output_dir={worker['output_dir']}"
        )
        processes.append((worker, subprocess.Popen(command, env=env)))

    failures: list[tuple[dict[str, Any], int]] = []
    for worker, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((worker, returncode))
    if failures:
        worker, returncode = failures[0]
        raise subprocess.CalledProcessError(returncode, str(worker["command"]))

    merge_sim_worker_episode_metrics(task.output_dir, sim_worker_plan)


def merge_sim_worker_episode_metrics(output_dir: Path, sim_worker_plan: list[dict[str, Any]]) -> None:
    records = []
    for worker in sim_worker_plan:
        metrics_path = Path(worker["output_dir"]) / "episode_metrics.jsonl"
        for record in load_jsonl_records(metrics_path):
            record["metadata"]["sim_worker_index"] = worker["worker_index"]
            record["metadata"]["output_dir"] = str(output_dir)
            records.append(record)
    records.sort(key=lambda item: int(item["episode_id"]))
    write_jsonl(output_dir / "episode_metrics.jsonl", records)
