"""Measure CPU HumanoidBench environment and Sample Factory throughput."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.executors.thread_env import set_thread_env_defaults
from latency_bench.utils.proc import cpu_seconds


BASE_TRAIN_CONFIG = REPO_ROOT / (
    "configs/examples/gymnasium/h1hand_balance_hard/small_model_train.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/humanoidbench_throughput/quick"
DEFAULT_TASKS = (
    "h1hand_balance_hard",
    "h1hand_hurdle",
    "h1hand_run",
    "h1hand_basketball",
    "h1strong_highbar_hard",
)
DEFAULT_ENV_COUNTS = (1, 4, 8, 16, 32)
ROLLOUT_STEPS = 64
MIN_SF_ENV_STEPS = 8192
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_cpu_stat() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in (CGROUP_ROOT / "cpu.stat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return values


def _cgroup_snapshot() -> dict[str, Any]:
    cpu_stat = _cgroup_cpu_stat()
    return {
        "cpu_max": (CGROUP_ROOT / "cpu.max").read_text(encoding="utf-8").strip(),
        "cpuset_cpus": (
            CGROUP_ROOT / "cpuset.cpus.effective"
        ).read_text(encoding="utf-8").strip(),
        "cpu_stat": cpu_stat,
        "memory_current_bytes": int(
            (CGROUP_ROOT / "memory.current").read_text(encoding="utf-8")
        ),
        "memory_peak_bytes": int(
            (CGROUP_ROOT / "memory.peak").read_text(encoding="utf-8")
        ),
    }


def _cpu_stat_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in after}


def _raw_worker(
    task_name: str,
    worker_id: int,
    warmup_seconds: float,
    measure_seconds: float,
    render: bool,
    ready_barrier: Any,
    measure_barrier: Any,
    start_time: Any,
    result_queue: Any,
) -> None:
    env = None
    try:
        import gymnasium as gym

        from latency_bench.envs.humanoidbench import HUMANOIDBENCH_ENV_SPECS
        from scripts.scaffold_humanoidbench import build_noop_action

        env_spec = HUMANOIDBENCH_ENV_SPECS[task_name]
        env = gym.make(
            env_spec["env_id"],
            render_mode="rgb_array" if render else None,
            width=256,
            height=256,
        )
        env.reset(seed=worker_id)
        action = build_noop_action(task_name, env)

        warmup_deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < warmup_deadline:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=worker_id)
            if render:
                env.render()

        ready_barrier.wait(timeout=120)
        measure_barrier.wait(timeout=120)
        common_start = start_time.value
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        deadline = common_start + measure_seconds
        steps = 0
        resets = 0
        while time.perf_counter() < deadline:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                env.reset(seed=worker_id + resets + 1)
                resets += 1
            if render:
                env.render()

        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        result_queue.put(
            {
                "worker_id": worker_id,
                "steps": steps,
                "resets": resets,
                "end_time": time.perf_counter(),
                "cpu_seconds": (cpu_end.ru_utime + cpu_end.ru_stime)
                - (cpu_start.ru_utime + cpu_start.ru_stime),
                "max_rss_mib": cpu_end.ru_maxrss / 1024,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if env is not None:
            env.close()


def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)


def run_raw_case(
    *,
    task_name: str,
    worker_count: int,
    warmup_seconds: float,
    measure_seconds: float,
    render: bool,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    ready_barrier = context.Barrier(worker_count + 1)
    measure_barrier = context.Barrier(worker_count + 1)
    start_time = context.Value("d", 0.0)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_raw_worker,
            args=(
                task_name,
                worker_id,
                warmup_seconds,
                measure_seconds,
                render,
                ready_barrier,
                measure_barrier,
                start_time,
                result_queue,
            ),
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()

    try:
        ready_barrier.wait(timeout=180)
        cgroup_before = _cgroup_snapshot()
        start_time.value = time.perf_counter()
        measure_barrier.wait(timeout=180)
    except BaseException:
        _terminate_processes(processes)
        raise

    results = [result_queue.get(timeout=measure_seconds + 180) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    _terminate_processes(processes)

    errors = [result["error"] for result in results if "error" in result]
    if errors:
        raise RuntimeError("HumanoidBench worker failed: " + "; ".join(errors))
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("HumanoidBench worker exited unsuccessfully")

    end_time = max(result["end_time"] for result in results)
    elapsed = end_time - start_time.value
    steps = sum(result["steps"] for result in results)
    cgroup_after = _cgroup_snapshot()
    return {
        "measurement": "raw_rgb" if render else "raw",
        "task_name": task_name,
        "env_count": worker_count,
        "render": render,
        "warmup_seconds": warmup_seconds,
        "measure_seconds": elapsed,
        "steps": steps,
        "resets": sum(result["resets"] for result in results),
        "aggregate_fps": steps / elapsed,
        "per_env_fps": steps / elapsed / worker_count,
        "cpu_seconds": sum(result["cpu_seconds"] for result in results),
        "max_worker_rss_mib": max(result["max_rss_mib"] for result in results),
        "cgroup_cpu_stat_delta": _cpu_stat_delta(
            cgroup_before["cpu_stat"], cgroup_after["cpu_stat"]
        ),
        "cgroup_memory_current_bytes_after": cgroup_after["memory_current_bytes"],
        "cgroup_memory_peak_bytes_after": cgroup_after["memory_peak_bytes"],
    }


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _sf_env_steps(raw_fps: float, env_count: int, measure_seconds: float) -> int:
    estimated_steps = max(MIN_SF_ENV_STEPS, int(raw_fps * measure_seconds))
    return _round_up(estimated_steps, env_count * ROLLOUT_STEPS)


def _latest_checkpoint_steps(checkpoint_dir: Path) -> int:
    pattern = re.compile(r"checkpoint_\d+_(\d+)\.pth$")
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    steps = [int(pattern.fullmatch(path.name).group(1)) for path in checkpoints]
    return max(steps)


def _latest_steady_state_fps_10s(log_path: Path) -> float:
    pattern = re.compile(r"Fps is \(10 sec: ([^,]+),")
    values = [float(value) for value in pattern.findall(log_path.read_text(encoding="utf-8"))]
    return next(value for value in reversed(values) if value == value)


def run_sample_factory_case(
    *,
    task_name: str,
    env_config: dict[str, Any],
    env_count: int,
    train_for_env_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_config(BASE_TRAIN_CONFIG)
    config["env"] = deepcopy(env_config)
    run_name = f"humanoidbench_{task_name}_n{env_count}"
    run_dir = output_dir / "sample_factory" / task_name / f"n{env_count}"
    run_dir.mkdir(parents=True, exist_ok=True)
    num_envs_per_worker = 1
    worker_num_splits = 1
    policy_workers_per_policy = 1
    config_path = run_dir / "config.yaml"
    config["experiment"]["name"] = run_name
    config["backend"]["device"] = "cpu"
    config["backend"]["train_dir"] = str(run_dir / "checkpoints")
    config["training"]["num_workers"] = env_count
    config["training"]["num_envs_per_worker"] = num_envs_per_worker
    config["training"]["worker_num_splits"] = worker_num_splits
    config["training"]["policy_workers_per_policy"] = policy_workers_per_policy
    config["training"]["batch_size"] = env_count * ROLLOUT_STEPS
    config["training"]["train_for_env_steps"] = train_for_env_steps
    config["training"]["async_rl"] = False
    config["training"]["batched_sampling"] = False
    config["logging"]["output_dir"] = str(run_dir / "logs")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    subprocess_env = os.environ.copy()
    subprocess_env.update(
        {
            "MUJOCO_GL": "egl",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    log_path = run_dir / "training.log"
    cpu_before = cpu_seconds()
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-m", "training.train", "--config", str(config_path)],
            cwd=REPO_ROOT,
            env=subprocess_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.perf_counter() - start
    actual_steps = _latest_checkpoint_steps(
        run_dir / "checkpoints" / run_name / "checkpoint_p0"
    )
    steady_state_fps_10s = _latest_steady_state_fps_10s(log_path)
    return {
        "measurement": "sample_factory",
        "task_name": task_name,
        "env_count": env_count,
        "num_workers": env_count,
        "num_envs_per_worker": num_envs_per_worker,
        "worker_num_splits": worker_num_splits,
        "policy_workers_per_policy": policy_workers_per_policy,
        "render": False,
        "requested_env_steps": train_for_env_steps,
        "batch_size": env_count * ROLLOUT_STEPS,
        "steps": actual_steps,
        "wall_seconds": elapsed,
        "aggregate_fps": actual_steps / elapsed,
        "steady_state_fps_10s": steady_state_fps_10s,
        "per_env_fps": actual_steps / elapsed / env_count,
        "cpu_seconds": cpu_seconds() - cpu_before,
        "run_dir": str(run_dir),
    }


def _add_scaling_metrics(rows: list[dict[str, Any]], measurement: str) -> None:
    fps_key = "steady_state_fps_10s" if measurement == "sample_factory" else "aggregate_fps"
    baseline = {
        row["task_name"]: row[fps_key]
        for row in rows
        if row["measurement"] == measurement and row["env_count"] == 1
    }
    for row in rows:
        if row["measurement"] != measurement:
            continue
        base_fps = baseline[row["task_name"]]
        row["speedup_vs_1"] = row[fps_key] / base_fps
        row["scaling_efficiency"] = row[fps_key] / (
            base_fps * row["env_count"]
        )
        row["realtime_env_capacity"] = row[fps_key] / 50.0


def _write_results(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "throughput.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "throughput.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_result_rows(output_dir: Path) -> list[dict[str, Any]]:
    result_path = output_dir / "throughput.jsonl"
    with result_path.open(encoding="utf-8") as result_file:
        return [json.loads(line) for line in result_file if line.strip()]


def _parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = set(DEFAULT_TASKS)
    if not tasks:
        raise ValueError("At least one HumanoidBench task is required")
    if any(task not in allowed for task in tasks):
        raise ValueError(f"Unknown HumanoidBench task in {tasks}")
    return tasks


def _parse_env_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(count < 1 for count in counts):
        raise ValueError("Env counts must contain positive integers")
    return counts


def run_benchmark(
    *,
    output_dir: Path,
    tasks: tuple[str, ...],
    env_counts: tuple[int, ...],
    warmup_seconds: float,
    measure_seconds: float,
    skip_sample_factory: bool,
    skip_raw: bool,
) -> list[dict[str, Any]]:
    if skip_raw:
        rows = [
            row
            for row in _read_result_rows(output_dir)
            if row["task_name"] in tasks
            and (
                row["measurement"] == "raw_rgb"
                or (
                    row["measurement"] == "raw"
                    and row["env_count"] in env_counts
                )
            )
        ]
        if 1 not in env_counts:
            rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {"speedup_vs_1", "scaling_efficiency", "realtime_env_capacity"}
                }
                for row in rows
            ]
        raw_rows = [row for row in rows if row["measurement"] == "raw"]
    else:
        rows = []
        raw_rows = []
        for task_name in tasks:
            for env_count in env_counts:
                row = run_raw_case(
                    task_name=task_name,
                    worker_count=env_count,
                    warmup_seconds=warmup_seconds,
                    measure_seconds=measure_seconds,
                    render=False,
                )
                raw_rows.append(row)
                rows.append(row)
                print(
                    f"raw task={task_name} envs={env_count} "
                    f"fps={row['aggregate_fps']:.1f}",
                    flush=True,
                )

            row = run_raw_case(
                task_name=task_name,
                worker_count=1,
                warmup_seconds=warmup_seconds,
                measure_seconds=measure_seconds,
                render=True,
            )
            rows.append(row)
            print(
                f"rgb task={task_name} fps={row['aggregate_fps']:.1f}",
                flush=True,
            )

        if 1 in env_counts:
            _add_scaling_metrics(rows, "raw")
    if skip_sample_factory:
        return rows

    from scripts.scaffold_humanoidbench import build_env_config, build_probe_report

    task_env_configs = {
        task_name: build_env_config(task_name, build_probe_report(task_name))
        for task_name in tasks
    }
    for raw_row in raw_rows:
        env_count = raw_row["env_count"]
        budget = _sf_env_steps(raw_row["aggregate_fps"], env_count, measure_seconds)
        row = run_sample_factory_case(
            task_name=raw_row["task_name"],
            env_config=task_env_configs[raw_row["task_name"]],
            env_count=env_count,
            train_for_env_steps=budget,
            output_dir=output_dir,
        )
        rows.append(row)
        print(
            f"sf task={row['task_name']} envs={env_count} "
            f"fps={row['aggregate_fps']:.1f}",
            flush=True,
        )
    if 1 in env_counts:
        _add_scaling_metrics(rows, "sample_factory")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--env-counts", default=",".join(str(item) for item in DEFAULT_ENV_COUNTS))
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--skip-sample-factory", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args(argv)

    tasks = _parse_tasks(args.tasks)
    env_counts = _parse_env_counts(args.env_counts)
    set_thread_env_defaults()
    metadata = {
        "tasks": tasks,
        "env_counts": env_counts,
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "sample_factory": not args.skip_sample_factory,
        "hardware": collect_hardware_fingerprint(),
        "cgroup_before": _cgroup_snapshot(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    rows = run_benchmark(
        output_dir=args.output_dir,
        tasks=tasks,
        env_counts=env_counts,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        skip_sample_factory=args.skip_sample_factory,
        skip_raw=args.skip_raw,
    )
    metadata["cgroup_after"] = _cgroup_snapshot()
    _write_results(args.output_dir, rows, metadata)


if __name__ == "__main__":
    mp.freeze_support()
    main()
