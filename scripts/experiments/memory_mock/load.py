"""Hold CUDA memory and optionally alternate idle/device-copy phases."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import signal
import threading

from latency_bench.executors.realtime.timing import monotonic_ms


HALF_PERIOD_MS = 5000.0


def periodic_position(now_ms, origin_ms):
    index = math.floor((now_ms - origin_ms) / HALF_PERIOD_MS)
    return index, index % 2 == 1, origin_ms + (index + 1) * HALF_PERIOD_MS


def schedule_position(now_ms, deadlines_ms):
    index = bisect_right(deadlines_ms, now_ms)
    if index == len(deadlines_ms):
        raise RuntimeError("Frozen memory-mock schedule exhausted; stop the load before its coverage ends.")
    return index, index % 2 == 1, deadlines_ms[index]


def frozen_schedule(seed):
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(seed))
    phases, end = [], 0.0
    while end < 7200:
        duration = float(rng.uniform(0.0, 10.0))
        phases.append({"phase_index": len(phases), "phase": "OFF" if len(phases) % 2 == 0 else "ON",
                       "duration_s": duration, "planned_start_offset_s": end,
                       "planned_end_offset_s": end + duration})
        end += duration
    return {"load_seed": seed, "numpy_version": np.__version__, "bit_generator": "PCG64",
            "distribution": "independent continuous Uniform(0.0, 10.0) seconds per phase",
            "coverage_seconds": end, "phases": phases}


def read_schedule(path):
    schedule = json.loads(path.read_text())
    if not schedule["phases"]:
        raise ValueError("A frozen schedule must contain at least one phase.")
    end = 0.0
    for index, row in enumerate(schedule["phases"]):
        duration = row["duration_s"]
        if (row["phase_index"] != index or row["phase"] != ("ON" if index % 2 else "OFF")
                or not math.isfinite(duration) or duration < 0
                or row["planned_start_offset_s"] != end or row["planned_end_offset_s"] != end + duration):
            raise ValueError(f"Invalid frozen schedule phase {index}: expected contiguous alternating OFF/ON phases.")
        end = row["planned_end_offset_s"]
    return schedule


def run(output: Path, mode: str, buffer_mib: int, schedule_path: Path | None, wait_for: Path | None):
    schedule = read_schedule(schedule_path) if mode == "schedule" else None
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "load.jsonl").open("x", buffering=1) as log:
        def record(event, **fields):
            log.write(json.dumps({"event": event, "time_ms": monotonic_ms(), **fields}) + "\n")

        # Waiting must not create a CUDA context or interfere with model warmup.
        while wait_for is not None and not wait_for.exists():
            if stopped.wait(0.05):
                return
        import torch

        buffer_bytes = buffer_mib * 1024**2
        first = torch.empty(buffer_bytes, dtype=torch.uint8, device="cuda:0")
        second = torch.empty_like(first)
        first.fill_(17)
        second.fill_(29)
        torch.cuda.synchronize()
        origin_ms = monotonic_ms()
        deadlines_ms = [origin_ms + row["planned_end_offset_s"] * 1000
                        for row in schedule["phases"]] if schedule is not None else None
        record("start", buffer_bytes_each=buffer_bytes, buffers=2, mode=mode, origin_ms=origin_ms,
               schedule_path=None if schedule_path is None else str(schedule_path),
               load_seed=None if schedule is None else schedule["load_seed"],
               numpy_version=None if schedule is None else schedule["numpy_version"],
               gpu_name=torch.cuda.get_device_name(0), logical_device="cuda:0",
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        previous_index = -1
        phase_start_ms = phase_deadline_ms = None
        batches = 0
        first_submit_ms = last_completed_ms = None
        report_ms = monotonic_ms()

        def report_work():
            nonlocal batches, first_submit_ms, report_ms
            if batches:
                record("work", batches=batches, copy_operations=2 * batches,
                       phase_index=previous_index, copied_bytes=2 * batches * buffer_bytes,
                       first_submit_time_ms=first_submit_ms, last_completed_time_ms=last_completed_ms)
                batches = 0
                first_submit_ms = None
                report_ms = monotonic_ms()

        try:
            if mode == "idle":
                record("switch", on=False, phase_index=0)
                (output / "load.ready").touch()
                stopped.wait()
            while not stopped.is_set():
                now_ms = monotonic_ms()
                if schedule is None:
                    phase_index, active, deadline_ms = periodic_position(now_ms, origin_ms)
                else:
                    phase_index, active, deadline_ms = schedule_position(now_ms, deadlines_ms)
                if phase_index != previous_index:
                    report_work()
                    if previous_index >= 0:
                        record("phase_end", phase_index=previous_index, actual_start_time_ms=phase_start_ms,
                               actual_end_time_ms=now_ms, reason="boundary", truncated=False)
                    if schedule is not None:
                        for skipped in range(previous_index + 1, phase_index):
                            record("skipped_phase", **schedule["phases"][skipped], reason="already_expired",
                                   actual_duration_s=0, copy_operations=0)
                    planned_start_ms = origin_ms + (schedule["phases"][phase_index]["planned_start_offset_s"] * 1000
                                                    if schedule is not None else phase_index * HALF_PERIOD_MS)
                    record("switch", time_ms=now_ms, on=active, phase_index=phase_index,
                           next_boundary_ms=deadline_ms, planned_start_time_ms=planned_start_ms,
                           planned_end_time_ms=deadline_ms)
                    previous_index = phase_index
                    phase_start_ms, phase_deadline_ms = now_ms, deadline_ms
                    (output / "load.ready").touch()
                if not active:
                    stopped.wait(max(0.0, (deadline_ms - monotonic_ms()) / 1000.0))
                    continue
                batch_start_ms = monotonic_ms()
                if batch_start_ms >= deadline_ms:
                    continue
                if first_submit_ms is None:
                    first_submit_ms = batch_start_ms
                first.copy_(second, non_blocking=True)
                second.copy_(first, non_blocking=True)
                # One synchronized pair bounds work still in flight at an OFF boundary.
                torch.cuda.synchronize()
                last_completed_ms = monotonic_ms()
                batches += 1
                if last_completed_ms > deadline_ms:
                    record("copy_overrun", phase_index=phase_index, copy_operations=2,
                           first_submit_time_ms=batch_start_ms, last_completed_time_ms=last_completed_ms,
                           planned_boundary_ms=deadline_ms, in_flight_at_boundary=True)
                if last_completed_ms - report_ms >= 1000.0:
                    report_work()
        finally:
            report_work()
            if phase_start_ms is not None:
                now_ms = monotonic_ms()
                record("phase_end", phase_index=previous_index, actual_start_time_ms=phase_start_ms,
                       actual_end_time_ms=now_ms, reason="stop", truncated=now_ms < phase_deadline_ms)
            record("stop", last_completed_time_ms=last_completed_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("schedule", help="Freeze >=2 hours of alternating Uniform(0,10)s phases")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--output", type=Path, required=True)
    load = commands.add_parser("run", help="Hold two CUDA buffers until SIGINT or SIGTERM")
    load.add_argument("--output", type=Path, required=True, help="Fresh directory for load.jsonl and load.ready")
    load.add_argument("--mode", choices=("idle", "periodic", "schedule"), required=True)
    load.add_argument("--buffer-mib", type=int, default=512, help="MiB per buffer; identical across compared runs")
    load.add_argument("--schedule", type=Path)
    load.add_argument("--wait-for", type=Path, help="Wait for this file before importing Torch or allocating CUDA memory")
    args = parser.parse_args()
    if args.command == "schedule":
        schedule = frozen_schedule(args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(schedule, handle, indent=2)
            handle.write("\n")
        return
    if args.buffer_mib <= 0:
        parser.error("--buffer-mib must be positive")
    if (args.mode == "schedule") != (args.schedule is not None):
        parser.error("--schedule is required only for --mode schedule")
    run(args.output, args.mode, args.buffer_mib, args.schedule, args.wait_for)


if __name__ == "__main__":
    main()
