"""Coordinate one realtime evaluation with a separately logged GPU memory workload."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.experiments.memory_mock.load import add_irregular_arguments, schedule_from_args

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    from latency_bench.core.config import load_config
    from latency_bench.utils.io import write_yaml

    config = load_config(args.config)
    if config["executor"]["mode"] != "realtime":
        raise ValueError("This measurement requires executor.mode: realtime; simulated delays ignore contention")
    if config["executor"]["inference_devices"] != ["cuda:0"]:
        raise ValueError("Use exactly one logical inference device cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not visible or "," in visible:
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    if min(args.episodes, args.buffer_mib, args.startup_timeout_seconds, args.run_timeout_seconds) <= 0:
        raise ValueError("Episode, buffer and timeout budgets must be positive")
    # Validate and freeze the schedule even for the idle control, allowing exact
    # reuse of its intended workload schedule when comparing conditions.
    schedule = schedule_from_args(args, args.load_seed)
    config["evaluation"].update(episodes=args.episodes, suites=[])
    config["evaluation"].pop("calibration_samples", None)
    config["experiment"]["seed"] = args.episode_seed
    config["executor"]["parallel_envs"] = 1
    evaluation_output = getattr(args, "evaluation_output", None) or args.output / "evaluation"
    config["logging"].update(output_dir=str(evaluation_output),
                             save_action_records=True, save_latency_records=True, save_step_records=True)
    config["logging"]["video"] = {"enabled": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "schedule.json", schedule)
    write_yaml(args.output / "evaluation.yaml", config)
    protocol = {
        "condition": args.condition, "load_seed": args.load_seed,
        "episode_seed": args.episode_seed, "episodes": args.episodes,
        "buffer_mib_each": args.buffer_mib, "cuda_visible_devices": visible,
        "env_fps": config["env"]["env_fps"], "obs_fps": config["env"]["obs_fps"],
        "max_env_steps": config["evaluation"]["max_env_steps"],
        "input_config": str(args.config.resolve()),
        "evaluation_output": str(evaluation_output),
        "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "repository_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "run_timeout_seconds": args.run_timeout_seconds,
        "scope": "One hardware-timed contention measurement; no fitting, training, or uploads",
        "interpretation": "Irregular timings are a declared synthetic workload, not calibrated deployment parameters",
    }
    write_json(args.output / "protocol.json", protocol)
    for relative in ["scripts/experiments/memory_mock/load.py", "scripts/experiments/memory_mock/measure.py",
                     "scripts/experiments/memory_mock/report.py", "latency_bench/eval/driver.py",
                     "latency_bench/executors/realtime/executor.py"]:
        target = args.output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return protocol


def profile_command(config, generated, session, session_id):
    """Run the profiling recipe's ordinary session under the shared load supervisor."""
    from latency_bench.eval.config import _evaluation_seed

    if "eval_entrypoint" in config["profiling"] or config["profiling"]["calibration"] is not None:
        raise ValueError("Memory contention profiling requires the standard evaluator and explicit fixed FPS")
    options = config["profiling"]["memory_mock"]
    command = [sys.executable, "-m", "scripts.experiments.memory_mock.measure",
               "--config", str(generated), "--output", str(session / "contention"),
               "--evaluation-output", str(session), "--episodes", str(config["evaluation"]["episodes"]),
               "--episode-seed", str(_evaluation_seed(config)),
               "--load-seed", str(options["seed_base"] + session_id)]
    for name in ("condition", "buffer_mib", "coverage_seconds", "off_seconds", "short_on_seconds",
                 "long_on_seconds", "long_probability", "startup_timeout_seconds", "run_timeout_seconds"):
        value = options[name]
        command.append("--" + name.replace("_", "-"))
        if isinstance(value, list):
            command.extend(str(v) for v in value)
        else:
            command.append(str(value))
    return command


def worker(output):
    from latency_bench.core.config import load_config
    from latency_bench.eval.driver import run_from_config
    from latency_bench.latency.hardware_info import collect_hardware_fingerprint
    from latency_bench.executors.realtime.timing import monotonic_ms

    write_json(output / "hardware.json", collect_hardware_fingerprint())

    def after_warmup(metadata):
        write_json(output / "warmup.json", metadata)
        (output / "warmup.done").touch()
        # The parent watches both processes, enforces a timeout, and terminates
        # this worker if allocation or workload execution fails.
        while not (output / "load/load.ready").exists():
            time.sleep(.05)
        write_json(output / "measurement_started.json", {"time_ms": monotonic_ms()})

    run_from_config(load_config(output / "evaluation.yaml"), on_warmup_complete=after_warmup)


def supervise(evaluation, load, output, *, startup_timeout, run_timeout):
    deadline = time.monotonic() + startup_timeout
    measuring = False
    while True:
        if load.poll() is not None:
            raise RuntimeError(f"Memory workload exited unexpectedly ({load.returncode}); inspect load_process.log")
        code = evaluation.poll()
        if code is not None:
            if code != 0:
                raise RuntimeError(f"Evaluation exited with code {code}; inspect evaluation_process.log")
            if not (output / "measurement_started.json").exists():
                raise RuntimeError("Evaluation ended without completing the warmup/load handshake")
            return
        if not measuring and (output / "measurement_started.json").exists():
            measuring = True
            deadline = time.monotonic() + run_timeout
        if time.monotonic() >= deadline:
            raise TimeoutError("Measurement timeout" if measuring else "Model warmup / load startup timeout")
        time.sleep(.1)


def stop_process_group(process):
    # Children are started in dedicated sessions, so this targets only this
    # experiment's process tree (including spawned inference/env workers).
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # The leader may have crashed before its descendants exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def measure(args):
    protocol = prepare(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": protocol["cuda_visible_devices"],
           "PYTHONUNBUFFERED": "1", "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    mode = "schedule" if args.condition == "irregular" else args.condition
    load_cmd = [sys.executable, "-m", "scripts.experiments.memory_mock.load", "run",
                "--mode", mode, "--buffer-mib", str(args.buffer_mib), "--output", str(args.output / "load"),
                "--wait-for", str(args.output / "warmup.done")]
    if mode == "schedule":
        load_cmd += ["--schedule", str(args.output / "schedule.json")]
    eval_cmd = [sys.executable, "-m", "scripts.experiments.memory_mock.measure", "--worker", str(args.output)]
    write_json(args.output / "commands.json", {"evaluation": eval_cmd, "load": load_cmd})
    processes = []

    def interrupted(*_):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with ExitStack() as stack:
            try:
                for name, command in (("load", load_cmd), ("evaluation", eval_cmd)):
                    log = stack.enter_context((args.output / f"{name}_process.log").open("x"))
                    processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                       stderr=subprocess.STDOUT, start_new_session=True))
                print(f"Run directory: {args.output}\nLogs: evaluation_process.log and load_process.log", flush=True)
                supervise(processes[1], processes[0], args.output,
                          startup_timeout=args.startup_timeout_seconds, run_timeout=args.run_timeout_seconds)
            finally:
                for process in reversed(processes):
                    stop_process_group(process)
        from scripts.experiments.memory_mock.report import report
        report(args.output)
        (args.output / "DONE").touch()
        print(f"Completed. Read {args.output / 'report.json'}", flush=True)
    except BaseException as exc:
        write_json(args.output / "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="An existing realtime config for logical cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing runs are never overwritten")
    parser.add_argument("--evaluation-output", type=Path, help="Profiling session's existing output directory")
    parser.add_argument("--condition", choices=("idle", "periodic", "irregular"), default="irregular")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-seed", type=int, default=42)
    parser.add_argument("--load-seed", type=int, default=42)
    parser.add_argument("--buffer-mib", type=int, default=512)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--run-timeout-seconds", type=int, default=7200)
    add_irregular_arguments(parser)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.evaluation_output is not None:
        args.evaluation_output = args.evaluation_output.expanduser().resolve()
    measure(args)


if __name__ == "__main__":
    main()
