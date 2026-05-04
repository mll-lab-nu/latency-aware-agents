from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.run import (
    _apply_eval_latency,
    _eval_latency_values,
    _latency_steps_to_ms,
    _train_latency_steps,
    apply_runtime_overrides,
    run_from_config,
)


def _parse_backends(value: str) -> list[str]:
    backends = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"serial", "subproc"}
    invalid = sorted(set(backends) - allowed)
    if invalid:
        raise ValueError(f"--backends must contain only {sorted(allowed)}; got {invalid}")
    if not backends:
        raise ValueError("--backends must contain at least one backend")
    return backends


def _cpu_seconds() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )


def _stage_configs(
    config: dict[str, Any],
    *,
    backend: str,
    output_root: Path,
) -> list[tuple[int | None, float | None, dict[str, Any], dict[str, Any] | None]]:
    eval_latency_values = _eval_latency_values(config)
    if eval_latency_values is None:
        stage_config = deepcopy(config)
        stage_config.setdefault("evaluation", {})["latency_bench_env_backend"] = backend
        stage_config["logging"]["output_dir"] = str(output_root / backend / "current_latency")
        return [(None, None, stage_config, None)]

    train_latency_steps = _train_latency_steps(config)
    stages = []
    for eval_latency_steps in eval_latency_values:
        stage_config = deepcopy(config)
        stage_config.setdefault("evaluation", {})["latency_bench_env_backend"] = backend
        configured_latency_ms = _latency_steps_to_ms(stage_config, eval_latency_steps)
        _apply_eval_latency(stage_config, eval_latency_steps, configured_latency_ms)
        stage_config["logging"]["output_dir"] = str(output_root / backend / f"eval_latency_{eval_latency_steps}")
        metadata = {
            "train_latency_steps": train_latency_steps,
            "eval_latency_steps": int(eval_latency_steps),
            "configured_latency_ms": configured_latency_ms,
        }
        stages.append((int(eval_latency_steps), configured_latency_ms, stage_config, metadata))
    return stages


def _run_stage(
    stage_config: dict[str, Any],
    *,
    extra_metadata: dict[str, Any] | None,
    write_summary: bool,
) -> tuple[int, float, float]:
    cpu_before = _cpu_seconds()
    wall_before = time.perf_counter()
    metrics = run_from_config(
        stage_config,
        extra_metadata=extra_metadata,
        write_summary=write_summary,
    )
    wall_sec = time.perf_counter() - wall_before
    cpu_sec = _cpu_seconds() - cpu_before
    return len(metrics), wall_sec, cpu_sec


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark latency_bench simulated batched env stepping backends. "
            "Worker step and IPC columns are left blank until detailed backend profiling is added."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backends", type=str, default="serial,subproc")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/env_backend_bench"))
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--run-name", type=str)
    args = parser.parse_args(argv)

    base_config = load_config(args.config)
    apply_runtime_overrides(
        base_config,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        output_dir=args.output_root,
        checkpoint_kind=args.checkpoint_kind,
    )

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "backend",
            "eval_latency_steps",
            "configured_latency_ms",
            "episodes",
            "wall_sec",
            "episodes_per_sec",
            "cpu_sec",
            "cpu_util",
            "worker_step_wall_sec",
            "ipc_wait_sec",
        ],
    )
    writer.writeheader()
    for backend in _parse_backends(args.backends):
        for eval_latency_steps, latency_ms, stage_config, metadata in _stage_configs(
            base_config,
            backend=backend,
            output_root=args.output_root,
        ):
            episodes, wall_sec, cpu_sec = _run_stage(
                stage_config,
                extra_metadata=metadata,
                write_summary=eval_latency_steps is None,
            )
            writer.writerow({
                "backend": backend,
                "eval_latency_steps": "" if eval_latency_steps is None else eval_latency_steps,
                "configured_latency_ms": "" if latency_ms is None else f"{latency_ms:.6f}",
                "episodes": episodes,
                "wall_sec": f"{wall_sec:.6f}",
                "episodes_per_sec": f"{episodes / wall_sec:.6f}" if wall_sec > 0 else "",
                "cpu_sec": f"{cpu_sec:.6f}",
                "cpu_util": f"{cpu_sec / wall_sec:.6f}" if wall_sec > 0 else "",
                "worker_step_wall_sec": "",
                "ipc_wait_sec": "",
            })


if __name__ == "__main__":
    main()
