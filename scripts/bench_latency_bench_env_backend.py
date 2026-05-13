from __future__ import annotations

import argparse
import contextlib
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
    allowed = {"auto", "serial", "subproc", "flappy_gpu_batched"}
    invalid = sorted(set(backends) - allowed)
    if invalid:
        raise ValueError(f"--backends must contain only {sorted(allowed)}; got {invalid}")
    if not backends:
        raise ValueError("--backends must contain at least one backend")
    return backends


def _parse_parallel_envs(value: str) -> list[int]:
    parallel_envs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parallel_envs:
        raise ValueError("--parallel-envs must contain at least one value")
    non_positive = [value for value in parallel_envs if int(value) <= 0]
    if non_positive:
        raise ValueError(f"--parallel-envs values must be positive; got {non_positive}")
    deduped: list[int] = []
    seen: set[int] = set()
    for item in parallel_envs:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


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
    requested_parallel_envs: Sequence[int],
) -> list[tuple[int | None, float | None, dict[str, Any], dict[str, Any]]]:
    eval_latency_values = _eval_latency_values(config)
    expected_episodes = int(config.get("experiment", {}).get("num_episodes", 1))
    base_metadata = {
        "expected_episodes": expected_episodes,
    }
    if eval_latency_values is None:
        stages = []
        for requested_parallel_envs_value in requested_parallel_envs:
            stage_config = deepcopy(config)
            evaluation_cfg = stage_config.setdefault("evaluation", {})
            evaluation_cfg["eval_parallel_envs"] = int(requested_parallel_envs_value)
            evaluation_cfg["latency_bench_env_backend"] = backend
            stage_config["logging"]["output_dir"] = str(
                output_root / f"parallel_{int(requested_parallel_envs_value)}" / backend / "current_latency"
            )
            metadata = {
                "requested_parallel_envs": int(requested_parallel_envs_value),
                "effective_parallel_envs": min(int(requested_parallel_envs_value), expected_episodes),
                **base_metadata,
            }
            stages.append((None, None, stage_config, metadata))
        return stages

    train_latency_steps = _train_latency_steps(config)
    stages = []
    for requested_parallel_envs_value in requested_parallel_envs:
        for eval_latency_steps in eval_latency_values:
            stage_config = deepcopy(config)
            evaluation_cfg = stage_config.setdefault("evaluation", {})
            evaluation_cfg["eval_parallel_envs"] = int(requested_parallel_envs_value)
            evaluation_cfg["latency_bench_env_backend"] = backend
            configured_latency_ms = _latency_steps_to_ms(stage_config, eval_latency_steps)
            _apply_eval_latency(stage_config, eval_latency_steps, configured_latency_ms)
            stage_config["logging"]["output_dir"] = str(
                output_root
                / f"parallel_{int(requested_parallel_envs_value)}"
                / backend
                / f"eval_latency_{eval_latency_steps}"
            )
            metadata = {
                "requested_parallel_envs": int(requested_parallel_envs_value),
                "effective_parallel_envs": min(int(requested_parallel_envs_value), expected_episodes),
                **base_metadata,
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
) -> tuple[int, int, bool, float, float]:
    expected_episodes = int(stage_config.get("experiment", {}).get("num_episodes", 1))
    cpu_before = _cpu_seconds()
    wall_before = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        metrics = run_from_config(
            stage_config,
            extra_metadata=extra_metadata,
            write_summary=write_summary,
        )
    wall_sec = time.perf_counter() - wall_before
    cpu_sec = _cpu_seconds() - cpu_before
    completed_episodes = len(metrics)
    return completed_episodes, expected_episodes, completed_episodes == expected_episodes, wall_sec, cpu_sec


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark latency_bench simulated batched env stepping backends. "
            "Worker step and IPC columns are left blank until detailed backend profiling is added."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backends", type=str, default="auto")
    parser.add_argument("--parallel-envs", type=str, default="1,4,16,32,64")
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
            "env_name",
            "policy_type",
            "requested_parallel_envs",
            "effective_parallel_envs",
            "backend",
            "eval_latency_steps",
            "configured_latency_ms",
            "episodes",
            "expected_episodes",
            "completed_ok",
            "wall_sec",
            "episodes_per_sec",
            "cpu_sec",
            "cpu_util",
            "output_dir",
            "worker_step_wall_sec",
            "ipc_wait_sec",
        ],
    )
    writer.writeheader()
    requested_parallel_envs = _parse_parallel_envs(args.parallel_envs)
    env_name = str(base_config.get("env", {}).get("name", ""))
    policy_type = str(base_config.get("policy", {}).get("type", ""))
    for backend in _parse_backends(args.backends):
        for eval_latency_steps, latency_ms, stage_config, metadata in _stage_configs(
            base_config,
            backend=backend,
            output_root=args.output_root,
            requested_parallel_envs=requested_parallel_envs,
        ):
            episodes, expected_episodes, completed_ok, wall_sec, cpu_sec = _run_stage(
                stage_config,
                extra_metadata=metadata,
                write_summary=eval_latency_steps is None,
            )
            writer.writerow({
                "env_name": env_name,
                "policy_type": policy_type,
                "requested_parallel_envs": metadata["requested_parallel_envs"],
                "effective_parallel_envs": metadata["effective_parallel_envs"],
                "backend": backend,
                "eval_latency_steps": "" if eval_latency_steps is None else eval_latency_steps,
                "configured_latency_ms": "" if latency_ms is None else f"{latency_ms:.6f}",
                "episodes": episodes,
                "expected_episodes": expected_episodes,
                "completed_ok": int(completed_ok),
                "wall_sec": f"{wall_sec:.6f}",
                "episodes_per_sec": f"{episodes / wall_sec:.6f}" if wall_sec > 0 else "",
                "cpu_sec": f"{cpu_sec:.6f}",
                "cpu_util": f"{cpu_sec / wall_sec:.6f}" if wall_sec > 0 else "",
                "output_dir": stage_config["logging"]["output_dir"],
                "worker_step_wall_sec": "",
                "ipc_wait_sec": "",
            })


if __name__ == "__main__":
    main()
