from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.core.timing import StageProfiler, profiler_scope
from latency_bench.executors.thread_env import set_thread_env_defaults
from latency_bench.latency.env_fps_calibration import select_env_fps_from_latency_samples
from latency_bench.latency.hardware_info import visible_cuda_device_count


DEFAULT_SAMPLES = 120
DEFAULT_SKIP = 20


def calibrate_env_fps_from_saturated_latency(
    *,
    config_path: str | Path,
    output: str | Path,
    target_latency_frames: int,
    worker_utilization_cap: float,
    samples: int = DEFAULT_SAMPLES,
    skip: int = DEFAULT_SKIP,
    paced_period_ms: float | None = None,
) -> dict[str, Any]:
    set_thread_env_defaults()
    config = load_config(config_path)
    saturated = _run_spawned_latency_probe(
        config_path=Path(config_path),
        samples=samples,
        skip=skip,
        period_ms=0.0,
    )
    calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=saturated["latency_samples_ms"],
        target_latency_frames=target_latency_frames,
        base_env_fps=float(config["env"]["env_fps"]),
        base_obs_fps=float(config["env"]["obs_fps"]),
        device_count=visible_cuda_device_count(),
        worker_utilization_cap=worker_utilization_cap,
    )
    payload: dict[str, Any] = {
        **calibration.to_dict(),
        "calibration_source": "saturated_policy_latency",
        "saturated_latency_summary": saturated["total_latency_ms"],
        "policy_stage_summary": {
            "policy_build_example_ms": saturated["policy_build_example_ms"],
            "policy_wrapper_predict_action_ms": saturated["policy_wrapper_predict_action_ms"],
            "policy_decode_ms": saturated["policy_decode_ms"],
        },
        "saturated_probe_samples": int(samples),
        "saturated_skip_samples": int(skip),
    }
    if paced_period_ms is not None:
        paced = _run_spawned_latency_probe(
            config_path=Path(config_path),
            samples=samples,
            skip=skip,
            period_ms=float(paced_period_ms),
        )
        payload["paced_period_ms"] = float(paced_period_ms)
        payload["paced_latency_summary"] = paced["total_latency_ms"]
        payload["cadence_sensitive_ratio"] = (
            paced["total_latency_ms"]["median"] / saturated["total_latency_ms"]["median"]
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _run_spawned_latency_probe(
    *,
    config_path: Path,
    samples: int,
    skip: int,
    period_ms: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_probe_child_main, args=(child, config_path, samples, skip, period_ms))
    process.start()
    child.close()
    result = parent.recv()
    process.join()
    return result


def _probe_child_main(connection: Any, config_path: Path, samples: int, skip: int, period_ms: float) -> None:
    config = load_config(config_path)
    result = measure_policy_latency_in_current_process(
        config=config,
        samples=samples,
        skip=skip,
        period_ms=period_ms,
        sleep_fn=time.sleep,
    )
    connection.send(result)
    connection.close()


def measure_policy_latency_in_current_process(
    *,
    config: dict[str, Any],
    samples: int,
    skip: int,
    period_ms: float,
    sleep_fn,
) -> dict[str, Any]:
    from latency_bench.core.actions import ActionResolver
    from latency_bench.envs.factory import build_env_adapter
    from latency_bench.latency.measurement import RuntimeLatencyMeasurer
    from latency_bench.policy.factory import build_policy

    worker_config = deepcopy(config)
    worker_config["policy"]["device"] = worker_config["latency"]["device"]
    action_resolver = ActionResolver(worker_config["env"])
    env = build_env_adapter(worker_config, action_resolver.noop_action)
    policy = build_policy(worker_config, action_resolver)
    measurer = RuntimeLatencyMeasurer(
        device=worker_config["latency"]["device"],
        sync_cuda=bool(worker_config["latency"]["sync_cuda"]),
    )

    latency_values: list[float] = []
    build_values: list[float] = []
    wrapper_values: list[float] = []
    decode_values: list[float] = []
    env.reset(seed=0)
    try:
        for obs_id in range(samples):
            sample_start = time.perf_counter()
            observation = env.observe()
            # sync_cuda=False: this loop measures policy latency, so a CUDA sync
            # inside the call would contaminate the calibration it produces.
            profiler = StageProfiler(enabled=True, sync_cuda=False)

            def call_policy(observation_arg):
                with profiler_scope(profiler):
                    return policy.predict(observation_arg)

            _output, record = measurer.run_and_measure(call_policy, observation, obs_id=obs_id)
            if obs_id >= skip:
                timing = profiler.timings
                latency_values.append(float(record.latency_ms))
                build_values.append(timing.get("policy_build_example_ms", 0.0))
                wrapper_values.append(timing.get("policy_wrapper_predict_action_ms", 0.0))
                decode_values.append(timing.get("policy_decode_ms", 0.0))
            env.step(action_resolver.noop_action)
            elapsed_ms = (time.perf_counter() - sample_start) * 1000.0
            sleep_ms = float(period_ms) - elapsed_ms
            if sleep_ms > 0.0:
                sleep_fn(sleep_ms / 1000.0)
    finally:
        env.close()

    return {
        "period_ms": float(period_ms),
        "sample_count": int(samples),
        "skip": int(skip),
        "latency_samples_ms": latency_values,
        "total_latency_ms": _summary(latency_values),
        "policy_build_example_ms": _summary(build_values),
        "policy_wrapper_predict_action_ms": _summary(wrapper_values),
        "policy_decode_ms": _summary(decode_values),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate env_fps from saturated policy latency samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-latency-frames", type=int, required=True)
    parser.add_argument("--worker-utilization-cap", type=float, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--paced-period-ms", type=float)
    args = parser.parse_args(argv)
    calibrate_env_fps_from_saturated_latency(
        config_path=args.config,
        output=args.output,
        target_latency_frames=args.target_latency_frames,
        worker_utilization_cap=args.worker_utilization_cap,
        samples=args.samples,
        skip=args.skip,
        paced_period_ms=args.paced_period_ms,
    )


if __name__ == "__main__":
    main()
