"""Measure saturated policy latency to select the deployment environment clock."""

from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.executors.thread_env import set_thread_env_defaults
from latency_bench.latency.env_fps_calibration import select_env_fps_from_latency_samples
from latency_bench.utils.stats import percentile_linear


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
        devices=config["executor"]["inference_devices"],
    )
    calibration = select_env_fps_from_latency_samples(
        latency_samples_by_slot_ms=saturated["latency_samples_by_slot_ms"],
        target_latency_frames=target_latency_frames,
        base_env_fps=config["env"]["env_fps"],
        base_obs_fps=config["env"]["obs_fps"],
        worker_utilization_cap=worker_utilization_cap,
    )
    payload: dict[str, Any] = {
        **calibration.to_dict(),
        "calibration_source": "saturated_policy_latency",
        "saturated_latency_summary": saturated["total_latency_ms"],
        "saturated_latency_by_slot_summary": saturated["latency_by_slot_summary"],
        "saturated_probe_samples": samples,
        "saturated_skip_samples": skip,
    }
    if paced_period_ms is not None:
        paced = _run_spawned_latency_probe(
            config_path=Path(config_path),
            samples=samples,
            skip=skip,
            period_ms=paced_period_ms,
            devices=config["executor"]["inference_devices"],
        )
        payload["paced_period_ms"] = paced_period_ms
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
    devices: list[str],
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    probes = []
    for device in devices:
        parent, child = ctx.Pipe()
        process = ctx.Process(
            target=_probe_child_main,
            args=(child, config_path, samples, skip, period_ms, device),
        )
        process.start()
        child.close()
        probes.append((parent, process))
    results = [parent.recv() for parent, _process in probes]
    for _parent, process in probes:
        process.join()
    slot_samples = [result["latency_samples_ms"] for result in results]
    return {
        "period_ms": period_ms,
        "sample_count": samples,
        "skip": skip,
        "latency_samples_by_slot_ms": slot_samples,
        "latency_samples_ms": [sample for samples in slot_samples for sample in samples],
        "total_latency_ms": _summary([sample for samples in slot_samples for sample in samples]),
        "latency_by_slot_summary": [result["total_latency_ms"] for result in results],
    }


def _probe_child_main(
    connection: Any,
    config_path: Path,
    samples: int,
    skip: int,
    period_ms: float,
    device: str,
) -> None:
    config = load_config(config_path)
    result = measure_policy_latency_in_current_process(
        config=config,
        samples=samples,
        skip=skip,
        period_ms=period_ms,
        device=device,
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
    device: str,
    sleep_fn,
) -> dict[str, Any]:
    from latency_bench.core.actions import ActionResolver
    from latency_bench.envs.factory import build_env_adapter
    from latency_bench.latency.measurement import RuntimeLatencyMeasurer
    from latency_bench.policy.factory import build_policy

    worker_config = deepcopy(config)
    worker_config["executor"]["inference_devices"] = [device]
    action_resolver = ActionResolver(worker_config["env"])
    env = build_env_adapter(worker_config, action_resolver.noop_action)
    policy = build_policy(worker_config, action_resolver)
    measurer = RuntimeLatencyMeasurer(
        device=device,
        sync_cuda=worker_config["executor"]["sync_cuda"],
    )

    latency_values: list[float] = []
    env.reset(seed=0)
    try:
        for obs_id in range(samples):
            sample_start = time.perf_counter()
            observation = env.observe()
            _output, record = measurer.run_and_measure(policy.predict, observation, obs_id=obs_id)
            if obs_id >= skip:
                latency_values.append(record.latency_ms)
            env.step(action_resolver.noop_action)
            elapsed_ms = (time.perf_counter() - sample_start) * 1000.0
            sleep_ms = period_ms - elapsed_ms
            if sleep_ms > 0.0:
                sleep_fn(sleep_ms / 1000.0)
    finally:
        env.close()

    return {
        "period_ms": period_ms,
        "sample_count": samples,
        "skip": skip,
        "latency_samples_ms": latency_values,
        "total_latency_ms": _summary(latency_values),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": percentile_linear(ordered, 90.0),
        "p95": percentile_linear(ordered, 95.0),
        "min": ordered[0],
        "max": ordered[-1],
    }
