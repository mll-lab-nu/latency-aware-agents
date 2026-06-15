from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from latency_bench.core.config import load_config
from latency_bench.executors.factory import build_executor
from latency_bench.latency.env_fps_calibration import select_env_fps_from_warmup
from latency_bench.latency.profile import load_latency_samples


def calibrate_env_fps_from_warmup(
    *,
    config_path: str | Path,
    output: str | Path,
    target_latency_frames: int,
    worker_utilization_cap: float,
) -> dict[str, float | int]:
    config = load_config(config_path)
    executor = build_executor(config)
    try:
        executor.run_warmup(eval_max_steps=int(config["evaluation"]["eval_max_steps"]))
    finally:
        close = getattr(executor, "close", None)
        if callable(close):
            close()
        else:
            executor.env.close()

    samples = load_latency_samples(Path(config["logging"]["output_dir"]) / "warmup_latencies.jsonl")
    calibration = select_env_fps_from_warmup(
        latency_samples_ms=samples,
        target_latency_frames=target_latency_frames,
        base_env_fps=float(config["env"]["env_fps"]),
        base_obs_fps=float(config["env"]["obs_fps"]),
        device_count=_visible_cuda_device_count(),
        worker_utilization_cap=worker_utilization_cap,
    )
    payload = calibration.to_dict()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _visible_cuda_device_count() -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return int(torch.cuda.device_count())
    stripped = visible_devices.strip()
    return 0 if stripped in {"", "-1"} else len([item for item in stripped.split(",") if item.strip()])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate env_fps from realtime warmup latency samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-latency-frames", type=int, required=True)
    parser.add_argument("--worker-utilization-cap", type=float, required=True)
    args = parser.parse_args(argv)
    calibrate_env_fps_from_warmup(
        config_path=args.config,
        output=args.output,
        target_latency_frames=args.target_latency_frames,
        worker_utilization_cap=args.worker_utilization_cap,
    )


if __name__ == "__main__":
    main()
