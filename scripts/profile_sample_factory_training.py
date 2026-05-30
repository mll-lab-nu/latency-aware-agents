from __future__ import annotations

import argparse
import csv
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
from training.common.config_adapter import import_integration_module


BASE_CONFIGS = {
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
}


def _csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _latency_config_for_case(case: str, *, env_fps: float) -> dict[str, Any]:
    def frames_ms(raw_frames: float) -> float:
        return raw_frames * 1000.0 / env_fps

    if case == "fixed":
        return {
            "type": "fixed",
            "fixed_latency_ms": frames_ms(4.0),
            "add_latency_info": False,
        }
    if case == "normal":
        return {
            "type": "normal_distribution",
            "mean_latency_ms": frames_ms(4.0),
            "std_latency_ms": frames_ms(1.0),
            "min_latency_ms": 0.0,
            "max_latency_ms": frames_ms(15.0),
            "add_latency_info": False,
        }
    return {
        "type": "uniform_distribution",
        "min_latency_ms": frames_ms(2.0),
        "max_latency_ms": frames_ms(6.0),
        "add_latency_info": False,
    }


def _stage_config(
    base_config: dict[str, Any],
    *,
    env_name: str,
    simulator: str,
    latency_case: str,
    async_rl: bool,
    train_for_env_steps: int,
    num_workers: int,
    sf_num_envs_per_worker: int,
    worker_num_splits: int,
    batch_size: int,
    gpu_env_batch_size: int,
    rollout: int | None,
    recurrence: int | None,
    num_batches_per_epoch: int | None,
    output_root: Path,
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    async_label = "async" if async_rl else "sync"
    dimension_label = (
        f"g{int(gpu_env_batch_size)}"
        if simulator == "gpu"
        else f"{int(num_workers)}w{int(sf_num_envs_per_worker)}e"
    )
    cfg["experiment"]["name"] = f"profile_{env_name}_{simulator}_{dimension_label}_{latency_case}_{async_label}"
    cfg["backend"]["run_mode"] = "train"
    cfg["backend"]["restart_behavior"] = "overwrite"
    cfg["env"]["simulator"] = simulator
    if simulator == "gpu":
        cfg["backend"]["device"] = "cuda"
        cfg["env"]["gpu_env_batch_size"] = int(gpu_env_batch_size)
    cfg["latency"] = _latency_config_for_case(latency_case, env_fps=float(cfg["env"]["env_fps"]))
    cfg["training"]["async_rl"] = async_rl
    cfg["training"]["train_for_env_steps"] = int(train_for_env_steps)
    cfg["training"]["num_workers"] = int(num_workers)
    cfg["training"]["num_envs_per_worker"] = int(sf_num_envs_per_worker)
    cfg["training"]["worker_num_splits"] = int(worker_num_splits)
    cfg["training"]["batch_size"] = int(batch_size)
    if rollout is not None:
        cfg["training"]["rollout"] = int(rollout)
    if recurrence is not None:
        cfg["training"]["recurrence"] = int(recurrence)
    if num_batches_per_epoch is not None:
        cfg["training"]["num_batches_per_epoch"] = int(num_batches_per_epoch)
    cfg["evaluation"]["eval_interval_steps"] = None
    cfg["logging"]["wandb_project"] = ""
    cfg["logging"]["output_dir"] = str(output_root / env_name / simulator / dimension_label / latency_case / async_label)
    return cfg


def _run_dimensions(args: argparse.Namespace, simulator: str) -> list[dict[str, int]]:
    if simulator == "gpu":
        return [
            {
                "num_workers": int(args.gpu_num_workers),
                "sf_num_envs_per_worker": int(args.gpu_sf_num_envs_per_worker),
                "worker_num_splits": int(args.gpu_worker_num_splits),
                "gpu_env_batch_size": int(batch_size),
            }
            for batch_size in _csv_items(args.gpu_env_batch_sizes)
        ]
    return [
        {
            "num_workers": int(args.cpu_num_workers),
            "sf_num_envs_per_worker": int(args.cpu_num_envs_per_worker),
            "worker_num_splits": int(args.cpu_worker_num_splits),
            "gpu_env_batch_size": 0,
        }
    ]


def _run_stage(cfg: dict[str, Any]) -> tuple[int, float]:
    module = import_integration_module(cfg)
    before = time.perf_counter()
    status = int(module.run_from_config(cfg))
    return status, time.perf_counter() - before


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Demon Attack and Deadly Corridor Sample Factory training.")
    parser.add_argument("--envs", default="demon_attack,deadly_corridor")
    parser.add_argument("--simulators", default="cpu,gpu")
    parser.add_argument("--latency-cases", default="fixed,normal,uniform")
    parser.add_argument("--async-rl-values", default="true,false")
    parser.add_argument("--train-for-env-steps", type=int, default=65536)
    parser.add_argument("--cpu-num-workers", type=int, default=16)
    parser.add_argument("--cpu-num-envs-per-worker", type=int, default=4)
    parser.add_argument("--cpu-worker-num-splits", type=int, default=2)
    parser.add_argument("--gpu-num-workers", type=int, default=1)
    parser.add_argument("--gpu-sf-num-envs-per-worker", type=int, default=1)
    parser.add_argument("--gpu-worker-num-splits", type=int, default=1)
    parser.add_argument("--gpu-env-batch-sizes", default="512,1024,1536")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--recurrence", type=int)
    parser.add_argument("--num-batches-per-epoch", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sample_factory_profile"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "env_name",
            "simulator",
            "latency_case",
            "async_rl",
            "status",
            "wall_sec",
            "train_for_env_steps",
            "sf_num_workers",
            "sf_num_envs_per_worker",
            "worker_num_splits",
            "gpu_env_batch_size",
            "effective_envs",
            "batch_size",
            "fps_estimate",
            "output_dir",
        ],
    )
    writer.writeheader()

    for env_name in _csv_items(args.envs):
        base_config = load_config(BASE_CONFIGS[env_name])
        for simulator in _csv_items(args.simulators):
            for dimensions in _run_dimensions(args, simulator):
                for latency_case in _csv_items(args.latency_cases):
                    for async_text in _csv_items(args.async_rl_values):
                        async_rl = async_text.lower() == "true"
                        cfg = _stage_config(
                            base_config,
                            env_name=env_name,
                            simulator=simulator,
                            latency_case=latency_case,
                            async_rl=async_rl,
                            train_for_env_steps=args.train_for_env_steps,
                            num_workers=dimensions["num_workers"],
                            sf_num_envs_per_worker=dimensions["sf_num_envs_per_worker"],
                            worker_num_splits=dimensions["worker_num_splits"],
                            batch_size=args.batch_size,
                            gpu_env_batch_size=dimensions["gpu_env_batch_size"],
                            rollout=args.rollout,
                            recurrence=args.recurrence,
                            num_batches_per_epoch=args.num_batches_per_epoch,
                            output_root=args.output_root,
                        )
                        effective_envs = (
                            dimensions["gpu_env_batch_size"]
                            if simulator == "gpu"
                            else dimensions["num_workers"] * dimensions["sf_num_envs_per_worker"]
                        )
                        status = 0
                        wall_sec = 0.0
                        if not args.dry_run:
                            status, wall_sec = _run_stage(cfg)
                        writer.writerow(
                            {
                                "env_name": env_name,
                                "simulator": simulator,
                                "latency_case": latency_case,
                                "async_rl": int(async_rl),
                                "status": status,
                                "wall_sec": f"{wall_sec:.6f}",
                                "train_for_env_steps": int(args.train_for_env_steps),
                                "sf_num_workers": dimensions["num_workers"],
                                "sf_num_envs_per_worker": dimensions["sf_num_envs_per_worker"],
                                "worker_num_splits": dimensions["worker_num_splits"],
                                "gpu_env_batch_size": "" if simulator != "gpu" else dimensions["gpu_env_batch_size"],
                                "effective_envs": effective_envs,
                                "batch_size": int(args.batch_size),
                                "fps_estimate": "" if wall_sec == 0.0 else f"{float(args.train_for_env_steps) / wall_sec:.6f}",
                                "output_dir": cfg["logging"]["output_dir"],
                            }
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
