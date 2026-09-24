"""Evaluate MIKASA H8 with its trained prefix and an independent execution delay."""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES

from scripts.starvla.train_task import (
    evaluate,
    evaluation_summary,
    mikasa_h8_execution_audit,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwenoft", "qwenpi_v3", "qwengr00t"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vla-python", type=Path, required=True)
    parser.add_argument("--task-python", type=Path, required=True)
    parser.add_argument("--eval-cuda-visible-devices", required=True)
    parser.add_argument("--inference-device", default="cuda:1")
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4242424242)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--setting", choices=("zero", "profile", *FIXED_LATENCIES), required=True)
    parser.add_argument("--label")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()
    if not args.smoke and not args.skip_wandb and args.wandb_run_id is None:
        parser.error("--wandb-run-id is required unless --skip-wandb is set")

    run = json.loads((args.run_dir / "run.json").read_text())
    training = yaml.safe_load((args.run_dir / "bootstrap.yaml").read_text())
    runtime = SimpleNamespace(
        task=run["task"],
        model=args.model,
        condition=run["condition"],
        eval_seed=args.seed,
        inference_device=args.inference_device,
        simulator_device=args.simulator_device,
        smoke=args.smoke,
        vla_python=args.vla_python,
        backbone_path=Path(training["framework"]["qwenvl"]["base_vlm"]),
        task_python=args.task_python,
        output_dir=args.run_dir,
        eval_episodes=args.episodes,
        eval_cuda_visible_devices=args.eval_cuda_visible_devices,
    )
    label = args.label or f"final-{args.setting}"
    evaluation = evaluate(
        runtime,
        args.run_dir / "inference",
        Path("unused"),
        label=label,
        wandb_checkpoint=None if args.smoke or args.skip_wandb else (
            args.wandb_run_id, 5000
        ),
        eval_latency_frames={"zero": 0, "profile": None, **FIXED_LATENCIES}[args.setting],
    )
    summary = evaluation_summary(evaluation)
    audit = mikasa_h8_execution_audit(evaluation)
    success_count = sum(
        episode["metadata"]["task_metrics"]["success"]
        for episode in summary["episodes"]
    )
    threshold_count = args.episodes // 2
    acceptance = {
        "metric": "success_once",
        "success_count": success_count,
        "episode_count": args.episodes,
        "success_rate": success_count / args.episodes,
        "threshold_count": threshold_count,
        "passed": success_count >= threshold_count,
    }
    summary["chunk_execution_audit"] = audit
    summary["acceptance"] = acceptance
    if not args.smoke:
        result_name = {
            "zero": "eval_l0.json",
            "profile": "eval_profile.json",
            **{condition: f"eval_fixed_{frames}f.json" for condition, frames in FIXED_LATENCIES.items()},
        }[args.setting]
        write_json(args.run_dir / "inference" / result_name, summary)
        write_json(
            args.run_dir / f"inference/chunk_execution_audit_{args.setting}.json",
            audit,
        )
        write_json(
            args.run_dir / f"inference/acceptance_{args.setting}.json",
            acceptance,
        )
        evaluation_record = {
            "seed": args.seed,
            "episodes": args.episodes,
            "executor": "simulated",
            "request_fps": 20,
            "simulated_worker_capacity": 1,
            "latency_method": audit["latency_method"],
            "startup_policy": "environment_starts_at_reset",
        }
        if args.setting == "profile":
            evaluation_record.update({
                "latency_seed": audit["latency_seed"],
                "profile_path": audit["profile_path"],
            })
        elif args.setting in FIXED_LATENCIES:
            evaluation_record.update({
                "fixed_latency_ms": audit["fixed_latency_ms"],
                "latency_raw_frames": audit["latency_raw_frames"],
            })
        if "evaluation_matrix" not in run:
            run["evaluation_matrix"] = {}
        if "acceptance_matrix" not in run:
            run["acceptance_matrix"] = {}
        run["evaluation_matrix"][args.setting] = evaluation_record
        run["acceptance_matrix"][args.setting] = acceptance
        if "evaluation" in run:
            del run["evaluation"]
        if "acceptance" in run:
            del run["acceptance"]
        write_json(args.run_dir / "run.json", run)
    print(json.dumps({"summary": summary, "audit": audit, "acceptance": acceptance}))


if __name__ == "__main__":
    main()
