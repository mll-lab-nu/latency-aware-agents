"""Evaluate or collect H8 student rollouts at one DAgger boundary."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import yaml

from latency_bench.core.types import Observation, PolicyOutput
from latency_bench.eval.driver import run_from_config
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.integrations.starvla_resident_dagger import ResidentInferencePool
from latency_bench.policy.base import PolicyRunner
from scripts.mikasa.evaluate import MikasaEnvStepBackend
from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import collect


class _ResidentPolicy(PolicyRunner):
    """Expose resident rank inference to the standard batched evaluator."""

    def __init__(self, pool: ResidentInferencePool):
        self.pool = pool

    def predict(self, observation: Observation) -> PolicyOutput:
        return self.pool.predict_batch([observation])[0]

    def predict_batch(self, observations: list[Observation]) -> list[PolicyOutput]:
        return self.pool.predict_batch(observations)

    def reset_state(self, slot_id: int | None = None) -> None:
        self.pool.reset_state(slot_id)


def _config(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    training = yaml.safe_load((args.source_run / "bootstrap.yaml").read_text(encoding="utf-8"))
    config["policy"]["checkpoint_path"] = str(args.checkpoint)
    config["policy"]["model_config_path"] = str(args.checkpoint.parent.parent / "config.yaml")
    config["policy"]["backbone_path"] = training["framework"]["qwenvl"]["base_vlm"]
    config["policy"]["task_contract_path"] = str(args.bootstrap / "lerobot/task_contract.json")
    config["policy"]["device"] = f"cuda:{args.inference_gpu}"
    config["executor"]["inference_devices"] = [f"cuda:{args.inference_gpu}"]
    config["executor"]["simulated_inference_pool"] = args.resident_control_dir is None
    config["env"]["simulator_device"] = f"cuda:{args.simulator_gpu}"
    config["experiment"]["seed"] = args.seed
    config["evaluation"]["eval_episodes"] = args.episodes
    config["evaluation"]["eval_parallel_envs"] = args.parallel_envs
    config["logging"]["output_dir"] = str(args.output)
    return config


def _resident_pool(args: argparse.Namespace) -> ResidentInferencePool:
    resident_config = json.loads(
        (args.resident_control_dir / "resident_policy.json").read_text(encoding="utf-8")
    )
    return ResidentInferencePool(
        socket_dir=Path(resident_config["socket_dir"]),
        step=args.resident_step,
        worker_count=args.num_gpus,
        batch_size=args.inference_batch_size,
    )


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.simulator_gpu)
    config = _config(args)
    if args.mode == "eval":
        args.output.mkdir(parents=True, exist_ok=False)
        config_path = args.output / "eval_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with ExitStack() as cleanup:
            policy = None
            if args.resident_control_dir is not None:
                pool = _resident_pool(args)
                cleanup.callback(pool.close)
                policy = _ResidentPolicy(pool)
            backend = MikasaEnvStepBackend(
                config=config, num_slots=args.parallel_envs,
            )
            run_from_config(config, env_backend=backend, policy=policy,
                            inference_devices=config["executor"]["inference_devices"])
        # The formal audit is owned by the existing MIKASA evaluation path.
        from scripts.starvla.train_task import evaluation_summary, mikasa_h8_execution_audit

        summary = evaluation_summary(args.output)
        audit = mikasa_h8_execution_audit(args.output)
        success_count = sum(
            episode["metadata"]["task_metrics"]["success"]
            for episode in summary["episodes"]
        )
        result = {
            "seed": args.seed,
            "episodes": args.episodes,
            "success_count": success_count,
            "success_rate": success_count / args.episodes,
            "mean_return": summary["mean_return"],
            "audit": audit,
            "checkpoint": str(args.checkpoint),
            "resident_step": args.resident_step,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return

    with ExitStack() as cleanup:
        if args.resident_control_dir is not None:
            pool = _resident_pool(args)
        else:
            pool = ProcessInferencePool(
                config=config, inference_devices=config["executor"]["inference_devices"]
            )
        cleanup.callback(pool.close)
        collection_args = argparse.Namespace(
            condition=args.condition,
            model=args.model,
            latency_config=args.latency_config,
            checkpoint=args.teacher_checkpoint,
            output=args.output,
            train_targets=args.head_budget,
            validation_targets=0,
            num_envs=args.parallel_envs,
        )
        result = collect(
            collection_args, student_pool=pool,
            seed_start=args.seed,
            simulator_device=f"cuda:{args.simulator_gpu}",
            student_source=str(args.checkpoint),
        )
    print(json.dumps({"valid_heads": result["splits"]["train"]["valid_actions"],
                      "episodes": len(result["saved_episodes"]),
                      "student_suffix_mae": result["student_suffix_mae"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eval", "collect"))
    parser.add_argument("--model", choices=("qwenoft", "qwenpi_v3", "qwengr00t"), required=True)
    parser.add_argument("--condition", choices=("profile-latency", "fixed-2", "zero-latency"), required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--latency-config", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--head-budget", type=int, default=16_000)
    parser.add_argument("--parallel-envs", type=int, default=10)
    parser.add_argument("--simulator-gpu", type=int, required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--resident-control-dir", type=Path)
    parser.add_argument("--resident-step", type=int)
    parser.add_argument("--num-gpus", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
