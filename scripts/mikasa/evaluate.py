#!/usr/bin/env python3
"""PPO evaluation and the existing native-backend latency evaluation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# Preserve direct script invocation from the simulator's Python environment.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend  # noqa: E402
from latency_bench.envs.tasks.mikasa import make_ppo_env  # noqa: E402


START_SEED = 4242424242


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].cpu().item()
    return np.asarray(value).reshape(-1)[0].item()


from latency_bench.policy.mikasa_ppo import MikasaPpoPolicy


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    env = make_ppo_env()
    policy = MikasaPpoPolicy(env, args.checkpoint)
    seeds = []
    successes = []
    returns = []
    lengths = []
    try:
        for episode_index in range(args.episodes):
            seed = START_SEED + episode_index
            observation, _ = env.reset(seed=seed)
            success_once = False
            episode_return = 0.0
            for step in range(60):
                action = policy.forward(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                success_once = success_once or bool(_scalar(info["success"]))
                episode_return += float(_scalar(reward))
                if bool(_scalar(terminated)) or bool(_scalar(truncated)):
                    break
            seeds.append(seed)
            successes.append(success_once)
            returns.append(episode_return)
            lengths.append(step + 1)
    finally:
        env.close()
    summary = {
        "seeds": seeds,
        "successes": successes,
        "success_rate": float(np.mean(successes)),
        "returns": returns,
        "lengths": lengths,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _latency_eval(argv: list[str]) -> None:
    from latency_bench.core.config import load_config
    from latency_bench.eval.config import apply_runtime_overrides
    from latency_bench.eval.driver import run_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--model-config-path", type=Path)
    parser.add_argument("--task-contract-path", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--latency-method", choices=("zero", "temporal"))
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--latency-seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.eval_config)
    apply_runtime_overrides(
        config,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        task_contract_path=args.task_contract_path,
        run_name=args.run_name,
        output_dir=args.output_dir,
        latency_method=args.latency_method,
        latency_profile_path=args.profile_path,
        latency_seed=args.latency_seed,
    )
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    backend = None
    if config["executor"]["mode"] == "simulated":
        backend = MikasaEnvStepBackend(
            config=config,
            num_slots=int(config["executor"]["parallel_envs"]),
        )
    run_from_config(
        config,
        env_backend=backend,
        inference_devices=config["executor"]["inference_devices"],
    )


def main() -> None:
    if sys.argv[1:2] == ["latency-eval"]:
        _latency_eval(sys.argv[2:])
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ppo",), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _evaluate(args)



if __name__ == "__main__":
    main()
