from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime

from training.demon_attack_ppo.logging import setup_wandb
from training.demon_attack_ppo.ppo_loop import train


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CleanRL-style PPO for DemonAttack using Atari frame observations."
    )
    parser.add_argument("--train-steps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--save", type=str, default="./models/demon_attack_cleanrl.pt")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--log-interval-seconds", type=float, default=10.0)
    parser.add_argument("--stats-window-size", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-sync-cuda", action="store_true")
    parser.add_argument("--profile-csv", type=str, default="")
    parser.add_argument("--profile-warmup-updates", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--learning-rate-final", type=float, default=2.5e-6)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-coef", type=float, default=0.15)
    parser.add_argument("--clip-coef-final", type=float, default=0.025)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--env-id", type=str, default="ALE/DemonAttack-v5")
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--base-channels", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--action-repeat", type=_positive_int, default=1)
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--clip-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--stack-frames", type=int, default=4)
    parser.add_argument(
        "--obs-mode",
        choices=["image_only", "latency_channel"],
        default="latency_channel",
    )
    parser.add_argument("--frame-skip-values", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--wandb-project", type=str, default="demon_attack_rl_queue_latency")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--metrics-jsonl", type=str, default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_id:
        args.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    wandb_module = setup_wandb(args)
    try:
        train(args, wandb_module=wandb_module)
    finally:
        wandb_module.finish()


if __name__ == "__main__":
    main()
