from __future__ import annotations

import argparse
import shlex
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config

GENERATED_ROOT = REPO_ROOT / "configs" / "generated" / "sample_factory_same_latency_sweep"

BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
}

DEFAULT_LATENCIES = {
    "flappy": [0, 1, 2, 3, 4, 5],
    "demon_attack": [0, 1, 2, 3, 4, 5, 10, 15],
    "deadly_corridor": [0, 1, 2, 3, 4, 5, 10, 15, 40],
}

DEFAULT_MAX_STEPS = {
    "flappy": 1800,
    "demon_attack": 3600,
    "deadly_corridor": 2100,
}


def parse_int_csv(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _build_train_config(
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int,
    train_dir: str,
    output_root: str,
    device: str,
    restart_behavior: str,
    train_for_env_steps: int | None,
) -> dict:
    config = deepcopy(load_config(BASE_CONFIGS[env_name]))
    run_name = f"{env_name}_sf_same_latency_l{latency}_seed{seed}"
    config["experiment"]["name"] = run_name
    config["experiment"]["seed"] = int(seed)
    config["experiment"]["max_env_steps"] = int(max_steps)
    config["backend"]["type"] = "sample_factory"
    config["backend"]["device"] = str(device)
    config["backend"]["train_dir"] = str(train_dir)
    config["backend"]["restart_behavior"] = str(restart_behavior)
    config["backend"]["run_mode"] = "train"
    config["env"]["max_episode_steps"] = int(max_steps)
    config["latency"]["train_enabled"] = True
    config["latency"]["train_values"] = [int(latency)]
    config["latency"]["train_mode"] = "episode_fixed"
    config["latency"]["train_sampling_mode"] = "uniform_all"
    config["evaluation"]["eval_latency_values"] = [int(latency)]
    config["evaluation"]["eval_max_steps"] = int(max_steps)
    config["logging"]["output_dir"] = str(Path(output_root) / "train" / env_name / f"latency_{latency}" / f"seed_{seed}")
    if train_for_env_steps is not None:
        config["training"]["train_for_env_steps"] = int(train_for_env_steps)
    return config


def _build_eval_config(
    train_config: dict,
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int,
    output_root: str,
    eval_episodes: int,
) -> dict:
    config = deepcopy(train_config)
    config["experiment"]["max_env_steps"] = int(max_steps)
    config["experiment"]["num_episodes"] = int(eval_episodes)
    config["backend"]["run_mode"] = "eval"
    config["latency"]["train_enabled"] = True
    config["latency"]["train_values"] = [int(latency)]
    config["latency"]["train_mode"] = "episode_fixed"
    config["latency"]["train_sampling_mode"] = "uniform_all"
    config["evaluation"]["eval_episodes"] = int(eval_episodes)
    config["evaluation"]["eval_latency_values"] = [int(latency)]
    config["evaluation"]["eval_max_steps"] = int(max_steps)
    config["logging"]["output_dir"] = str(Path(output_root) / "eval" / env_name / f"latency_{latency}" / f"seed_{seed}")
    return config


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, default="flappy,demon_attack,deadly_corridor")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--flappy-latencies", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--demon-attack-latencies", type=str, default="0,1,2,3,4,5,10,15")
    parser.add_argument("--deadly-corridor-latencies", type=str, default="0,1,2,3,4,5,10,15,40")
    parser.add_argument("--flappy-max-steps", type=int, default=DEFAULT_MAX_STEPS["flappy"])
    parser.add_argument("--demon-attack-max-steps", type=int, default=DEFAULT_MAX_STEPS["demon_attack"])
    parser.add_argument("--deadly-corridor-max-steps", type=int, default=DEFAULT_MAX_STEPS["deadly_corridor"])
    parser.add_argument("--train-dir", type=str, default="/workspace/sample_factory_runs")
    parser.add_argument("--output-root", type=str, default="/workspace/latency_sweeps")
    parser.add_argument("--generated-root", type=Path, default=GENERATED_ROOT)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--restart-behavior", type=str, default="resume")
    parser.add_argument("--train-for-env-steps", type=int)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    selected_envs = [item.strip() for item in args.envs.split(",") if item.strip()]
    seeds = parse_int_csv(args.seeds)
    latency_by_env = {
        "flappy": parse_int_csv(args.flappy_latencies) or DEFAULT_LATENCIES["flappy"],
        "demon_attack": parse_int_csv(args.demon_attack_latencies) or DEFAULT_LATENCIES["demon_attack"],
        "deadly_corridor": parse_int_csv(args.deadly_corridor_latencies) or DEFAULT_LATENCIES["deadly_corridor"],
    }
    max_steps_by_env = {
        "flappy": int(args.flappy_max_steps),
        "demon_attack": int(args.demon_attack_max_steps),
        "deadly_corridor": int(args.deadly_corridor_max_steps),
    }

    generated_root = Path(args.generated_root)
    generated_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = ["env\tlatency\tseed\ttrain_config\teval_config\ttrain_command\teval_command"]
    train_commands: list[str] = []
    eval_commands: list[str] = []

    for env_name in selected_envs:
        if env_name not in BASE_CONFIGS:
            raise ValueError(f"Unsupported env '{env_name}'. Expected one of: {sorted(BASE_CONFIGS)}")
        for latency in latency_by_env[env_name]:
            for seed in seeds:
                max_steps = max_steps_by_env[env_name]
                train_config = _build_train_config(
                    env_name=env_name,
                    seed=int(seed),
                    latency=int(latency),
                    max_steps=max_steps,
                    train_dir=args.train_dir,
                    output_root=args.output_root,
                    device=args.device,
                    restart_behavior=args.restart_behavior,
                    train_for_env_steps=args.train_for_env_steps,
                )
                eval_config = _build_eval_config(
                    train_config=train_config,
                    env_name=env_name,
                    seed=int(seed),
                    latency=int(latency),
                    max_steps=max_steps,
                    output_root=args.output_root,
                    eval_episodes=args.eval_episodes,
                )

                run_name = train_config["experiment"]["name"]
                env_dir = generated_root / env_name / f"latency_{latency}" / f"seed_{seed}"
                train_path = env_dir / "train.yaml"
                eval_path = env_dir / "eval.yaml"
                _write_yaml(train_path, train_config)
                _write_yaml(eval_path, eval_config)

                train_cmd = f"python -m training.train --config {shlex.quote(str(train_path))}"
                eval_cmd = f"python -m training.train --config {shlex.quote(str(eval_path))}"
                manifest_rows.append(
                    "\t".join(
                        [
                            env_name,
                            str(int(latency)),
                            str(int(seed)),
                            str(train_path),
                            str(eval_path),
                            train_cmd,
                            eval_cmd,
                        ]
                    )
                )
                train_commands.append(train_cmd)
                eval_commands.append(eval_cmd)

    (generated_root / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (generated_root / "train_commands.txt").write_text("\n".join(train_commands) + "\n", encoding="utf-8")
    (generated_root / "eval_commands.txt").write_text("\n".join(eval_commands) + "\n", encoding="utf-8")
    print(f"Wrote sweep configs to {generated_root}")
    print(f"Train commands: {generated_root / 'train_commands.txt'}")
    print(f"Eval commands: {generated_root / 'eval_commands.txt'}")
    print(f"Manifest: {generated_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
