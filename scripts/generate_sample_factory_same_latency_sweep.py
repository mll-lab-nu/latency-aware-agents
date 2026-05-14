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

from training.sample_factory_common.config_adapter import load_training_config

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
    "flappy": 10000,
    "demon_attack": 25000,
    "deadly_corridor": 2100,
}

TOP_LEVEL_SAMPLE_FACTORY_KEYS = (
    "experiment",
    "backend",
    "env",
    "latency",
    "training",
    "evaluation",
    "logging",
)

LATENCY_KEYS = (
    "train_values",
    "train_mode",
    "train_sampling_mode",
    "distribution",
    "normal_mean",
    "normal_std",
    "normal_min",
    "normal_max",
    "uniform_min",
    "uniform_max",
    "episodes_per_stage",
    "randomize_within_stage",
    "within_episode_change_interval",
    "within_episode_change_prob",
    "add_latency_info",
)

LOGGING_KEYS = (
    "output_dir",
    "wandb_project",
    "wandb_group",
    "wandb_job_type",
    "wandb_tags",
)

EXPERIMENT_KEYS = (
    "name",
    "seed",
)

ENV_KEYS_BY_NAME = {
    "flappy": (
        "name",
        "gym_id",
        "use_lidar",
        "normalize_obs",
        "audio_on",
        "frame_stack",
        "screen_size",
        "pipe_gap",
        "bird_color",
        "pipe_color",
        "background",
        "score_limit",
        "debug",
        "max_episode_steps",
    ),
    "demon_attack": (
        "name",
        "env_id",
        "screen_size",
        "noop_max",
        "action_repeat",
        "frame_stack",
        "terminal_on_life_loss",
        "clip_reward",
        "repeat_action_probability",
        "full_action_space",
        "max_episode_steps",
    ),
    "deadly_corridor": (
        "name",
        "env_id",
        "env_frameskip",
        "frame_stack",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
        "max_episode_steps",
    ),
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
    frame_stack_override: int | None = None,
) -> dict:
    config = deepcopy(load_training_config(BASE_CONFIGS[env_name]))
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
    if frame_stack_override is not None:
        config["env"]["frame_stack"] = int(frame_stack_override)
    config["latency"]["train_values"] = [int(latency)]
    config["latency"]["train_mode"] = "episode_fixed"
    config["latency"]["train_sampling_mode"] = "uniform_all"
    config["latency"]["add_latency_info"] = False
    config["evaluation"]["eval_latency_values"] = [int(latency)]
    config["logging"]["output_dir"] = str(Path(output_root) / "train" / env_name / f"latency_{latency}" / f"seed_{seed}")
    if train_for_env_steps is not None:
        config["training"]["train_for_env_steps"] = int(train_for_env_steps)
    return _prune_sample_factory_config(config)


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
    config["latency"]["train_values"] = [int(latency)]
    config["latency"]["train_mode"] = "episode_fixed"
    config["latency"]["train_sampling_mode"] = "uniform_all"
    config["latency"]["add_latency_info"] = False
    config["evaluation"]["eval_episodes"] = int(eval_episodes)
    config["evaluation"]["eval_latency_values"] = [int(latency)]
    config["logging"]["output_dir"] = str(Path(output_root) / "eval" / env_name / f"latency_{latency}" / f"seed_{seed}")
    return _prune_sample_factory_config(config)


def _prune_sample_factory_config(config: dict) -> dict:
    pruned = {key: deepcopy(config[key]) for key in TOP_LEVEL_SAMPLE_FACTORY_KEYS if key in config}
    experiment_cfg = pruned.setdefault("experiment", {})
    pruned["experiment"] = {
        key: deepcopy(experiment_cfg[key])
        for key in EXPERIMENT_KEYS
        if key in experiment_cfg
    }
    env_cfg = pruned.setdefault("env", {})
    env_name = str(env_cfg.get("name", "")).strip().lower()
    env_keys = ENV_KEYS_BY_NAME.get(env_name, tuple(env_cfg.keys()))
    pruned["env"] = {
        key: deepcopy(env_cfg[key])
        for key in env_keys
        if key in env_cfg
    }
    latency_cfg = pruned.setdefault("latency", {})
    pruned["latency"] = {key: deepcopy(latency_cfg[key]) for key in LATENCY_KEYS if key in latency_cfg}
    logging_cfg = pruned.setdefault("logging", {})
    pruned["logging"] = {
        key: deepcopy(logging_cfg[key])
        for key in LOGGING_KEYS
        if key in logging_cfg and not (isinstance(logging_cfg[key], str) and not logging_cfg[key].strip())
    }
    return pruned


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
    parser.add_argument("--flappy-frame-stack", type=int)
    parser.add_argument("--demon-attack-frame-stack", type=int)
    parser.add_argument("--deadly-corridor-frame-stack", type=int)
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
    frame_stack_by_env = {
        "flappy": args.flappy_frame_stack,
        "demon_attack": args.demon_attack_frame_stack,
        "deadly_corridor": args.deadly_corridor_frame_stack,
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
                    frame_stack_override=frame_stack_by_env[env_name],
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
