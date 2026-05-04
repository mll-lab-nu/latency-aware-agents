from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from training.sample_factory_common.config_adapter import validate_training_config

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
    "wandb_name",
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
        "obs_resize",
        "obs_mode",
        "use_gpu_render",
        "gpu_render_device",
        "gpu_render_profile",
        "gpu_render_profile_interval",
        "pipe_gap",
        "bird_color",
        "pipe_color",
        "background",
        "score_limit",
        "debug",
        "debug_timelimit_diagnostics",
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
        "fps_env",
        "env_frameskip",
        "frame_stack",
        "res_w",
        "res_h",
        "wide_aspect_ratio",
        "max_episode_steps",
    ),
}

WANDB_ENV_SLUGS = {
    "flappy": "flappy-bird",
    "demon_attack": "demon-attack",
    "deadly_corridor": "deadly-corridor",
}

LATENCY_BENCH_POLICY_TYPES = {
    "flappy": "flappy_sf",
    "demon_attack": "demon_attack_sf",
    "deadly_corridor": "deadly_corridor_sf",
}


def parse_int_csv(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _build_wandb_name(env_name: str, latency: int, seed: int) -> str:
    env_slug = WANDB_ENV_SLUGS.get(env_name, env_name.replace("_", "-"))
    return f"{env_slug}-l{int(latency)}-s{int(seed)}"


def _build_run_name(env_name: str, latency: int, seed: int) -> str:
    return f"{env_name}_sf_same_latency_l{int(latency)}_seed{int(seed)}"


def _build_full_run_config(
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int | None,
    train_dir: str,
    output_root: str,
    device: str,
    restart_behavior: str,
    train_for_env_steps: int | None,
    frame_stack_override: int | None = None,
) -> dict:
    if env_name not in BASE_CONFIGS:
        raise ValueError(f"Unsupported env '{env_name}'. Expected one of: {sorted(BASE_CONFIGS)}")
    if int(latency) < 0:
        raise ValueError("latency must be non-negative")

    config = deepcopy(load_config(BASE_CONFIGS[env_name]))
    run_name = _build_run_name(env_name, latency, seed)
    experiment_cfg = config.setdefault("experiment", {})
    backend_cfg = config.setdefault("backend", {})
    env_cfg = config.setdefault("env", {})
    latency_cfg = config.setdefault("latency", {})
    logging_cfg = config.setdefault("logging", {})

    experiment_cfg["name"] = run_name
    experiment_cfg["seed"] = int(seed)
    backend_cfg["type"] = "sample_factory"
    backend_cfg["device"] = str(device)
    backend_cfg["train_dir"] = str(train_dir)
    backend_cfg["restart_behavior"] = str(restart_behavior)
    backend_cfg["run_mode"] = "train"
    if frame_stack_override is not None:
        env_cfg["frame_stack"] = int(frame_stack_override)
    if env_name == "flappy":
        env_cfg["obs_resize"] = [84, 84]
        env_cfg.setdefault("obs_mode", "image_only")
        env_cfg.pop("screen_size", None)
    latency_cfg["train_values"] = [int(latency)]
    latency_cfg["train_mode"] = "episode_fixed"
    latency_cfg["train_sampling_mode"] = "uniform_all"
    latency_cfg["add_latency_info"] = False
    logging_cfg["output_dir"] = str(
        Path(output_root) / "train" / env_name / f"latency_{int(latency)}" / f"seed_{int(seed)}"
    )
    logging_cfg["wandb_name"] = _build_wandb_name(env_name, latency, seed)
    training_cfg = config.setdefault("training", {})
    if train_for_env_steps is not None:
        training_cfg["train_for_env_steps"] = int(train_for_env_steps)
    validate_training_config(config)
    return config


def _build_train_config(
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int | None,
    train_dir: str,
    output_root: str,
    device: str,
    restart_behavior: str,
    train_for_env_steps: int | None,
    frame_stack_override: int | None = None,
) -> dict:
    return _prune_sample_factory_config(
        _build_full_run_config(
            env_name=env_name,
            seed=seed,
            latency=latency,
            max_steps=max_steps,
            train_dir=train_dir,
            output_root=output_root,
            device=device,
            restart_behavior=restart_behavior,
            train_for_env_steps=train_for_env_steps,
            frame_stack_override=frame_stack_override,
        )
    )


def _latency_steps_to_ms(config: dict, latency_steps: int) -> float:
    latency_steps = int(latency_steps)
    if latency_steps < 0:
        raise ValueError("latency_steps must be non-negative")
    fps_env = float(config["env"]["fps_env"])
    if fps_env <= 0:
        raise ValueError("env.fps_env must be positive")
    return latency_steps * 1000.0 / fps_env


def _latency_bench_policy_type(env_name: str) -> str:
    try:
        return LATENCY_BENCH_POLICY_TYPES[env_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported env '{env_name}'. Expected one of: {sorted(LATENCY_BENCH_POLICY_TYPES)}") from exc


def _require_eval_max_steps(config: dict) -> int:
    eval_max_steps = config.get("evaluation", {}).get("eval_max_steps")
    if eval_max_steps is None:
        raise ValueError("evaluation.eval_max_steps must be declared in the base config")
    return int(eval_max_steps)


def _drop_eval_time_limit_fields(config: dict) -> None:
    # Eval configs use evaluation.eval_max_steps as the single persisted step-limit source.
    config.setdefault("env", {}).pop("max_episode_steps", None)


def _align_demon_attack_raw_eval_config(config: dict, env_name: str) -> None:
    if env_name != "demon_attack":
        return
    if not bool(config.get("evaluation", {}).get("eval_raw_reward", False)):
        return
    env_cfg = config.setdefault("env", {})
    env_cfg["terminal_on_life_loss"] = True
    env_cfg["clip_reward"] = False


def _build_latency_bench_eval_config(
    full_config: dict,
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int | None,
    output_root: str,
    eval_episodes: int,
    latency_bench_policy_device: str = "auto",
) -> dict:
    config = deepcopy(full_config)
    _require_eval_max_steps(config)
    _drop_eval_time_limit_fields(config)
    _align_demon_attack_raw_eval_config(config, env_name)

    experiment_cfg = config.setdefault("experiment", {})
    experiment_cfg["name"] = _build_run_name(env_name, latency, seed)
    experiment_cfg["seed"] = int(seed)
    experiment_cfg["num_episodes"] = int(eval_episodes)

    config.setdefault("backend", {})["run_mode"] = "eval"

    latency_cfg = config.setdefault("latency", {})
    latency_cfg["train_values"] = [int(latency)]
    latency_cfg["train_mode"] = "episode_fixed"
    latency_cfg["train_sampling_mode"] = "uniform_all"
    latency_cfg["add_latency_info"] = False
    if int(latency) == 0:
        latency_cfg["type"] = "zero"
        latency_cfg.pop("fixed_latency_ms", None)
    else:
        latency_cfg["type"] = "fixed"
        latency_cfg["fixed_latency_ms"] = _latency_steps_to_ms(config, int(latency))

    evaluation_cfg = config.setdefault("evaluation", {})
    evaluation_cfg["eval_episodes"] = int(eval_episodes)

    policy_cfg = deepcopy(config.get("policy", {}))
    policy_cfg.update(
        {
            "type": _latency_bench_policy_type(env_name),
            "checkpoint_path": "",
            "device": str(latency_bench_policy_device),
            "deterministic": True,
        }
    )
    config["policy"] = policy_cfg
    config["logging"] = {
        "output_dir": str(
            Path(output_root) / "latency_bench" / env_name / f"train_latency_{int(latency)}" / f"seed_{int(seed)}"
        )
    }
    if env_name == "deadly_corridor":
        config.setdefault("env", {})["noop_action"] = [0, 0, 0, 0]
    return config


def _build_eval_config(
    train_config: dict,
    env_name: str,
    seed: int,
    latency: int,
    max_steps: int | None,
    output_root: str,
    eval_episodes: int,
) -> dict:
    config = deepcopy(train_config)
    _require_eval_max_steps(config)
    _drop_eval_time_limit_fields(config)
    _align_demon_attack_raw_eval_config(config, env_name)
    config["experiment"]["num_episodes"] = int(eval_episodes)
    config["backend"]["run_mode"] = "eval"
    config["latency"]["train_values"] = [int(latency)]
    config["latency"]["train_mode"] = "episode_fixed"
    config["latency"]["train_sampling_mode"] = "uniform_all"
    config["latency"]["add_latency_info"] = False
    config["evaluation"]["eval_episodes"] = int(eval_episodes)
    config["logging"] = {
        "output_dir": str(
            Path(output_root) / "eval" / env_name / f"latency_{latency}" / f"seed_{seed}"
        )
    }
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, default="flappy,demon_attack,deadly_corridor")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--flappy-latencies", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--demon-attack-latencies", type=str, default="0,1,2,3,4,5,10,15")
    parser.add_argument("--deadly-corridor-latencies", type=str, default="0,1,2,3,4,5,10,15,40")
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
    parser.add_argument("--latency-bench-policy-device", type=str, default="auto")
    args = parser.parse_args(argv)

    selected_envs = [item.strip() for item in args.envs.split(",") if item.strip()]
    seeds = parse_int_csv(args.seeds)
    latency_by_env = {
        "flappy": parse_int_csv(args.flappy_latencies) or DEFAULT_LATENCIES["flappy"],
        "demon_attack": parse_int_csv(args.demon_attack_latencies) or DEFAULT_LATENCIES["demon_attack"],
        "deadly_corridor": parse_int_csv(args.deadly_corridor_latencies) or DEFAULT_LATENCIES["deadly_corridor"],
    }
    frame_stack_by_env = {
        "flappy": args.flappy_frame_stack,
        "demon_attack": args.demon_attack_frame_stack,
        "deadly_corridor": args.deadly_corridor_frame_stack,
    }

    generated_root = Path(args.generated_root)
    generated_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = [
        "\t".join(
            [
                "env",
                "latency",
                "seed",
                "train_config",
                "eval_config",
                "latency_bench_eval_config",
                "train_command",
                "eval_command",
                "latency_bench_eval_command",
            ]
        )
    ]
    train_commands: list[str] = []
    eval_commands: list[str] = []
    latency_bench_eval_commands: list[str] = []

    for env_name in selected_envs:
        if env_name not in BASE_CONFIGS:
            raise ValueError(f"Unsupported env '{env_name}'. Expected one of: {sorted(BASE_CONFIGS)}")
        for latency in latency_by_env[env_name]:
            for seed in seeds:
                full_config = _build_full_run_config(
                    env_name=env_name,
                    seed=int(seed),
                    latency=int(latency),
                    max_steps=None,
                    train_dir=args.train_dir,
                    output_root=args.output_root,
                    device=args.device,
                    restart_behavior=args.restart_behavior,
                    train_for_env_steps=args.train_for_env_steps,
                    frame_stack_override=frame_stack_by_env[env_name],
                )
                train_config = _prune_sample_factory_config(full_config)
                eval_config = _build_eval_config(
                    train_config=full_config,
                    env_name=env_name,
                    seed=int(seed),
                    latency=int(latency),
                    max_steps=None,
                    output_root=args.output_root,
                    eval_episodes=args.eval_episodes,
                )
                latency_bench_eval_config = _build_latency_bench_eval_config(
                    full_config=full_config,
                    env_name=env_name,
                    seed=int(seed),
                    latency=int(latency),
                    max_steps=None,
                    output_root=args.output_root,
                    eval_episodes=args.eval_episodes,
                    latency_bench_policy_device=args.latency_bench_policy_device,
                )

                env_dir = generated_root / env_name / f"latency_{latency}" / f"seed_{seed}"
                train_path = env_dir / "train.yaml"
                eval_path = env_dir / "eval.yaml"
                latency_bench_eval_path = env_dir / "latency_bench_eval.yaml"
                _write_yaml(train_path, train_config)
                _write_yaml(eval_path, eval_config)
                _write_yaml(latency_bench_eval_path, latency_bench_eval_config)

                train_cmd = f"python -m training.train --config {shlex.quote(str(train_path))}"
                eval_cmd = f"python -m training.train --config {shlex.quote(str(eval_path))}"
                latency_bench_eval_cmd = (
                    f"python -m latency_bench.run --config {shlex.quote(str(latency_bench_eval_path))}"
                )
                manifest_rows.append(
                    "\t".join(
                        [
                            env_name,
                            str(int(latency)),
                            str(int(seed)),
                            str(train_path),
                            str(eval_path),
                            str(latency_bench_eval_path),
                            train_cmd,
                            eval_cmd,
                            latency_bench_eval_cmd,
                        ]
                    )
                )
                train_commands.append(train_cmd)
                eval_commands.append(eval_cmd)
                latency_bench_eval_commands.append(latency_bench_eval_cmd)

    (generated_root / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (generated_root / "train_commands.txt").write_text("\n".join(train_commands) + "\n", encoding="utf-8")
    (generated_root / "eval_commands.txt").write_text("\n".join(eval_commands) + "\n", encoding="utf-8")
    (generated_root / "latency_bench_eval_commands.txt").write_text(
        "\n".join(latency_bench_eval_commands) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote sweep configs to {generated_root}")
    print(f"Train commands: {generated_root / 'train_commands.txt'}")
    print(f"Eval commands: {generated_root / 'eval_commands.txt'}")
    print(f"Latency bench eval commands: {generated_root / 'latency_bench_eval_commands.txt'}")
    print(f"Manifest: {generated_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
