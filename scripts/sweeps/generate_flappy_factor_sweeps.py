from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config, merge_dicts
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    distribution_frames_to_ms_latency_config,
)
from scripts.generate_sample_factory_same_latency_sweep import _prune_sample_factory_config
from scripts.generate_sample_factory_same_latency_sweep import _align_demon_attack_raw_eval_config
from scripts.generate_sample_factory_same_latency_sweep import _drop_eval_time_limit_fields
from scripts.generate_sample_factory_same_latency_sweep import LATENCY_BENCH_POLICY_TYPES


DEFAULT_SPEC = REPO_ROOT / "configs" / "experiments" / "flappy" / "factor_sweeps" / "sweep.yaml"
MANIFEST_COLUMNS = (
    "sweep",
    "latency_label",
    "frame_stack",
    "obs_fps",
    "seed",
    "train_config",
    "latency_bench_eval_config",
    "train_command",
    "latency_bench_eval_command",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _frame_ms(config: dict[str, Any]) -> float:
    return 1000.0 / float(config["env"]["env_fps"])


def _env_name(config: dict[str, Any]) -> str:
    return str(config["env"]["name"])


def _env_slug(env_name: str) -> str:
    return str(env_name).replace("_", "-")


def _number_label(value: int | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def _parse_int_csv(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _apply_latency_setting(config: dict[str, Any], setting: dict[str, Any], seed: int) -> None:
    latency_cfg = config.setdefault("latency", {})
    kind = str(setting["kind"])
    for key in (
        "fixed_latency_ms",
        "mean_latency_ms",
        "std_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "seed",
    ):
        latency_cfg.pop(key, None)

    if kind == "fixed":
        latency_cfg["type"] = "fixed"
        latency_cfg["fixed_latency_ms"] = float(setting["frames"]) * _frame_ms(config)
    else:
        spec = _latency_distribution_spec(kind, setting)
        latency_cfg.update(
            distribution_frames_to_ms_latency_config(
                spec,
                env_fps=float(config["env"]["env_fps"]),
                seed=int(seed),
            )
        )
    latency_cfg["add_latency_info"] = False


def _latency_distribution_spec(kind: str, setting: dict[str, Any]) -> LatencyDistributionSpec:
    if kind == "normal":
        return LatencyDistributionSpec(
            type="normal",
            mean=float(setting["mean"]),
            std=float(setting["std"]),
            min=float(setting["min"]),
            max=float(setting["max"]),
        )
    return LatencyDistributionSpec(
        type="uniform",
        mean=None,
        std=None,
        min=float(setting["min"]),
        max=float(setting["max"]),
    )


def _run_axis_fragment(sweep_name: str, frame_stack: int, obs_fps: int | float) -> str:
    if sweep_name == "frame_stack":
        return f"fs{int(frame_stack)}"
    return f"obs{_number_label(obs_fps)}"


def _run_name(
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
) -> str:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps)
    return f"{env_name}_{sweep_name}_{latency_label}_{axis}_seed{int(seed)}"


def _wandb_name(
    env_name: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
) -> str:
    axis = _run_axis_fragment(sweep_name, frame_stack, obs_fps)
    return f"{_env_slug(env_name)}-{axis}-{latency_label}-s{int(seed)}"


def _run_dir(
    generated_root: Path,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
) -> Path:
    return generated_root / sweep_name / latency_label / _run_axis_fragment(sweep_name, frame_stack, obs_fps) / f"seed_{int(seed)}"


def _output_dir(
    output_root: Path,
    stage: str,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
) -> Path:
    return (
        output_root
        / stage
        / sweep_name
        / latency_label
        / _run_axis_fragment(sweep_name, frame_stack, obs_fps)
        / f"seed_{int(seed)}"
    )


def _build_full_train_config(
    base_config: dict[str, Any],
    *,
    sweep_name: str,
    sweep_cfg: dict[str, Any],
    latency_label: str,
    latency_setting: dict[str, Any],
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
    train_dir: Path,
    output_root: Path,
    device: str,
    restart_behavior: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    env_name = _env_name(config)
    config.setdefault("experiment", {})["name"] = _run_name(
        env_name,
        sweep_name,
        latency_label,
        frame_stack,
        obs_fps,
        seed,
    )
    config["experiment"]["seed"] = int(seed)
    config.setdefault("backend", {})["type"] = "sample_factory"
    config["backend"]["device"] = str(device)
    config["backend"]["train_dir"] = str(train_dir)
    config["backend"]["restart_behavior"] = str(restart_behavior)
    config["backend"]["run_mode"] = "train"
    config.setdefault("env", {})["frame_stack"] = int(frame_stack)
    config["env"]["obs_fps"] = obs_fps
    _apply_latency_setting(config, latency_setting, int(seed))
    config.setdefault("logging", {})["output_dir"] = str(
        _output_dir(output_root, "train", sweep_name, latency_label, frame_stack, obs_fps, seed)
    )
    config["logging"]["wandb_group"] = str(sweep_cfg["wandb_group"])
    config["logging"]["wandb_name"] = _wandb_name(env_name, sweep_name, latency_label, frame_stack, obs_fps, seed)
    return config


def _build_latency_bench_eval_config(
    base_config: dict[str, Any],
    full_train_config: dict[str, Any],
    *,
    sweep_name: str,
    latency_label: str,
    frame_stack: int,
    obs_fps: int | float,
    seed: int,
    output_root: Path,
    latency_bench_policy_device: str,
) -> dict[str, Any]:
    config = merge_dicts(base_config, full_train_config)
    env_name = _env_name(config)
    config.setdefault("backend", {})["run_mode"] = "eval"
    _drop_eval_time_limit_fields(config)
    _align_demon_attack_raw_eval_config(config, env_name)
    config["experiment"]["num_episodes"] = int(config["evaluation"]["eval_episodes"])
    config["policy"] = merge_dicts(
        base_config.get("policy", {}),
        {
            "type": LATENCY_BENCH_POLICY_TYPES[env_name],
            "checkpoint_path": "",
            "device": str(latency_bench_policy_device),
            "deterministic": True,
        },
    )
    config["logging"]["output_dir"] = str(
        _output_dir(output_root, "latency_bench", sweep_name, latency_label, frame_stack, obs_fps, seed)
    )
    config["logging"].pop("wandb_name", None)
    if env_name == "deadly_corridor":
        config.setdefault("env", {})["noop_action"] = [0, 0, 0, 0]
    return config


def _iter_runs(spec: dict[str, Any], *, seeds: Sequence[int] | None = None):
    effective_seeds = [int(seed) for seed in (spec["seeds"] if seeds is None else seeds)]
    latency_settings = spec["latency_settings"]
    for sweep_name, sweep_cfg in spec["sweeps"].items():
        for latency_label, latency_setting in latency_settings.items():
            for frame_stack in sweep_cfg["frame_stack"]:
                for obs_fps in sweep_cfg["obs_fps"]:
                    for seed in effective_seeds:
                        yield (
                            sweep_name,
                            sweep_cfg,
                            latency_label,
                            latency_setting,
                            int(frame_stack),
                            float(obs_fps),
                            int(seed),
                        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Flappy frame_stack and obs_fps factor sweep configs.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--restart-behavior", type=str)
    parser.add_argument("--latency-bench-policy-device", type=str, default="auto")
    args = parser.parse_args(argv)

    spec_path = args.spec.expanduser()
    if not spec_path.is_absolute():
        spec_path = (REPO_ROOT / spec_path).resolve()
    spec = _load_yaml(spec_path)
    base_config_path = _resolve_path(spec["base_config"], base_dir=REPO_ROOT)
    base_config = load_config(base_config_path)
    generated_root = Path(args.generated_root or spec["generated_root"]).expanduser()
    output_root = Path(args.output_root or spec["output_root"])
    train_dir = Path(args.train_dir or spec["train_dir"])
    device = str(args.device or spec.get("device", "gpu"))
    restart_behavior = str(args.restart_behavior or spec.get("restart_behavior", "resume"))
    seeds = None if args.seeds is None else _parse_int_csv(args.seeds)

    manifest_rows = ["\t".join(MANIFEST_COLUMNS)]
    train_commands: list[str] = []
    latency_bench_eval_commands: list[str] = []

    for sweep_name, sweep_cfg, latency_label, latency_setting, frame_stack, obs_fps, seed in _iter_runs(
        spec,
        seeds=seeds,
    ):
        full_train_config = _build_full_train_config(
            base_config,
            sweep_name=sweep_name,
            sweep_cfg=sweep_cfg,
            latency_label=latency_label,
            latency_setting=latency_setting,
            frame_stack=frame_stack,
            obs_fps=obs_fps,
            seed=seed,
            train_dir=train_dir,
            output_root=output_root,
            device=device,
            restart_behavior=restart_behavior,
        )
        train_config = _prune_sample_factory_config(full_train_config)
        latency_bench_eval_config = _build_latency_bench_eval_config(
            base_config,
            full_train_config,
            sweep_name=sweep_name,
            latency_label=latency_label,
            frame_stack=frame_stack,
            obs_fps=obs_fps,
            seed=seed,
            output_root=output_root,
            latency_bench_policy_device=args.latency_bench_policy_device,
        )

        run_dir = _run_dir(generated_root, sweep_name, latency_label, frame_stack, obs_fps, seed)
        train_path = run_dir / "train.yaml"
        latency_bench_eval_path = run_dir / "latency_bench_eval.yaml"
        _write_yaml(train_path, train_config)
        _write_yaml(latency_bench_eval_path, latency_bench_eval_config)

        train_cmd = f"python -m training.train --config {shlex.quote(str(train_path))}"
        latency_bench_eval_cmd = f"python -m latency_bench.run --config {shlex.quote(str(latency_bench_eval_path))}"
        train_commands.append(train_cmd)
        latency_bench_eval_commands.append(latency_bench_eval_cmd)
        manifest_rows.append(
            "\t".join(
                [
                    sweep_name,
                    latency_label,
                    str(frame_stack),
                    str(obs_fps),
                    str(seed),
                    str(train_path),
                    str(latency_bench_eval_path),
                    train_cmd,
                    latency_bench_eval_cmd,
                ]
            )
        )

    generated_root.mkdir(parents=True, exist_ok=True)
    (generated_root / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (generated_root / "train_commands.txt").write_text("\n".join(train_commands) + "\n", encoding="utf-8")
    (generated_root / "latency_bench_eval_commands.txt").write_text(
        "\n".join(latency_bench_eval_commands) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote sweep configs to {generated_root}")
    print(f"Train commands: {generated_root / 'train_commands.txt'}")
    print(f"Latency bench eval commands: {generated_root / 'latency_bench_eval_commands.txt'}")
    print(f"Manifest: {generated_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
