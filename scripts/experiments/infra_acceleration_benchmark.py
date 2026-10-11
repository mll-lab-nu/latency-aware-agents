from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.config import load_config
from latency_bench.eval.suites import has_eval_suites
from latency_bench.eval.sweeps import run_eval_suites_from_config, run_eval_sweep_from_config
from training.common.checkpoints import find_checkpoint
from training.common.config_adapter import import_integration_module
from training.common.latency_bench_periodic_eval import build_latency_bench_config_from_training_config
from latency_bench.utils.io import write_csv
from latency_bench.utils.proc import cpu_seconds


BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
}

ENV_TITLES = {
    "flappy": "Flappy Bird",
    "demon_attack": "Demon Attack",
}

PARITY_TESTS = {
    "flappy": ["tests/training/gpu/infra_acceleration/test_flappy.py"],
    "demon_attack": ["tests/training/gpu/infra_acceleration/test_demon_attack.py"],
}


@dataclass(frozen=True)
class Budget:
    train_for_env_steps: int | None
    eval_episodes: int
    repeats: int
    target_train_seconds: float | None = None
    calibration_env_steps: int = 16_384


BUDGETS = {
    "smoke": Budget(train_for_env_steps=4096, eval_episodes=2, repeats=1),
    "short-stable": Budget(
        train_for_env_steps=None,
        eval_episodes=8,
        repeats=3,
        target_train_seconds=45.0,
        calibration_env_steps=16_384,
    ),
    "paper-grade": Budget(
        train_for_env_steps=None,
        eval_episodes=20,
        repeats=3,
        target_train_seconds=180.0,
        calibration_env_steps=16_384,
    ),
}


@dataclass(frozen=True)
class ConfigPair:
    baseline: dict[str, Any]
    accelerated: dict[str, Any]


def _csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_config_pair(
    base_config: dict[str, Any],
    *,
    env_name: str,
    output_root: Path,
    budget: Budget,
    repeat_index: int,
) -> ConfigPair:
    baseline = _stage_config(
        base_config,
        env_name=env_name,
        variant="baseline",
        output_root=output_root,
        budget=budget,
        repeat_index=repeat_index,
    )
    accelerated = _stage_config(
        base_config,
        env_name=env_name,
        variant="accelerated",
        output_root=output_root,
        budget=budget,
        repeat_index=repeat_index,
    )
    _apply_backend_variant(baseline, env_name=env_name, variant="baseline")
    _apply_backend_variant(accelerated, env_name=env_name, variant="accelerated")
    return ConfigPair(baseline=baseline, accelerated=accelerated)


def calibrate_train_budget(
    base_config: dict[str, Any],
    *,
    env_name: str,
    output_root: Path,
    budget: Budget,
) -> tuple[Budget, dict[str, Any]]:
    calibration_budget = Budget(
        train_for_env_steps=budget.calibration_env_steps,
        eval_episodes=budget.eval_episodes,
        repeats=1,
        target_train_seconds=budget.target_train_seconds,
        calibration_env_steps=budget.calibration_env_steps,
    )
    calibration_config = _stage_config(
        base_config,
        env_name=env_name,
        variant="baseline_calibration",
        output_root=output_root,
        budget=calibration_budget,
        repeat_index=0,
    )
    _apply_backend_variant(calibration_config, env_name=env_name, variant="baseline")
    result = run_training(calibration_config)
    if int(result["status"]) != 0:
        raise_training_failed(calibration_config, result)
    calibration_fps = float(result["env_steps"]) / float(result["wall_sec"])
    resolved_steps = int(round(calibration_fps * float(budget.target_train_seconds)))
    resolved_budget = Budget(
        train_for_env_steps=resolved_steps,
        eval_episodes=budget.eval_episodes,
        repeats=budget.repeats,
        target_train_seconds=budget.target_train_seconds,
        calibration_env_steps=budget.calibration_env_steps,
    )
    row = {
        "env_name": env_name,
        "env_title": ENV_TITLES[env_name],
        "phase": "calibration",
        "variant": "baseline",
        "backend_label": backend_label(calibration_config, phase="train"),
        "status": int(result["status"]),
        "wall_sec": float(result["wall_sec"]),
        "cpu_sec": float(result["cpu_sec"]),
        "calibration_env_steps": budget.calibration_env_steps,
        "calibration_fps": calibration_fps,
        "target_train_seconds": float(budget.target_train_seconds),
        "resolved_train_for_env_steps": resolved_steps,
        "output_dir": calibration_config["logging"]["output_dir"],
    }
    return resolved_budget, row


def _stage_config(
    base_config: dict[str, Any],
    *,
    env_name: str,
    variant: str,
    output_root: Path,
    budget: Budget,
    repeat_index: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    run_name = f"infra_accel_{env_name}_{variant}_r{int(repeat_index)}"
    config["experiment"]["name"] = run_name
    config["backend"]["run_mode"] = "train"
    config["backend"]["restart_behavior"] = "overwrite"
    config["backend"]["train_dir"] = str(output_root / "checkpoints")
    config["training"]["train_for_env_steps"] = int(budget.train_for_env_steps)
    config["evaluation"]["eval_interval_steps"] = None
    config["evaluation"]["episodes"] = int(budget.eval_episodes)
    config["logging"]["output_dir"] = str(output_root / "runs" / env_name / variant / f"repeat_{repeat_index}")
    config["logging"]["wandb_project"] = ""
    config["logging"]["wandb_tags"] = ""
    if "video" in config["logging"]:
        config["logging"]["video"]["enabled"] = False
    config["logging"]["save_step_records"] = False
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False
    extra_args = list(config["backend"]["extra_args"]) if "extra_args" in config["backend"] else []
    extra_args.extend(["--decorrelate_envs_on_one_worker", "False", "--heartbeat_reporting_interval", "900"])
    config["backend"]["extra_args"] = extra_args
    return config


def _apply_backend_variant(config: dict[str, Any], *, env_name: str, variant: str) -> None:
    env_cfg = config["env"]
    if env_name == "flappy":
        env_cfg["use_gpu_render"] = True
        env_cfg["gpu_render_device"] = "cuda"
        config["backend"]["device"] = "cuda"
        if variant == "baseline":
            env_cfg["simulator"] = "cpu"
            return
        env_cfg["simulator"] = "gpu"
        return
    if env_name == "demon_attack":
        config["backend"]["device"] = "gpu"
        if variant == "baseline":
            env_cfg["simulator"] = "cpu"
            training_cfg = config["training"]
            training_cfg["num_workers"] = 16
            training_cfg["num_envs_per_worker"] = 4
            training_cfg["worker_num_splits"] = 2
            training_cfg["rollout"] = 128
            training_cfg["num_batches_per_epoch"] = 2
            return
        env_cfg["simulator"] = "gpu"
        return
    raise KeyError(env_name)


def backend_label(config: dict[str, Any], *, phase: str) -> str:
    env_name = config["env"]["name"]
    simulator = config["env"]["simulator"]
    if phase == "train":
        if env_name == "flappy" and bool(config["env"]["use_gpu_render"]) and simulator == "cpu":
            return "torch_gpu_render_cpu_core"
        if env_name == "flappy" and bool(config["env"]["use_gpu_render"]) and simulator == "gpu":
            return "cuda_core_render"
        if env_name == "demon_attack" and simulator == "gpu":
            return "cule_cuda"
        if env_name == "demon_attack" and str(config["backend"]["device"]) == "gpu" and simulator == "cpu":
            return "gpu_policy_cpu_env"
        return f"{simulator}_reference"
    if env_name == "flappy" and bool(config["env"]["use_gpu_render"]) and simulator == "gpu":
        return "flappy_gpu_batched"
    return "latency_bench_eval_adapter"


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(config["logging"]["output_dir"]) / "train_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--run-training-config", str(config_path)]
    before_cpu = cpu_seconds()
    before_wall = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT)
    wall_sec = time.perf_counter() - before_wall
    cpu_sec = cpu_seconds() - before_cpu
    status = int(result.returncode)
    env_steps = int(config["training"]["train_for_env_steps"])
    if status == 0:
        env_steps = read_sample_factory_collected_env_steps(config)
    return {
        "status": status,
        "wall_sec": wall_sec,
        "cpu_sec": cpu_sec,
        "env_steps": env_steps,
    }


def run_training_config_once(config: dict[str, Any]) -> int:
    module = import_integration_module(config)
    with contextlib.redirect_stdout(sys.stderr):
        return int(module.run_from_config(config))


def read_sample_factory_collected_env_steps(config: dict[str, Any]) -> int:
    checkpoint_path = find_checkpoint(
        config["backend"]["train_dir"],
        config["experiment"]["name"],
        kind="latest",
    )
    return int(checkpoint_path.stem.rsplit("_", 1)[1])


def raise_training_failed(config: dict[str, Any], result: dict[str, Any]) -> None:
    raise RuntimeError(
        "training failed: "
        f"env={config['env']['name']} "
        f"experiment={config['experiment']['name']} "
        f"status={int(result['status'])} "
        f"output_dir={config['logging']['output_dir']}"
    )


def raise_eval_failed(config: dict[str, Any], result: dict[str, Any], *, output_dir: Path) -> None:
    raise RuntimeError(
        "eval failed: "
        f"env={config['env']['name']} "
        f"experiment={config['experiment']['name']} "
        f"status={int(result['status'])} "
        f"output_dir={output_dir}"
    )


def run_eval(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    checkpoint_path = find_checkpoint(
        config["backend"]["train_dir"],
        config["experiment"]["name"],
        kind="latest",
    )
    eval_config = build_latency_bench_config_from_training_config(
        config,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "eval_config.json"
    result_path = output_dir / "eval_result.json"
    config_path.write_text(json.dumps(eval_config, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-eval-config",
        str(config_path),
        "--eval-result-path",
        str(result_path),
    ]
    before_cpu = cpu_seconds()
    before_wall = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT)
    wall_sec = time.perf_counter() - before_wall
    cpu_sec = cpu_seconds() - before_cpu
    status = int(result.returncode)
    if status != 0:
        return {
            "status": status,
            "wall_sec": wall_sec,
            "cpu_sec": cpu_sec,
        }
    eval_result = json.loads(result_path.read_text(encoding="utf-8"))
    eval_result["status"] = status
    eval_result["wall_sec"] = wall_sec
    eval_result["cpu_sec"] = cpu_sec
    return eval_result


def run_eval_config_once(eval_config: dict[str, Any], *, result_path: Path) -> int:
    with contextlib.redirect_stdout(sys.stderr):
        if has_eval_suites(eval_config):
            summaries = run_eval_suites_from_config(eval_config, progress=True)
        else:
            metrics = run_eval_sweep_from_config(eval_config, progress=True)
            summaries = [
                {
                    "returns": [float(metric.episode_return) for metric in metrics],
                    "lengths": [int(metric.survival_steps) for metric in metrics],
                    "mean_return": sum(float(metric.episode_return) for metric in metrics) / len(metrics),
                    "mean_length": sum(int(metric.survival_steps) for metric in metrics) / len(metrics),
                }
            ]
    episodes = sum(len(summary["returns"]) for summary in summaries)
    episode_steps = sum(sum(int(length) for length in summary["lengths"]) for summary in summaries)
    mean_return = _weighted_mean(summaries, "mean_return", "returns")
    mean_length = _weighted_mean(summaries, "mean_length", "lengths")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "episodes": episodes,
                "episode_steps": episode_steps,
                "mean_return": mean_return,
                "mean_length": mean_length,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _weighted_mean(summaries: Sequence[dict[str, Any]], value_key: str, count_key: str) -> float:
    total_count = sum(len(summary[count_key]) for summary in summaries)
    return sum(float(summary[value_key]) * len(summary[count_key]) for summary in summaries) / float(total_count)


def raw_train_row(config: dict[str, Any], *, variant: str, repeat_index: int, result: dict[str, Any]) -> dict[str, Any]:
    wall_sec = float(result["wall_sec"])
    env_steps = int(result["env_steps"])
    return {
        "env_name": config["env"]["name"],
        "env_title": ENV_TITLES[config["env"]["name"]],
        "phase": "train",
        "variant": variant,
        "repeat": int(repeat_index),
        "backend_label": backend_label(config, phase="train"),
        "status": int(result["status"]),
        "wall_sec": wall_sec,
        "cpu_sec": float(result["cpu_sec"]),
        "cpu_util": float(result["cpu_sec"]) / wall_sec,
        "env_steps": env_steps,
        "fps": env_steps / wall_sec,
        "episodes": "",
        "episode_steps": "",
        "episodes_per_sec": "",
        "episode_steps_per_sec": "",
        "mean_return": "",
        "mean_length": "",
        "output_dir": config["logging"]["output_dir"],
    }


def raw_eval_row(config: dict[str, Any], *, variant: str, repeat_index: int, result: dict[str, Any]) -> dict[str, Any]:
    wall_sec = float(result["wall_sec"])
    episodes = int(result["episodes"])
    episode_steps = int(result["episode_steps"])
    return {
        "env_name": config["env"]["name"],
        "env_title": ENV_TITLES[config["env"]["name"]],
        "phase": "eval",
        "variant": variant,
        "repeat": int(repeat_index),
        "backend_label": backend_label(config, phase="eval"),
        "status": int(result["status"]),
        "wall_sec": wall_sec,
        "cpu_sec": float(result["cpu_sec"]),
        "cpu_util": float(result["cpu_sec"]) / wall_sec,
        "env_steps": "",
        "fps": "",
        "episodes": episodes,
        "episode_steps": episode_steps,
        "episodes_per_sec": episodes / wall_sec,
        "episode_steps_per_sec": episode_steps / wall_sec,
        "mean_return": float(result["mean_return"]),
        "mean_length": float(result["mean_length"]),
        "output_dir": config["logging"]["output_dir"],
    }


def summarize_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(str(row["env_name"]), str(row["phase"]))][str(row["variant"])].append(dict(row))

    summaries = []
    for (env_name, phase), by_variant in sorted(grouped.items()):
        baseline = by_variant["baseline"]
        accelerated = by_variant["accelerated"]
        baseline_wall = _mean_float(row["wall_sec"] for row in baseline)
        accelerated_wall = _mean_float(row["wall_sec"] for row in accelerated)
        row = {
            "env_name": env_name,
            "env_title": ENV_TITLES[env_name],
            "phase": phase,
            "baseline_backend": baseline[0]["backend_label"],
            "accelerated_backend": accelerated[0]["backend_label"],
            "baseline_wall_sec": baseline_wall,
            "accelerated_wall_sec": accelerated_wall,
            "speedup": baseline_wall / accelerated_wall,
            "time_reduction_pct": (1.0 - accelerated_wall / baseline_wall) * 100.0,
            "eval_backend_changed": int(baseline[0]["backend_label"] != accelerated[0]["backend_label"]) if phase == "eval" else "",
        }
        if phase == "train":
            baseline_steps = sum(float(item["env_steps"]) for item in baseline)
            accelerated_steps = sum(float(item["env_steps"]) for item in accelerated)
            row["baseline_fps"] = baseline_steps / sum(float(item["wall_sec"]) for item in baseline)
            row["accelerated_fps"] = accelerated_steps / sum(float(item["wall_sec"]) for item in accelerated)
            row["baseline_episodes_per_sec"] = ""
            row["accelerated_episodes_per_sec"] = ""
        else:
            baseline_episodes = sum(float(item["episodes"]) for item in baseline)
            accelerated_episodes = sum(float(item["episodes"]) for item in accelerated)
            row["baseline_fps"] = ""
            row["accelerated_fps"] = ""
            row["baseline_episodes_per_sec"] = baseline_episodes / sum(float(item["wall_sec"]) for item in baseline)
            row["accelerated_episodes_per_sec"] = accelerated_episodes / sum(float(item["wall_sec"]) for item in accelerated)
        summaries.append(row)
    return summaries


def _mean_float(values: Iterable[Any]) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / float(len(parsed))


def run_parity(envs: Sequence[str], *, output_root: Path, timeout_sec: float) -> list[dict[str, Any]]:
    rows = []
    for env_name in envs:
        before = time.perf_counter()
        command = [sys.executable, "-m", "pytest", "-q", *PARITY_TESTS[env_name]]
        try:
            result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout_sec)
            stdout = result.stdout
            stderr = result.stderr
            status = int(result.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if exc.stdout is not None else ""
            stderr = exc.stderr if exc.stderr is not None else ""
            status = 124
        out_dir = output_root / "parity_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{env_name}.stdout.txt").write_text(stdout, encoding="utf-8")
        (out_dir / f"{env_name}.stderr.txt").write_text(stderr, encoding="utf-8")
        passed = status == 0
        skipped = " skipped" in stdout or " skipped" in stderr
        parity_status = "timeout" if status == 124 else "passed" if passed and not skipped else "passed_with_skips" if passed else "failed"
        rows.append(
            {
                "env_name": env_name,
                "env_title": ENV_TITLES[env_name],
                "parity_source": "pytest_semantic_trace_tests",
                "obs_equal": int(passed),
                "reward_equal": int(passed),
                "terminated_equal": int(passed),
                "truncated_equal": int(passed),
                "info_equal": int(passed),
                "behavior_changed": int(not passed),
                "proof_complete": int(passed and not skipped),
                "parity_status": parity_status,
                "status": status,
                "wall_sec": time.perf_counter() - before,
                "timeout_sec": timeout_sec,
                "log_stdout": str(out_dir / f"{env_name}.stdout.txt"),
                "log_stderr": str(out_dir / f"{env_name}.stderr.txt"),
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark infrastructure acceleration for small-model paper figures.")
    parser.add_argument("--run-training-config", type=Path)
    parser.add_argument("--run-eval-config", type=Path)
    parser.add_argument("--eval-result-path", type=Path)
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="paper-grade")
    parser.add_argument("--envs", default="flappy,demon_attack")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--train-for-env-steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "paper" / "latency-sensitive-vla" / "figures" / "3_5_infra_acceleration" / "small_model" / "data",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--parity-timeout-sec", type=float, default=600.0)
    parser.add_argument("--train-cooldown-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_training_config is not None:
        return run_training_config_once(json.loads(args.run_training_config.read_text(encoding="utf-8")))
    if args.run_eval_config is not None:
        return run_eval_config_once(
            json.loads(args.run_eval_config.read_text(encoding="utf-8")),
            result_path=args.eval_result_path,
        )

    base_budget = BUDGETS[args.budget]
    budget = Budget(
        train_for_env_steps=args.train_for_env_steps
        if args.train_for_env_steps is not None
        else base_budget.train_for_env_steps,
        eval_episodes=int(args.eval_episodes if args.eval_episodes is not None else base_budget.eval_episodes),
        repeats=int(args.repeats if args.repeats is not None else base_budget.repeats),
        target_train_seconds=None if args.train_for_env_steps is not None else base_budget.target_train_seconds,
        calibration_env_steps=base_budget.calibration_env_steps,
    )
    envs = _csv_items(args.envs)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    staged_configs: list[dict[str, Any]] = []
    resolved_train_steps: dict[str, int] = {}

    for env_name in envs:
        base_config = load_config(BASE_CONFIGS[env_name])
        if budget.train_for_env_steps is None:
            if args.dry_run:
                env_budget = Budget(
                    train_for_env_steps=budget.calibration_env_steps,
                    eval_episodes=budget.eval_episodes,
                    repeats=budget.repeats,
                    target_train_seconds=budget.target_train_seconds,
                    calibration_env_steps=budget.calibration_env_steps,
                )
            else:
                env_budget, calibration_row = calibrate_train_budget(
                    base_config,
                    env_name=env_name,
                    output_root=output_root,
                    budget=budget,
                )
                calibration_rows.append(calibration_row)
        else:
            env_budget = budget
        resolved_train_steps[env_name] = env_budget.train_for_env_steps
        for repeat_index in range(env_budget.repeats):
            pair = build_config_pair(
                base_config,
                env_name=env_name,
                output_root=output_root,
                budget=env_budget,
                repeat_index=repeat_index,
            )
            for variant, config in (("baseline", pair.baseline), ("accelerated", pair.accelerated)):
                staged_configs.append(config)
                if args.dry_run:
                    continue
                if not args.skip_train:
                    train_result = run_training(config)
                    if int(train_result["status"]) != 0:
                        raise_training_failed(config, train_result)
                    raw_rows.append(raw_train_row(config, variant=variant, repeat_index=repeat_index, result=train_result))
                    if args.train_cooldown_sec:
                        time.sleep(float(args.train_cooldown_sec))
                if not args.skip_eval:
                    eval_output = output_root / "runs" / env_name / variant / f"repeat_{repeat_index}" / "eval"
                    eval_result = run_eval(config, output_dir=eval_output)
                    if int(eval_result["status"]) != 0:
                        raise_eval_failed(config, eval_result, output_dir=eval_output)
                    raw_rows.append(raw_eval_row(config, variant=variant, repeat_index=repeat_index, result=eval_result))

    if args.dry_run:
        write_jsonl(output_root / "staged_configs.jsonl", staged_configs)
        return 0

    write_jsonl(output_root / "raw_results.jsonl", raw_rows)
    write_jsonl(output_root / "calibration_results.jsonl", calibration_rows)
    write_csv(output_root / "calibration_results.csv", calibration_rows)
    summary_rows = summarize_results(raw_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    parity_rows = [] if args.skip_parity else run_parity(envs, output_root=output_root, timeout_sec=args.parity_timeout_sec)
    write_csv(output_root / "parity_summary.csv", parity_rows)
    metadata = {
        "budget": args.budget,
        "target_train_seconds": budget.target_train_seconds,
        "calibration_env_steps": budget.calibration_env_steps,
        "resolved_train_for_env_steps": resolved_train_steps,
        "eval_episodes": budget.eval_episodes,
        "repeats": budget.repeats,
        "parity_timeout_sec": args.parity_timeout_sec,
        "train_cooldown_sec": float(args.train_cooldown_sec),
        "envs": envs,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
