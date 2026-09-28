"""Run one MIKASA H8 DAgger arm with resident two-rank inference."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import yaml

from latency_bench.data.starvla_tasks import read_task_contract
from scripts.starvla.dagger_mixture import write_dagger_mixture
from scripts.starvla.train_task import inference_bundle
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES


ROOT = Path(__file__).resolve().parents[3]
STARVLA = ROOT / "third_party/starVLA"
PACKAGER_PROJECT = ROOT / "third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot"
DEVELOPMENT_SEED = 4_242_424_442
HELDOUT_SEED = 4_242_424_542


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(command: list[str | Path], log: Path, *, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(str(value) for value in command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(value) for value in command], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def _rename_partial(path: Path) -> None:
    if path.exists():
        path.rename(path.with_name(f"{path.name}.interrupted-{int(time.time())}"))


def _rollout(args, mode: str, destination: Path, *, seed: int, episodes: int,
             checkpoint: Path, resident_step: int | None = None,
             eval_config: Path | None = None) -> None:
    if mode == "eval" and (destination / "result.json").is_file():
        return
    if mode == "collect" and (destination / "collection.json").is_file():
        return
    _rename_partial(destination)
    command = [
        args.task_python, "-m",
        "scripts.tasks.mikasa_intercept_grab_fast.dagger_rollout", mode,
        "--model", args.model, "--condition", args.condition,
        "--eval-config", eval_config or args.eval_config, "--source-run", args.source_run,
        "--latency-config", args.latency_config,
        "--bootstrap", args.bootstrap, "--checkpoint", checkpoint,
        "--teacher-checkpoint", args.teacher_checkpoint,
        "--output", destination, "--seed", seed,
        "--episodes", episodes, "--head-budget", args.head_budget,
        "--parallel-envs", args.collect_envs if mode == "collect" else args.eval_envs,
        "--simulator-gpu", args.simulator_gpu,
        "--inference-gpu", args.inference_gpu,
        "--inference-batch-size", args.inference_batch_size,
        "--num-gpus", 2,
    ]
    if resident_step is not None:
        command += ["--resident-control-dir", args.output / "dagger_control",
                    "--resident-step", resident_step]
    with args.gpu_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run(command, destination.with_suffix(".log"))


def _eval(args, step: int, checkpoint: Path, *, resident: bool) -> dict:
    destination = args.output / "evaluations" / f"step_{step:04d}"
    if args.condition in FIXED_LATENCIES:
        latencies = {}
        for latency in range(5):
            cell = destination / f"eval_{latency}"
            _rollout(args, "eval", cell, seed=DEVELOPMENT_SEED,
                     episodes=args.eval_episodes, checkpoint=checkpoint,
                     resident_step=step if resident else None,
                     eval_config=args.eval_configs[latency])
            latencies[str(latency)] = json.loads((cell / "result.json").read_text())
        result = {**latencies[str(FIXED_LATENCIES[args.condition])], "latencies": latencies}
        # A checkpoint becomes a selection candidate after all five cells finish.
        _write_json(destination / "result.json", result)
    else:
        _rollout(args, "eval", destination, seed=DEVELOPMENT_SEED,
                 episodes=args.eval_episodes, checkpoint=checkpoint,
                 resident_step=step if resident else None)
        result = json.loads((destination / "result.json").read_text())
    _update_best(args)
    return result


def _update_best(args) -> None:
    records = []
    for result_file in (args.output / "evaluations").glob("step_*/result.json"):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        records.append((result["success_count"], result["mean_return"],
                        int(result_file.parent.name.removeprefix("step_"))))
    best_step = max(records)[2]
    marker = args.output / "best.json"
    if marker.is_file() and json.loads(marker.read_text())["step"] == best_step:
        return
    bundle = args.output / "bundles/best"
    if best_step:
        if bundle.exists():
            shutil.rmtree(bundle)
        source = args.output / "training/dagger/checkpoints" / f"steps_{best_step}_pytorch_model.pt"
        source_run = json.loads((args.source_run / "run.json").read_text(encoding="utf-8"))
        inference_bundle(args.output / "training/dagger", source, bundle,
                         bootstrap=args.bootstrap,
                         backbone=source_run["backbone"])
        best_checkpoint = bundle / "checkpoints/model.pt"
    else:
        best_checkpoint = args.source_policy_checkpoint
    _write_json(marker, {"step": best_step, "checkpoint": str(best_checkpoint)})


def _delta(args, round_number: int, checkpoint: Path,
           *, resident_step: int | None) -> dict:
    directory = args.output / "deltas" / f"round-{round_number:02d}"
    raw = directory / "raw"
    _rollout(args, "collect", raw,
             seed=10_000_000 + round_number * 1_000_000,
             episodes=0, checkpoint=checkpoint, resident_step=resident_step)
    lerobot = directory / "lerobot"
    if not (lerobot / "task_contract.json").is_file():
        if lerobot.exists():
            shutil.rmtree(lerobot)
        command = [
            args.uv, "run", "--frozen", "--project", PACKAGER_PROJECT, "python",
            "-m", "scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset",
            "--raw-root", raw, "--output-train", lerobot,
            "--normalization-source", args.bootstrap / "lerobot",
            "--task", args.task,
        ]
        package_env = os.environ | {"PYTHONPATH": str(ROOT)}
        _run(command, directory / "package.log", env=package_env)
    result = json.loads((raw / "collection.json").read_text(encoding="utf-8"))
    heads = result["splits"]["train"]["valid_actions"]
    if heads < args.head_budget:
        raise RuntimeError(f"round {round_number} collected only {heads} valid heads")
    original = read_task_contract(args.bootstrap / "lerobot")
    delta_contract = read_task_contract(lerobot)
    if delta_contract["normalization"] != original["normalization"]:
        raise RuntimeError(f"round {round_number} changed bootstrap normalization")
    return result


def _mixture(args, count: int) -> None:
    deltas = [args.output / "deltas" / f"round-{index:02d}"
              for index in range(1, count + 1)]
    robot_type = read_task_contract(args.bootstrap / "lerobot")["robot_type"]
    write_dagger_mixture(args.bootstrap, deltas, robot_type,
                         args.output / "dagger_mixture.json")


def _checkpoint(args, step: int) -> tuple[Path, Path]:
    base = args.output / "training/dagger/checkpoints" / f"steps_{step}"
    return base.with_name(base.name + "_pytorch_model.pt"), base.with_name(base.name + "_state")


def _complete_checkpoint(args, step: int) -> bool:
    weights, state = _checkpoint(args, step)
    files = [state / "latest", state / "pytorch_model/mp_rank_00_model_states.pt"]
    files.extend(state / f"random_states_{rank}.pkl" for rank in range(2))
    files.extend(state / "pytorch_model" /
                 f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                 for rank in range(2))
    return weights.is_file() and all(path.is_file() and path.stat().st_size > 0
                                     for path in files)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False


def _trainer_config(args, resume_state: Path | None) -> Path:
    config = yaml.safe_load((args.source_run / "bootstrap.yaml").read_text(encoding="utf-8"))
    config["run_id"] = "dagger"
    config["run_root_dir"] = str(args.output / "training")
    config["wandb_name"] = f"mikasa-h8-dagger-{args.condition}-{args.model}"
    config["wandb_group"] = args.output.name
    config["wandb_tags"] = ["mikasa", "naive-h8" if args.condition == "zero-latency"
                            else "conditioned-h8", "dagger", args.condition, args.model]
    data = config["datasets"]["vla_data"]
    data["task_contract_path"] = str(args.bootstrap / "lerobot/task_contract.json")
    data["custom_mixtures_path"] = str(args.output / "dagger_mixture.json")
    trainer = config["trainer"]
    if args.condition == "zero-latency":
        data["per_device_batch_size"] = 64 if args.model == "qwenpi_v3" else 32
        trainer["gradient_accumulation_steps"] = 1 if args.model == "qwenpi_v3" else 2
    trainer["max_train_steps"] = args.total_steps
    trainer["stop_after_steps"] = args.total_steps
    trainer["save_interval"] = args.round_steps
    trainer["eval_interval"] = args.total_steps
    trainer["num_warmup_steps"] = 0 if args.smoke else 100
    trainer["pretrained_checkpoint"] = str(resume_state or args.source_checkpoint)
    trainer["is_resume"] = resume_state is not None
    trainer["dagger_control_dir"] = str(args.output / "dagger_control")
    path = args.output / "dagger.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _launch_trainer(args) -> int:
    control = args.output / "dagger_control"
    existing = control / "trainer.json"
    if existing.is_file():
        pid = json.loads(existing.read_text(encoding="utf-8"))["pid"]
        if _alive(pid):
            return pid
    states = [step for step in range(args.round_steps, args.total_steps + 1, args.round_steps)
              if _complete_checkpoint(args, step)]
    current_step = states[-1] if states else 0
    if current_step >= args.total_steps:
        return 0
    resume_state = _checkpoint(args, current_step)[1] if current_step else None
    partial = args.output / "training/dagger"
    if current_step == 0 and partial.exists():
        _rename_partial(partial)
    if current_step:
        (control / f"step_{current_step}.ready").unlink(missing_ok=True)
        for rank in range(2):
            (control / f"step_{current_step}.rank_{rank}.ready").unlink(missing_ok=True)
            (control / f"step_{current_step}.rank_{rank}.sock").unlink(missing_ok=True)
    config_path = _trainer_config(args, resume_state)
    command = [
        args.vla_python, "-m", "accelerate.commands.launch",
        "--config_file", STARVLA / "starVLA/config/deepseeds/deepspeed_zero2.yaml",
        "--num_processes", "2", "--main_process_port", args.port,
        STARVLA / "starVLA/training/train_starvla.py", "--config_yaml", config_path,
    ]
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.train_devices,
        "CUDA_HOME": str(args.vla_python.parent.parent),
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(STARVLA))),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled" if args.smoke else "online",
        "WANDB_RUN_ID": f"mikasa-h8-dagger-{args.condition}-{args.model}-"
                        + json.loads((args.source_run / "run.json").read_text())["run_id"],
        "WANDB_RESUME": "allow",
    }
    env["PATH"] = os.pathsep.join((str(args.vla_python.parent),
                                   str(args.vla_python.parent.parent / "bin"), env["PATH"]))
    log = args.output / "dagger.log"
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen([str(value) for value in command], cwd=STARVLA,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    _write_json(existing, {"pid": process.pid, "command": [str(value) for value in command]})
    print(f"trainer_pid={process.pid} resumed_from={resume_state or args.source_checkpoint}", flush=True)
    return process.pid


def _wait_boundary(args, pid: int, step: int) -> None:
    control = args.output / "dagger_control"
    ready = control / f"step_{step}.ready"
    while True:
        complete = _complete_checkpoint(args, step)
        alive = pid != 0 and _alive(pid)
        if complete and (not alive if step == args.total_steps else ready.is_file()):
            return
        if not alive:
            raise RuntimeError(f"trainer exited before complete DAgger step {step}; see dagger.log")
        time.sleep(1)


def _heldout(args) -> None:
    best = json.loads((args.output / "best.json").read_text(encoding="utf-8"))
    if args.condition in FIXED_LATENCIES:
        results = {}
        for latency in range(5):
            destination = args.output / "heldout" / f"eval_{latency}"
            _rollout(args, "eval", destination, seed=4_242_424_242,
                     episodes=args.heldout_episodes, checkpoint=Path(best["checkpoint"]),
                     eval_config=args.eval_configs[latency])
            results[str(latency)] = json.loads((destination / "result.json").read_text())
        _write_json(args.output / "result.json", {
            "source_checkpoint": str(args.source_checkpoint),
            "dagger_steps": args.total_steps, "best_step": best["step"],
            "selection_latency": FIXED_LATENCIES[args.condition], "heldout": results,
        })
        return
    latest = _bundle(args, "latest", args.output / "training/dagger",
                     _checkpoint(args, args.total_steps)[0])
    checkpoints = {
        "baseline": args.source_policy_checkpoint,
        "best": Path(best["checkpoint"]),
        "latest": latest,
    }
    results = {}
    if args.condition == "zero-latency":
        settings = {
            "zero": args.eval_config,
            "profile": args.eval_config_profile,
            "fixed-2": args.eval_config_fixed,
        }
        for setting, config in settings.items():
            results[setting] = {}
            for name, checkpoint in checkpoints.items():
                destination = args.output / "heldout" / setting / name
                _rollout(args, "eval", destination, seed=HELDOUT_SEED,
                         episodes=args.heldout_episodes, checkpoint=checkpoint,
                         eval_config=config)
                results[setting][name] = json.loads(
                    (destination / "result.json").read_text(encoding="utf-8")
                )
    else:
        for name, checkpoint in checkpoints.items():
            destination = args.output / "heldout" / name
            _rollout(args, "eval", destination, seed=HELDOUT_SEED,
                     episodes=args.heldout_episodes, checkpoint=checkpoint)
            results[name] = json.loads((destination / "result.json").read_text(encoding="utf-8"))
    _write_json(args.output / "result.json", {
        "source_checkpoint": str(args.source_checkpoint),
        "dagger_steps": args.total_steps,
        "best_step": best["step"],
        "heldout": results,
    })


def _bundle(args, name: str, trained: Path, weights: Path) -> Path:
    destination = args.output / "bundles" / name
    checkpoint = destination / "checkpoints/model.pt"
    required = (checkpoint, destination / "config.yaml",
                destination / "dataset_statistics.json", destination / "task_contract.json")
    if not all(path.is_file() for path in required):
        if destination.exists():
            shutil.rmtree(destination)
        source_run = json.loads((args.source_run / "run.json").read_text(encoding="utf-8"))
        inference_bundle(trained, weights, destination,
                         bootstrap=args.bootstrap,
                         backbone=source_run["backbone"])
    return checkpoint


def run(args) -> None:
    args.task = "mikasa_h8" if args.condition == "zero-latency" else "mikasa_conditioned_h8"
    source_run = json.loads((args.source_run / "run.json").read_text())
    if (source_run["task"], source_run["model"], source_run["condition"]) != (
        args.task, args.model, args.condition
    ):
        raise ValueError("DAgger source must match the requested task, model and training condition")
    sft_step = source_run["training"]["sft_updates"]
    args.source_checkpoint = (
        args.source_run / f"training/bootstrap/checkpoints/steps_{sft_step}_pytorch_model.pt"
    )
    args.round_steps = 2 if args.smoke else 500
    args.rounds = 2 if args.smoke else 10
    args.total_steps = args.round_steps * args.rounds
    if args.smoke:
        args.head_budget, args.eval_episodes, args.heldout_episodes = 8, 2, 2
        args.collect_envs, args.eval_envs = 2, 2
    else:
        args.head_budget, args.eval_episodes, args.heldout_episodes = 16_000, 100, 200
        args.collect_envs, args.eval_envs = 128, 10
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "run.json"
    source = {"source_run": str(args.source_run), "source_run_id": source_run["run_id"],
              "source_checkpoint": str(args.source_checkpoint),
              "bootstrap": str(args.bootstrap), "model": args.model,
              "condition": args.condition, "train_devices": args.train_devices,
              "simulator_gpu": args.simulator_gpu, "rounds": args.rounds,
              "round_steps": args.round_steps, "head_budget": args.head_budget,
              "development_latency": FIXED_LATENCIES[args.condition] if args.condition in FIXED_LATENCIES else None}
    if args.condition == "zero-latency":
        source["port"] = args.port
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != source:
            raise RuntimeError("existing DAgger run has a different source or budget")
    else:
        _write_json(marker, source)
    if args.condition in FIXED_LATENCIES:
        args.eval_configs = {}
        config_dir = args.output / "eval_configs"
        config_dir.mkdir(exist_ok=True)
        for latency in range(5):
            config = yaml.safe_load(args.eval_config.read_text())
            config["latency"] = ({"method": "zero", "sync_cuda": False} if latency == 0 else
                                 {"method": "fixed", "fixed_latency_ms": latency * 50.0})
            path = config_dir / f"eval_{latency}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            args.eval_configs[latency] = path
    control = args.output / "dagger_control"
    control.mkdir(exist_ok=True)
    resident_config = control / "resident_policy.json"
    if not resident_config.is_file():
        socket_dir = Path("/tmp") / f"mikasa-h8-dagger-{uuid.uuid4().hex[:10]}"
        _write_json(resident_config, {"socket_dir": str(socket_dir), "policy": {
            "task_contract_path": str(args.bootstrap / "lerobot/task_contract.json"),
            "state_info_key": "mikasa_proprio",
            "image_views_info_key": "mikasa_image_views",
        }})
    Path(json.loads(resident_config.read_text())["socket_dir"]).mkdir(parents=True, exist_ok=True)

    args.source_policy_checkpoint = _bundle(
        args, "source", args.source_run / "training/bootstrap", args.source_checkpoint
    )
    _eval(args, 0, args.source_policy_checkpoint, resident=False)
    _delta(args, 1, args.source_policy_checkpoint, resident_step=None)
    _mixture(args, 1)
    pid = _launch_trainer(args)
    for step in range(args.round_steps, args.total_steps + 1, args.round_steps):
        if step < args.total_steps and (control / f"step_{step}.continue.json").is_file():
            continue
        _wait_boundary(args, pid, step)
        weights = _checkpoint(args, step)[0]
        eval_checkpoint = (weights if step < args.total_steps else
                           _bundle(args, "latest", args.output / "training/dagger", weights))
        result = _eval(args, step, eval_checkpoint, resident=step < args.total_steps)
        if step == args.total_steps:
            break
        next_round = step // args.round_steps + 1
        collection = _delta(args, next_round, weights, resident_step=step)
        _mixture(args, next_round)
        metrics = {
            "optimizer_steps": step,
            "eval/success_rate": result["success_rate"],
            "eval/mean_return": result["mean_return"],
            "collection/valid_heads": collection["splits"]["train"]["valid_actions"],
            "round": next_round,
        }
        if args.condition in FIXED_LATENCIES:
            for latency, cell in result["latencies"].items():
                metrics.update({f"eval/L{latency}/{key}": cell[key]
                                for key in ("success_rate", "mean_return", "episodes")})
        _write_json(control / f"step_{step}.continue.json", {"metrics": metrics})
    _heldout(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwenoft", "qwenpi_v3", "qwengr00t"), required=True)
    parser.add_argument("--condition", choices=("profile-latency", *FIXED_LATENCIES, "zero-latency"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--eval-config-profile", type=Path)
    parser.add_argument("--eval-config-fixed", type=Path)
    parser.add_argument("--latency-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vla-python", type=Path, required=True)
    parser.add_argument("--task-python", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--train-devices", required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    parser.add_argument("--simulator-gpu", type=int, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
