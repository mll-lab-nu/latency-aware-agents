"""Zero-teacher → Balance VLA DAgger with explicit H8/H16 label contracts.

Owns the experiment schedule and parallel sampling. Physics/label branching,
export, distributed training and GPU inference reuse existing implementations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
import wandb
import yaml

from latency_bench.core.config import load_config
from scripts.starvla import artifacts, runtime
from scripts.starvla.dagger_mixture import write_dagger_mixture
from latency_bench.policy.action_prefix import resolve_action_prefix
from latency_bench.integrations.starvla.task_contracts import task_spec
from scripts.starvla.train_task import BACKBONES, inference_bundle, training_config
from sample_factory.algo.mc_ppo.evaluation import comparison, write_json


ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SEED = 941731
MODELS = ("qwenoft", "qwenpi_v3", "qwengr00t")
LABEL_MODES = ("zero_full", "profile_conditioned")
TEACHER_QUERY_CADENCE_RAW_FRAMES = 1
TEACHER_REVISION = "a8210caa0e7432f208b377d678ca5f10f5f1b879"
TEACHER_ROOT = (ROOT / "outputs/starvla/artifacts/models/zero-latency"
                / "humanoidbench-balance-simple/small-policy/fasttd3-h1-v1")
TEACHER = TEACHER_ROOT / "checkpoint.pth"
TEACHER_CONFIG = TEACHER_ROOT / "fasttd3_train.yaml"


def _write_json_atomic(path: Path, value) -> None:
    """Publish one complete controller message or mixture revision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _AttachedTrainerProcess:
    """Observe a still-running Accelerate launcher after its controller restarted."""

    def __init__(self, pid, command):
        self.pid = pid
        self.args = command

    def poll(self):
        command_path = Path(f"/proc/{self.pid}/cmdline")
        if not command_path.is_file():
            return 0
        live_command = command_path.read_bytes().replace(b"\0", b" ").decode()
        return None if str(self.args[-1]) in live_command else 0

    def wait(self):
        while self.poll() is None:
            time.sleep(1.0)
        return 0


def _attached_trainer(control_dir):
    process_path = control_dir / "trainer.json"
    if not process_path.is_file():
        return None
    process = json.loads(process_path.read_text(encoding="utf-8"))
    attached = _AttachedTrainerProcess(process["pid"], process["command"])
    return attached if attached.poll() is None else None


def _resident_socket_dir(control_dir):
    longest_path = control_dir / "step_5000.rank_0.sock"
    if len(os.fsencode(longest_path)) < 100:
        return control_dir
    alias = Path("/tmp") / f"lb-dagger-{uuid.uuid4().hex[:12]}"
    alias.symlink_to(control_dir.resolve(), target_is_directory=True)
    return alias


def vla_experiment_config(model, label_mode, *, horizon, profile_path=None):
    """Use the requested zero/profile deployment condition and model horizon."""
    config = load_config(ROOT / "configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml")
    spec = task_spec(f"balance_profile_h{horizon}")
    if label_mode == "zero_full":
        config["latency"] = {"method": "zero", "seed": 271828, "add_latency_info": True}
    else:
        config["latency"] = {
            "method": "temporal",
            "profile_path": str(
                profile_path
                if profile_path is not None
                else ROOT / "results/profiling/downloads/profiles" / model
                / "1x-rtx3090/g1_sonic_balance_simple"
                / "instance_3a0d42681a03715c/profile.json"
            ),
            "profile_worker_slot": 0,
            "seed": 271828,
            "add_latency_info": True,
        }
    config["training"]["ppo"]["chunk_horizon"] = horizon
    config["training"]["ppo"]["active_heads"] = [0, horizon]
    settings = ({"mode": "none"} if label_mode == "zero_full" else
                {"mode": "profile_mean", "profile_path": config["latency"]["profile_path"]})
    config["policy"]["action_prefix"] = resolve_action_prefix(
        settings, env_fps=config["env"]["env_fps"],
        action_horizon=spec.action_horizon, action_dim=spec.action_dim,
    )
    return config


from latency_bench.envs.tasks.humanoidbench.profile_collection import rollout_batch


def _rollout_quality_metrics(prefix, result):
    metrics = {
        f"{prefix}/{key}": result[key]
        for key in ("mean_length", "mean_return", "n", "ge100", "ge500")
    }
    if "teacher_controlled" in result:
        metrics[f"{prefix}/teacher_controlled"] = int(result["teacher_controlled"])
    if "applied_head_counts" in result:
        counts = result["applied_head_counts"]
    elif result["episodes"] and "applied_heads" in result["episodes"][0]:
        counts = np.asarray([episode["applied_heads"] for episode in result["episodes"]]).sum(0)
    else:
        counts = None
    if counts is not None:
        total = sum(counts)
        metrics.update({f"{prefix}/applied_head_{index}_fraction": value / total
                        for index, value in enumerate(counts)})
    for name in ("action_ready_frame_counts", "worker_service_frame_counts"):
        if name in result:
            total = sum(result[name].values())
            metrics.update({f"{prefix}/{name.removesuffix('_counts')}_{frame}_fraction": count / total
                            for frame, count in result[name].items()})
    for name in ("prefix_head_counts", "loss_head_counts"):
        if name in result:
            metrics.update({f"{prefix}/{name.removesuffix('_counts')}_{index}": count
                            for index, count in enumerate(result[name])})
    if "worker_timings_sum" in result:
        metrics.update({f"{prefix}/timing/{name}": value
                        for name, value in result["worker_timings_sum"].items()})

    return metrics


def _wandb_stage_run(args, stage, config=None):
    suffix = "" if stage == "bootstrap" else f"-{stage}"
    disabled = args.smoke or args.collect_bootstrap_only
    return wandb.init(project="starvla_tasks", entity=None if disabled else os.environ["WANDB_ENTITY"],
                      id=args.wandb_runs[stage]["id"],
                      name=f"{args.task}-{args.label_mode}-{args.model}{suffix}",
                      group=args.run_id, mode="disabled" if disabled else "online", dir=str(args.output_dir),
                      config=config, resume="allow")


def _log_periodic_eval(args, stage, training_step, optimizer_steps, evaluation, best_score):
    with _wandb_stage_run(args, stage) as eval_run:
        eval_run.log({"global_step": training_step, "optimizer_steps": optimizer_steps,
                      "eval/mean_length": evaluation["mean_length"],
                      "eval/mean_return": evaluation["mean_return"],
                      "eval/best_mean_length": best_score, "eval/n": evaluation["n"],
                      "eval/ge100": evaluation["ge100"], "eval/ge500": evaluation["ge500"]})


def publish_result(args, bundle, evaluation, result, run_record):
    eval_name = "eval_l0.json" if args.condition == "zero-latency" else "eval_profile.json"
    write_json(bundle / eval_name, evaluation)
    write_json(bundle / "provenance.json", {
        "task": "humanoidbench_balance_simple", "model": args.model,
        "run_id": args.run_id, "condition": args.condition,
        "action_horizon": args.horizon, "label_mode": args.label_mode,
        "teacher": str(args.teacher), "teacher_config": str(args.teacher_config),
        "teacher_revision": args.teacher_revision,
        "selected_checkpoint": str(bundle / "checkpoints/model.pt"),
        "selection": "best development mean episode length over 32 fixed seeds",
        "optimizer_steps": result["optimizer_steps"],
        "development_seeds": run_record["development_seeds"],
        "heldout_seeds": run_record["heldout_seeds"],
        "wandb_runs": run_record["wandb_runs"],
    })
    (bundle / "README.md").write_text(
        f"# {args.model} — HumanoidBench Balance Simple\n\n"
        f"DAgger with {args.label_mode} supervision, action horizon {args.horizon}, "
        "and a 50 Hz H1 teacher queried every physical frame.\n")
    revision = artifacts.publish_model(
        bundle, task="humanoidbench_balance_simple", model=args.model,
        run_id=args.run_id, condition=args.condition, action_horizon=args.horizon)
    publication = {
        "repo_id": artifacts.MODEL_REPO_ID,
        "path": artifacts.model_path(
            "humanoidbench_balance_simple", args.model, args.run_id,
            condition=args.condition, action_horizon=args.horizon),
        "revision": revision,
    }
    write_json(args.output_dir / "publication.json", publication)
    return publication


def _training_launch(
    args,
    bootstrap,
    deltas,
    phase,
    steps,
    stop_step,
    pretrained,
    resume,
    *,
    control_dir=None,
    save_interval=500,
):
    training_phase = "bootstrap" if phase == "bootstrap" else "dagger"
    config = training_config(args, bootstrap=bootstrap, deltas=deltas, backbone=args.backbone,
                             run_id=training_phase, steps=steps, stop_step=stop_step,
                             pretrained=pretrained, resume=resume)
    config["wandb_name"] = f"{args.task}-{args.label_mode}-{args.model}"
    config["trainer"]["save_interval"] = save_interval
    if control_dir is not None:
        config["trainer"]["dagger_control_dir"] = str(control_dir)
    config["wandb_tags"] = ["balance", f"h{args.horizon}", args.label_mode, "h1-teacher",
                            "smoke" if args.smoke else "formal"]
    config["wandb_entity"] = None if args.smoke else os.environ["WANDB_ENTITY"]
    config["wandb_group"] = args.run_id
    config["framework"]["qwenvl"]["enable_gradient_checkpointing"] = args.gradient_checkpointing
    config["datasets"]["vla_data"]["sequential_step_sampling"] = False
    # One held-out minibatch does not need another 32 persistent loader workers.
    config["datasets"]["vla_data"]["eval_num_workers"] = 0
    robot_type = json.loads((bootstrap / "lerobot/task_contract.json").read_text())["robot_type"]
    mixture_path = Path(config["datasets"]["vla_data"]["custom_mixtures_path"])
    write_dagger_mixture(bootstrap, deltas, robot_type, mixture_path)
    stem = phase if stop_step is None else f"{phase}_{stop_step}"
    path = args.output_dir / f"{stem}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = runtime.environment() | {"CUDA_HOME": str(args.vla_python.resolve().parent.parent),
                           "WANDB_MODE": "disabled" if args.smoke else "online",
                           "WANDB_RUN_ID": args.wandb_runs[training_phase]["id"], "WANDB_RESUME": "allow"}
    command = [args.vla_python, "-m", "accelerate.commands.launch", "--config_file",
               runtime.STARVLA / "starVLA/config/deepseeds/deepspeed_zero2.yaml", "--num_processes",
               args.num_gpus,
               "--main_process_port", args.port, runtime.STARVLA / "starVLA/training/train_starvla.py",
               "--config_yaml", path]
    return command, env, mixture_path, args.output_dir / f"{stem}.log"


def train_segment(args, bootstrap, deltas, phase, steps, boundary, pretrained, resume):
    """Run the one-process bootstrap segment."""
    command, env, _mixture_path, log = _training_launch(
        args, bootstrap, deltas, phase, steps, boundary, pretrained, resume)
    runtime.run_command(command, log, cwd=runtime.STARVLA, env=env)
    training_phase = "bootstrap" if phase == "bootstrap" else "dagger"
    trained = args.output_dir / "training" / training_phase
    weights = trained / "checkpoints" / f"steps_{boundary}_pytorch_model.pt"
    bundle = inference_bundle(trained, weights, args.output_dir / f"bundles/{phase}_{boundary}",
                              bootstrap=bootstrap, backbone=BACKBONES["qwen"])
    state = trained / "checkpoints" / f"steps_{boundary}_state"
    return bundle, state


def start_dagger_training(
    args, budget, bootstrap, deltas, pretrained, resume, *, resume_step
):
    """Start the single long-lived DAgger trainer and return its control paths."""
    control_dir = args.output_dir / "dagger_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    socket_dir = _resident_socket_dir(control_dir)
    _write_json_atomic(control_dir / "resident_policy.json", {
        "socket_dir": str(socket_dir),
        "policy": {
            "task_contract_path": str(bootstrap / "lerobot/task_contract.json"),
            **runtime.TASK_INPUT_KEYS[task_spec("balance").name],
        },
    })
    if resume_step:
        (control_dir / f"step_{resume_step}.ready").unlink(missing_ok=True)
        for rank in range(args.num_gpus):
            (control_dir / f"step_{resume_step}.rank_{rank}.ready").unlink(missing_ok=True)
            (control_dir / f"step_{resume_step}.rank_{rank}.sock").unlink(missing_ok=True)
    command, env, mixture_path, log = _training_launch(
        args,
        bootstrap,
        deltas,
        "dagger",
        budget["rounds"] * budget["round_steps"],
        None,
        pretrained,
        resume,
        control_dir=control_dir,
        save_interval=budget["round_steps"],
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(str(part) for part in command), flush=True)
    print(f"Log: {log}", flush=True)
    with log.open("a" if resume else "w") as output:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=runtime.STARVLA,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    _write_json_atomic(control_dir / "trainer.json", {
        "pid": process.pid,
        "command": [str(part) for part in command],
    })
    return process, control_dir, mixture_path


def _wait_for_round_ready(process, ready_path):
    while not ready_path.is_file():
        return_code = process.poll()
        if return_code is not None:
            raise subprocess.CalledProcessError(return_code, process.args)
        time.sleep(1.0)


def _complete_checkpoint_steps(checkpoint_dir, total_steps):
    steps = []
    for state in checkpoint_dir.glob("steps_*_state"):
        step = int(state.name.removeprefix("steps_").removesuffix("_state"))
        weights = checkpoint_dir / f"steps_{step}_pytorch_model.pt"
        if state.is_dir() and weights.is_file() and step <= total_steps:
            steps.append(step)
    return sorted(steps)


def _completed_rollout(directory):
    episodes_path = directory / "episodes.json"
    if episodes_path.is_file():
        return json.loads(episodes_path.read_text())
    if directory.exists():
        directory.rename(directory.with_name(
            f"{directory.name}.interrupted-{int(time.time())}"
        ))
    return None


def _dagger_continue_metrics(round_number, optimizer_steps, evaluation, best_score, collection):
    return {
        "round": round_number,
        "optimizer_steps": optimizer_steps,
        "eval/mean_length": evaluation["mean_length"],
        "eval/mean_return": evaluation["mean_return"],
        "eval/best_mean_length": best_score,
        "eval/n": evaluation["n"],
        "eval/ge100": evaluation["ge100"],
        "eval/ge500": evaluation["ge500"],
        **{f"collection/{key}": collection[key] for key in (
            "seconds", "startup_seconds", "steady_seconds", "conversion_seconds",
            "labels_per_second", "generated_labels", "num_envs", "inference_seconds",
        )},
        **_rollout_quality_metrics("collection", collection),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vla-python", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    parser.add_argument("--teacher-config", type=Path, default=TEACHER_CONFIG)
    parser.add_argument("--teacher-revision", default=TEACHER_REVISION)
    parser.add_argument("--wandb-entity", default="saberrr-zju")
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--collect-bootstrap-only", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="qwenpi_v3")
    parser.add_argument("--horizon", type=int, choices=(8, 16), required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, required=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--port", type=int, default=29767)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--rollout-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    args.output_dir = args.output.resolve()
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    args.run_id = f"{args.output_dir.parent.name}-{args.output_dir.name}"
    os.environ["WANDB_RUN_GROUP"] = args.run_id
    args.task = "balance"
    args.condition = "zero-latency" if args.label_mode == "zero_full" else "latency-aware"
    args.num_workers, args.prefetch_factor = 16, 3
    args.persistent_workers, args.pin_memory = True, True
    args.normalization_source = None
    torch.set_num_threads(1)
    torch.manual_seed(1)
    teacher_oracle_seeds = list(range(800001, 800003 if args.smoke else 800033))
    if args.resume:
        config = json.loads((args.output_dir / "environment.json").read_text())
        # Saved prefix configuration identifies the new data/selection protocol.
        if "action_prefix" not in config["policy"]:
            raise ValueError("Resume requires a fixed-K run; evaluate legacy bundles in a new output directory")
        args.horizon = config["training"]["ppo"]["chunk_horizon"]
        budget = json.loads((args.output_dir / "budget.json").read_text())
        args.model, args.label_mode = budget["model"], budget["label_mode"]
        args.condition = "zero-latency" if args.label_mode == "zero_full" else "latency-aware"
        run_record = json.loads((args.output_dir / "run.json").read_text())
        history_path = args.output_dir / "history.json"
        history = json.loads(history_path.read_text()) if history_path.is_file() else []
        development = run_record["development_seeds"]
        heldout = run_record["heldout_seeds"]
        args.wandb_runs = run_record["wandb_runs"]
    else:
        config = vla_experiment_config(
            args.model, args.label_mode, horizon=args.horizon, profile_path=args.profile_path
        )
        config["env"]["make_kwargs"]["render_mode"] = "rgb_array"
        config["env"]["make_kwargs"]["camera_name"] = "head_camera"
        config["backend"]["initial_model_path"] = str(args.teacher)
        write_json(args.output_dir / "environment.json", config)
        profile_path = None if args.label_mode == "zero_full" else config["latency"]["profile_path"]
        budget = {"bootstrap_labels": 512 if args.smoke else 71000,
                  "bootstrap_steps": 2 if args.smoke else 5000,
                  "round_labels": 512 if args.smoke else 16000,
                  "round_steps": 2 if args.smoke else 500,
                  "rounds": 2 if args.smoke else 10, "micro_batch": args.micro_batch,
                  "global_batch": 128, "gradient_checkpointing": args.gradient_checkpointing,
                  "num_gpus": args.num_gpus,
                  "workers_per_rank": 16, "prefetch": 3,
                  "inference_seed": INFERENCE_SEED, "profile_is_deployment_proxy": True,
                  "rollout_workers": args.rollout_workers,
                  "inference_batch_size": args.inference_batch_size,
                  "rollout_overlap": args.rollout_overlap, "model": args.model,
                  "action_horizon": args.horizon,
                  "state_dim": 198 + args.horizon * 43 + args.horizon,
                  "label_mode": args.label_mode,
                  "teacher_query_cadence_raw_frames": TEACHER_QUERY_CADENCE_RAW_FRAMES,
                  "teacher_revision": args.teacher_revision,

                  "teacher_oracle_seeds": teacher_oracle_seeds,
                  "profile_path": profile_path}
        write_json(args.output_dir / "budget.json", budget)
        write_json(args.output_dir / "semantic_checklist.json", {
            "bootstrap_training": ("data_source_only" if args.collect_bootstrap_only
                                   else "from_random_initialization"),
            "model_geometry": f"{args.model}_h{args.horizon}",
            "gradient_checkpointing": args.gradient_checkpointing,
            "teacher": "zero50_fasttd3_h1_step_200000",
            "teacher_chunking": "closed_loop_requery_every_1_frame",
            "teacher_revision": args.teacher_revision,

            "label_mode": args.label_mode,
            "action_prefix": config["policy"]["action_prefix"],
            "profile_loss": "all_teacher_suffix_heads_until_termination",
            "replay": "round1_bootstrap_latest_50_50_then_bootstrap_latest_history_40_40_20",
            "dagger_optimizer": "single_continuous_optimizer_scheduler_rng_across_rounds",
            "rollout_checkpoint": "latest",
            "selection_checkpoint": "best_only_for_reporting",
            "development_eval_episodes": (0 if args.collect_bootstrap_only
                                            else (2 if args.smoke else 32)),
            "heldout_eval_episodes": (0 if args.collect_bootstrap_only
                                       else (2 if args.smoke else 64)),
            "teacher_oracle_episodes": len(teacher_oracle_seeds),
        })
        development = list(range(810001, 810003 if args.smoke else 810033))
        heldout = list(range(820101, 820103)) if args.smoke else list(range(820001, 820065))
        history = []
        args.wandb_runs = {
            "bootstrap": {"id": f"{args.run_id}-bootstrap"},
            "dagger": {"id": f"{args.run_id}-dagger"},
        }
        run_record = {"wandb_runs": args.wandb_runs, "development_seeds": development,
                      "heldout_seeds": heldout, "teacher": str(args.teacher),
                      "teacher_config": str(args.teacher_config),
                      "teacher_revision": args.teacher_revision,

                      "bootstrap_bundle": None}
        write_json(args.output_dir / "run.json", run_record)
        write_json(args.output_dir / "history.json", history)
    args.collection_seed = 0
    teacher_oracle_path = args.output_dir / "teacher_oracle/episodes.json"
    teacher_oracle = (
        json.loads(teacher_oracle_path.read_text())
        if teacher_oracle_path.is_file()
        else rollout_batch(args, config, teacher_oracle_path.parent, bundle=None, run=None,
                           seeds=teacher_oracle_seeds)
    )
    bootstrap = args.output_dir / "bootstrap"
    args.normalization_source = bootstrap / "lerobot" if args.resume else None
    best_entry = max(history, key=lambda entry: entry["evaluation"]["mean_length"]) if history else None
    best_bundle = Path(best_entry["bundle"]) if best_entry is not None else None
    best_score = best_entry["evaluation"]["mean_length"] if best_entry is not None else -float("inf")
    best_stage = ("bootstrap" if best_entry["round"] == 0 else "dagger") if best_entry else None
    optimizer_steps = history[-1]["optimizer_steps"] if history else 0
    bootstrap_best = (Path(run_record["bootstrap_bundle"])
                      if run_record["bootstrap_bundle"] is not None else None)
    completed = {entry["round"]: entry for entry in history}
    if 0 in completed:
        bootstrap_best = Path(completed[0]["bundle"])
    else:
        directory = bootstrap
        args.collection_seed = 4100001
        with _wandb_stage_run(args, "bootstrap", budget) as run:
            run.define_metric("eval/*", step_metric="optimizer_steps")
            for key, value in _rollout_quality_metrics("teacher_oracle", teacher_oracle).items():
                run.summary[key] = value
            run.log({"round": 0})
            if (directory / "episodes.json").is_file():
                collection = json.loads((directory / "episodes.json").read_text())
            else:
                collection = rollout_batch(
                    args, config, directory, bundle=None,
                    run=run, count=budget["bootstrap_labels"]
                )
            run.log({"round": 0, "collection/labels": budget["bootstrap_labels"],
                     **{f"collection/{key}": collection[key] for key in (
                         "seconds", "startup_seconds", "steady_seconds", "conversion_seconds",
                         "labels_per_second", "generated_labels", "num_envs", "inference_seconds")},
                     **_rollout_quality_metrics("collection", collection)})
            args.wandb_runs["bootstrap"]["url"] = run.url
            write_json(args.output_dir / "run.json", run_record)
        if args.collect_bootstrap_only:
            write_json(args.output_dir / "bootstrap_ready.json", {
                "bootstrap": str(directory), "labels": collection["generated_labels"],
                "model": args.model, "horizon": args.horizon, "label_mode": args.label_mode,
            })
            return
        args.normalization_source = bootstrap / "lerobot"
        checkpoint_dir = args.output_dir / "training/bootstrap/checkpoints"
        state = checkpoint_dir / f"steps_{budget['bootstrap_steps']}_state"
        weights = checkpoint_dir / f"steps_{budget['bootstrap_steps']}_pytorch_model.pt"
        bootstrap_best = args.output_dir / f"bundles/bootstrap_{budget['bootstrap_steps']}"
        if not (state.is_dir() and weights.is_file()):
            partial_steps = _complete_checkpoint_steps(checkpoint_dir, budget["bootstrap_steps"])
            partial_state = (checkpoint_dir / f"steps_{partial_steps[-1]}_state"
                             if partial_steps else None)
            bootstrap_best, state = train_segment(
                args, bootstrap, [], "bootstrap", budget["bootstrap_steps"],
                budget["bootstrap_steps"], partial_state, partial_state is not None)
        elif not (bootstrap_best / "checkpoints/model.pt").is_file():
            inference_bundle(args.output_dir / "training/bootstrap", weights, bootstrap_best,
                             bootstrap=bootstrap, backbone=BACKBONES["qwen"])
        evaluation_dir = args.output_dir / f"evaluation/bootstrap_{budget['bootstrap_steps']}"
        evaluation = (json.loads((evaluation_dir / "episodes.json").read_text())
                      if (evaluation_dir / "episodes.json").is_file()
                      else rollout_batch(args, config, evaluation_dir, bundle=bootstrap_best,
                                         run=None, seeds=development))
        optimizer_steps = budget["bootstrap_steps"]
        if evaluation["mean_length"] > best_score:
            best_bundle, best_score, best_stage = bootstrap_best, evaluation["mean_length"], "bootstrap"
        history.append({"round": 0, "phase_step": budget["bootstrap_steps"],
                        "optimizer_steps": optimizer_steps,
                        "bundle": str(bootstrap_best), "best_bundle": str(best_bundle),
                        "evaluation": evaluation})
        completed[0] = history[-1]
        write_json(args.output_dir / "history.json", history)
        write_json(args.output_dir / "checkpoints.json", {
            "best": str(best_bundle), "latest": str(bootstrap_best),
            "best_mean_length": best_score, "optimizer_steps": optimizer_steps})
        _log_periodic_eval(args, "bootstrap", budget["bootstrap_steps"],
                           optimizer_steps, evaluation, best_score)
        run_record["bootstrap_bundle"] = str(bootstrap_best)
        write_json(args.output_dir / "run.json", run_record)

    args.normalization_source = bootstrap / "lerobot"
    run_record["bootstrap_bundle"] = str(bootstrap_best)
    write_json(args.output_dir / "run.json", run_record)
    total_dagger_steps = budget["rounds"] * budget["round_steps"]
    dagger_checkpoint_dir = args.output_dir / "training/dagger/checkpoints"
    checkpoint_steps = _complete_checkpoint_steps(dagger_checkpoint_dir, total_dagger_steps)
    current_step = checkpoint_steps[-1] if checkpoint_steps else 0
    robot_type = json.loads((bootstrap / "lerobot/task_contract.json").read_text())["robot_type"]

    def record_dagger_boundary(
        boundary, *, log_wandb, resident_control_dir=None
    ):
        nonlocal best_bundle, best_score, best_stage, optimizer_steps
        round_number = boundary // budget["round_steps"]
        if round_number in completed:
            entry = completed[round_number]
            return Path(entry["bundle"]), entry["evaluation"]
        phase = f"dagger_{round_number:02d}"
        latest_bundle = args.output_dir / f"bundles/{phase}_{boundary}"
        weights = dagger_checkpoint_dir / f"steps_{boundary}_pytorch_model.pt"
        if not (latest_bundle / "checkpoints/model.pt").is_file():
            inference_bundle(args.output_dir / "training/dagger", weights, latest_bundle,
                             bootstrap=bootstrap, backbone=BACKBONES["qwen"])
        evaluation_dir = args.output_dir / f"evaluation/{phase}_{boundary}"
        evaluation = _completed_rollout(evaluation_dir)
        if evaluation is None:
            evaluation = rollout_batch(
                args, config, evaluation_dir, bundle=latest_bundle,
                run=None, seeds=development,
                resident_control_dir=resident_control_dir,
                resident_step=boundary,
            )
        optimizer_steps = budget["bootstrap_steps"] + boundary
        if evaluation["mean_length"] > best_score:
            best_bundle, best_score, best_stage = latest_bundle, evaluation["mean_length"], "dagger"
        entry = {"round": round_number, "phase_step": budget["round_steps"],
                 "optimizer_steps": optimizer_steps, "bundle": str(latest_bundle),
                 "best_bundle": str(best_bundle), "evaluation": evaluation}
        history.append(entry)
        completed[round_number] = entry
        write_json(args.output_dir / "history.json", history)
        write_json(args.output_dir / "checkpoints.json", {
            "best": str(best_bundle), "latest": str(latest_bundle),
            "best_mean_length": best_score, "optimizer_steps": optimizer_steps})
        if log_wandb:
            _log_periodic_eval(args, "dagger", boundary, optimizer_steps, evaluation, best_score)
        return latest_bundle, evaluation

    def collect_dagger_round(round_number, bundle, *, run, resident_control_dir=None, step=None):
        directory = args.output_dir / f"dagger_{round_number:02d}"
        args.collection_seed = 4100001 + round_number * 100000
        collection = _completed_rollout(directory)
        if collection is not None:
            return directory, collection
        collection = rollout_batch(
            args, config, directory, bundle=bundle, run=run,
            count=budget["round_labels"], resident_control_dir=resident_control_dir,
            resident_step=step,
        )
        return directory, collection

    if current_step < total_dagger_steps:
        if current_step == 0:
            latest = bootstrap_best
            next_round = 1
            with _wandb_stage_run(args, "dagger", budget) as run:
                run.define_metric("eval/*", step_metric="optimizer_steps")
                for key, value in _rollout_quality_metrics("teacher_oracle", teacher_oracle).items():
                    run.summary[key] = value
                next_directory, collection = collect_dagger_round(
                    next_round, latest, run=run
                )
                run.log({"round": next_round,
                         "optimizer_steps": budget["bootstrap_steps"],
                         **_rollout_quality_metrics("collection", collection)})
                args.wandb_runs["dagger"]["url"] = run.url
                write_json(args.output_dir / "run.json", run_record)
            deltas = [next_directory]
            pretrained = bootstrap_best / "checkpoints/model.pt"
        else:
            latest = (
                Path(completed[current_step // budget["round_steps"]]["bundle"])
                if current_step // budget["round_steps"] in completed
                else None
            )
            deltas = [
                args.output_dir / f"dagger_{round_number:02d}"
                for round_number in range(1, current_step // budget["round_steps"] + 1)
            ]
            restored_continue = (
                args.output_dir / "dagger_control"
                / f"step_{current_step}.continue.json"
            )
            if restored_continue.is_file():
                deltas.append(
                    args.output_dir
                    / f"dagger_{current_step // budget['round_steps'] + 1:02d}"
                )
            pretrained = dagger_checkpoint_dir / f"steps_{current_step}_state"

        control_dir = args.output_dir / "dagger_control"
        process = _attached_trainer(control_dir)
        if process is None:
            process, control_dir, mixture_path = start_dagger_training(
                args, budget, bootstrap, deltas, pretrained, current_step > 0,
                resume_step=current_step,
            )
        else:
            mixture_path = args.output_dir / "dagger_mixture.json"

        if current_step:
            continue_path = control_dir / f"step_{current_step}.continue.json"
            if not continue_path.is_file():
                _wait_for_round_ready(process, control_dir / f"step_{current_step}.ready")
                latest, evaluation = record_dagger_boundary(
                    current_step, log_wandb=True, resident_control_dir=control_dir
                )
                next_round = current_step // budget["round_steps"] + 1
                next_directory, collection = collect_dagger_round(
                    next_round, latest, run=None,
                    resident_control_dir=control_dir, step=current_step,
                )
                deltas.append(next_directory)
                write_dagger_mixture(bootstrap, deltas, robot_type, mixture_path)
                optimizer_steps = budget["bootstrap_steps"] + current_step
                _write_json_atomic(
                    continue_path,
                    {"metrics": _dagger_continue_metrics(
                        next_round, optimizer_steps, evaluation, best_score, collection)},
                )

        for boundary in range(current_step + budget["round_steps"],
                              total_dagger_steps + 1, budget["round_steps"]):
            if boundary < total_dagger_steps:
                _wait_for_round_ready(process, control_dir / f"step_{boundary}.ready")
            else:
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, process.args)
            latest, evaluation = record_dagger_boundary(
                boundary,
                log_wandb=boundary == total_dagger_steps,
                resident_control_dir=control_dir if boundary < total_dagger_steps else None,
            )
            if boundary == total_dagger_steps:
                continue
            next_round = boundary // budget["round_steps"] + 1
            next_directory, collection = collect_dagger_round(
                next_round, latest, run=None,
                resident_control_dir=control_dir, step=boundary,
            )
            deltas.append(next_directory)
            write_dagger_mixture(bootstrap, deltas, robot_type, mixture_path)
            optimizer_steps = budget["bootstrap_steps"] + boundary
            _write_json_atomic(
                control_dir / f"step_{boundary}.continue.json",
                {"metrics": _dagger_continue_metrics(
                    next_round, optimizer_steps, evaluation, best_score, collection)},
            )
    else:
        latest = Path(completed[budget["rounds"]]["bundle"])
        optimizer_steps = completed[budget["rounds"]]["optimizer_steps"]

    results, evaluated = {}, {}
    for name, bundle in (("bootstrap", bootstrap_best), ("best", best_bundle), ("latest", latest)):
        directory = args.output_dir / f"heldout/{name}"
        if bundle not in evaluated:
            evaluated[bundle] = _completed_rollout(directory)
            if evaluated[bundle] is None:
                evaluated[bundle] = rollout_batch(
                    args, config, directory, bundle=bundle, run=None, seeds=heldout
                )
        else:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "episodes.json", evaluated[bundle])
        results[name] = evaluated[bundle]
    result = {"optimizer_steps": optimizer_steps,
              "new_optimizer_steps": optimizer_steps,
              "budget": budget, "heldout": results,
              "best_vs_bootstrap": comparison(results["best"]["episodes"], results["bootstrap"]["episodes"])}
    write_json(args.output_dir / "result.json", result)
    final_stage = "dagger"
    for name, stage in (("bootstrap", "bootstrap"), ("best", best_stage), ("latest", final_stage)):
        with _wandb_stage_run(args, stage) as run:
            run.log({f"heldout/{name}/mean_length": results[name]["mean_length"],
                     f"heldout/{name}/mean_return": results[name]["mean_return"],
                     f"heldout/{name}/n": results[name]["n"],
                     f"heldout/{name}/ge100": results[name]["ge100"],
                     f"heldout/{name}/ge500": results[name]["ge500"]})
            if name == "best":
                run.summary["heldout_best_mean_length"] = results[name]["mean_length"]
    with _wandb_stage_run(args, final_stage) as run:
        run.summary["completed_optimizer_steps"] = optimizer_steps
        run.summary["new_optimizer_steps"] = result["new_optimizer_steps"]
        run.summary["best_vs_bootstrap_mean_length_gain"] = result["best_vs_bootstrap"]["mean_length_gain"]
        run.summary["best_vs_bootstrap_paired_95_interval"] = result["best_vs_bootstrap"]["paired_95_interval"]
        run.summary["best_vs_bootstrap_accepted"] = result["best_vs_bootstrap"]["accepted"]
    if args.publish:
        publication = publish_result(args, best_bundle, results["best"], result, run_record)
        print(f"MODEL_REVISION={publication['revision']}")


if __name__ == "__main__":
    main()
