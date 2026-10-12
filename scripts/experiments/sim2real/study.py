"""One pinned sim2real experiment: calibration, fit, held-out runs and replay.

This module owns the experimental protocol. Process scheduling across experiments
and machines belongs to the caller.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t
from huggingface_hub import HfApi

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.core.config import load_config
from scripts.experiments.sim2real.assets import DATASET_REPO, download_real_experiment
from scripts.experiments.sim2real import runtime
from scripts.experiments.sim2real.assets import prepare_assets, prepare_model_assets, portable_eval_config
from latency_bench.latency.hardware_info import (
    collect_hardware_fingerprint, describe_participating_gpus, instance_id_from_gpu_deployment,
    public_hardware_projection,
)
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.utils.io import (
    load_jsonl_records,
    read_json,
    write_csv,
    write_json,
    write_json_atomic,
    write_yaml,
)

from latency_bench.utils.paths import REPO_ROOT


REGISTRY = REPO_ROOT / "configs/experiments/sim2real.yaml"


def admitted_latencies(output, count):
    samples = (
        row["observation_to_action_latency_ms"]
        for row in load_jsonl_records(output / "observation_attempts.jsonl")
        if row["submitted"]
    )
    return [next(samples) for _ in range(count)]


def calibrate(base, training_profile, output, *, inference_device, simulator_device, entry, protocol):
    """Match training-clock mean latency using two independent realtime probes."""
    result_path = output / "calibration.json"
    if result_path.exists():
        return read_json(result_path)
    training_fps = base["env"]["env_fps"]
    stride = training_fps / base["env"]["obs_fps"]
    target = training_profile["summary"]["mean_ms"] * training_fps / 1000.0
    fps = training_fps
    rounds = []
    for index in range(protocol["calibration_rounds"]):
        probe_dir = output / f"probe_{index}"
        config = runtime.run_config(
            base, probe_dir, inference_device=inference_device,
            simulator_device=simulator_device, env_fps=fps, run_id=-(index + 1), protocol=protocol,
        )
        config["experiment"]["seed"] = protocol["seed"] - protocol["calibration_samples"] * (index + 1)
        # At least one request is admitted per episode. The driver finishes
        # the current episode once the sample budget has been collected.
        config["evaluation"].update(
            episodes=protocol["calibration_samples"], calibration_samples=protocol["calibration_samples"],
        )
        runtime.execute_stage(config, probe_dir, entry)
        samples = admitted_latencies(probe_dir, protocol["calibration_samples"])
        mean_ms = mean(samples)
        actual_frames = mean_ms * fps / 1000.0
        rounds.append({
            "env_fps": fps, "obs_fps": fps / stride, "sample_count": len(samples),
            "mean_latency_ms": mean_ms, "mean_frames": actual_frames,
            "mean_raw_frames": mean(ms_to_raw_frames(value, fps) for value in samples),
            "target_frame_error": actual_frames - target,
        })
        fps = target * 1000.0 / mean_ms
    result = {
        "training_env_fps": training_fps, "training_obs_fps": base["env"]["obs_fps"],
        "training_profile_env_fps": training_profile["env_fps"],
        "training_mean_latency_ms": training_profile["summary"]["mean_ms"],
        "target_mean_frames": target, "observation_stride": stride,
        "selected_env_fps": fps, "selected_obs_fps": fps / stride, "rounds": rounds,
        "predicted_mean_raw_frames": mean(ms_to_raw_frames(value, fps) for value in samples),
    }
    write_json_atomic(result_path, result)
    return result


def fit_profile(output, *, model, task, deployment, experiment_id, calibration, contract):
    """Only the manifest's fitting sessions contribute to the frozen profile."""
    protocol = contract["protocol"]
    destination = output / "profile"
    if not (destination / "DONE").exists():
        if destination.exists():
            shutil.rmtree(destination)
        build_profile_from_traces(
            sessions=[{
                "measure_session_id": index,
                "observation_attempts": output / "real" / str(index) / "observation_attempts.jsonl",
            } for index in protocol["fit"]],
            output_dir=destination, model_id=model, workload_id=task,
            gpu_class=deployment["gpu_class"], instance_id=deployment["instance_id"],
            source_run_id=experiment_id, env_fps=calibration["selected_env_fps"],
            metadata={
                "fit_session_ids": list(protocol["fit"]), "validation_session_ids": list(protocol["valid"]),
                "experiment_contract": contract,
            },
        )
        (destination / "DONE").touch()
    return destination / "profile.json"


def confidence_interval(values):
    average = mean(values)
    half_width = t.ppf(0.975, len(values) - 1) * stdev(values) / len(values) ** 0.5
    return {"mean": average, "ci95_low": average - half_width, "ci95_high": average + half_width}


def paired_comparison(real, simulated):
    differences = [sim - measured for sim, measured in zip(simulated, real)]
    return {
        "simulated": confidence_interval(simulated),
        "paired_sim_minus_real": confidence_interval(differences),
        "mean_absolute_paired_error": mean(abs(value) for value in differences),
    }


def run_rows(output, groups):
    """Read portable run summaries from the requested completed stages."""
    rows = []
    for method, ids in groups:
        for index in ids:
            directory = output / "real" / str(index) if method in {"fit", "valid"} else output / "sim" / method / str(index)
            episodes = load_jsonl_records(directory / "episode_metrics.jsonl")
            row = {
                "condition": method, "run_id": index, "episodes": len(episodes),
                "mean_return": mean(item["episode_return_env"] for item in episodes),
                "mean_length": mean(item["survival_steps"] for item in episodes),
                "episode_metrics": (directory / "episode_metrics.jsonl").relative_to(output).as_posix(),
            }
            if "task_metrics" in episodes[0]["metadata"]:
                for name in episodes[0]["metadata"]["task_metrics"]:
                    row[f"task/{name}"] = mean(item["metadata"]["task_metrics"][name] for item in episodes)
            rows.append(row)
    return rows


def summarize_real(output, protocol):
    """Finish the real stage independently of any simulated results."""
    rows = run_rows(output, [("fit", protocol["fit"]), ("valid", protocol["valid"])])
    real = [row["mean_return"] for row in rows if row["condition"] == "valid"]
    result = {
        "return_field": "episode_return_env", "statistical_unit": "run_mean",
        "n_validation_runs": len(protocol["valid"]), "episodes_per_run": protocol["episodes"],
        "real": confidence_interval(real), "runs": rows,
    }
    task_metrics = {}
    for column in (key for key in rows[0] if key.startswith("task/")):
        real_values = [row[column] for row in rows if row["condition"] == "valid"]
        task_metrics[column.removeprefix("task/")] = {
            "real": confidence_interval(real_values),
        }
    result["task_metrics"] = task_metrics
    calibration = read_json(output / "calibration/calibration.json")
    timing = {}
    for name, ids in (("fit", protocol["fit"]), ("valid", protocol["valid"])):
        latencies = []
        drops = 0
        for index in ids:
            for attempt in load_jsonl_records(output / "real" / str(index) / "observation_attempts.jsonl"):
                if attempt["submitted"]:
                    latencies.append(attempt["observation_to_action_latency_ms"])
                else:
                    drops += 1
        fps = calibration["selected_env_fps"]
        frames = mean(latencies) * fps / 1000.0
        timing[name] = {
            "admitted_count": len(latencies), "capacity_drops": drops,
            "mean_latency_ms": mean(latencies), "mean_frames": frames,
            "mean_raw_frames": mean(ms_to_raw_frames(value, fps) for value in latencies),
            "target_frame_error": frames - calibration["target_mean_frames"],
        }
    result["timing"] = timing
    write_csv(output / "real_runs.csv", rows)
    write_json(output / "real_summary.json", result)


def summarize_sim(output, manifest):
    """Compare replays with the frozen held-out run means."""
    result = read_json(output / "real_summary.json")
    real_rows = result.pop("runs")
    protocol = manifest["protocol"]
    simulated_rows = run_rows(output, [(method, range(len(protocol["valid"]))) for method in protocol["methods"]])
    real = [row["mean_return"] for row in real_rows if row["condition"] == "valid"]
    result["methods"] = {
        method: paired_comparison(real, [row["mean_return"] for row in simulated_rows if row["condition"] == method])
        for method in protocol["methods"]
    }
    for name, metrics in result["task_metrics"].items():
        values = [row[f"task/{name}"] for row in real_rows if row["condition"] == "valid"]
        metrics["methods"] = {
            method: paired_comparison(values, [row[f"task/{name}"] for row in simulated_rows if row["condition"] == method])
            for method in protocol["methods"]
        }
    write_csv(output / "runs.csv", [*real_rows, *simulated_rows])
    write_json(output / "comparison.json", result)


def run_real(base, training_profile, entry, *, output, model, task, deployment,
             inference_device, simulator_device, contract, fingerprint, registry, resume):
    """Collect one local real experiment, stopping after held-out evaluation."""
    contract_path = output / "contract.json"
    if resume:
        # The existing profiling resume contract also rejects changed identity;
        # a held-out result must never be paired with a different model/profile.
        if read_json(contract_path) != contract:
            raise ValueError(f"Experiment contract does not match: {contract_path}")
    else:
        output.mkdir(parents=True)
        write_json_atomic(contract_path, contract)
        write_json(output / "hardware.json", fingerprint)
    protocol = contract["protocol"]
    write_yaml(output / "archived_eval.yaml", base)
    public_hardware = public_hardware_projection(
        fingerprint,
        cuda_visible_devices=deployment["gpus"][0]["uuid"],
    )
    public_hardware["instance_id"] = deployment["instance_id"]
    write_json(output / "hardware_public.json", public_hardware)
    manifest = {
        "task": task, "model": model, "assets": entry, "target_deployment": public_hardware,
        "experiment_id": output.name, "protocol": contract["protocol"],
        **{key: registry[key] for key in ("model_repo", "model_revision", "backbone")},
    }
    write_json_atomic(output / "experiment.json", manifest)
    write_yaml(output / "eval_template.yaml", portable_eval_config(base, entry, registry))
    print(f"EXPERIMENT_ID={output.name}\nOUTPUT_DIR={output}", flush=True)
    calibration = calibrate(
        base, training_profile, output / "calibration", inference_device=inference_device,
        simulator_device=simulator_device, entry=entry, protocol=protocol,
    )
    last_fit = next(index for index in reversed(protocol["collection_order"]) if index in protocol["fit"])
    for index in protocol["collection_order"]:
        path = output / "real" / str(index)
        runtime.execute_stage(runtime.run_config(
            base, path, inference_device=inference_device, simulator_device=simulator_device,
            env_fps=calibration["selected_env_fps"], run_id=index, protocol=protocol,
        ), path, entry)
        if index == last_fit:
            fit_profile(
                output, model=model, task=task, deployment=deployment, experiment_id=output.name,
                calibration=calibration, contract=manifest,
            )
    summarize_real(output, protocol)
    (output / "REAL_DONE").touch()
    print(f"SOURCE_DIR={output}", flush=True)
    return output


def run_sim(base, entry, *, output, manifest, source_revision,
            inference_device, simulator_device, fingerprint, parallel_runs=1):
    """Replay a pinned real experiment on the caller's independent execution GPU."""
    (output / "REAL_DONE").read_bytes()
    source = str(output.resolve()) if source_revision is None else {
        "repo_id": DATASET_REPO, "path": manifest["source_ref"], "revision": source_revision,
    }
    contract = {
        "source": source,
        "python": sys.executable, "config": base,
        "inference_device": inference_device, "simulator_device": simulator_device,
        "parallel_runs": parallel_runs,
        "gpus": [{"uuid": gpu["uuid"], "name": gpu["name"]} for gpu in fingerprint["gpus"]],
    }
    contract_path = output / "simulation_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError(f"Simulation contract does not match: {contract_path}")
    else:
        write_json_atomic(contract_path, contract)
        write_json(output / "simulation_hardware.json", fingerprint)
        write_json(output / "simulation_hardware_public.json", {
            "host_gpu_names": [gpu["name"] for gpu in fingerprint["gpus"]],
            "inference_device": inference_device, "simulator_device": simulator_device,
            "python_version": sys.version,
        })
    write_json(output / "simulation_source.json", {
        "source": source,
    })
    calibration = read_json(output / "calibration/calibration.json")
    profile_path = output / "profile/profile.json"
    protocol = manifest["protocol"]
    if parallel_runs == 1:
        for method in protocol["methods"]:
            for index, valid_run in enumerate(protocol["valid"]):
                path = output / "sim" / method / str(index)
                runtime.execute_stage(runtime.run_config(
                    base, path, inference_device=inference_device, simulator_device=simulator_device,
                    env_fps=calibration["selected_env_fps"], run_id=valid_run,
                    method=method, profile_path=profile_path, latency_seed=index,
                    protocol=protocol,
                ), path, entry)
    else:
        # The optional parallel runtime imports model/executor dependencies only for replay.
        from scripts.experiments.sim2real.replay import simulate
        simulate(base, output, manifest, inference_device=inference_device,
                 simulator_device=simulator_device, parallel_runs=parallel_runs)
    summarize_sim(output, manifest)
    (output / "SIM_DONE").touch()
    return output


def main(argv=None):
    registry = load_config(REGISTRY)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REGISTRY,
                        help="real-stage registry and protocol; replay reads the pinned manifest")
    parser.add_argument("--stage", choices=("real", "sim"), required=True)
    parser.add_argument("--task", choices=registry["tasks"])
    parser.add_argument("--model", choices=registry["models"])
    parser.add_argument("--source-dir", type=Path, help="Completed local real experiment")
    parser.add_argument("--source-ref")
    parser.add_argument("--source-revision")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--simulator-gpu", type=int)
    parser.add_argument("--parallel-runs", type=int, default=1,
                        help="independent sim run slots sharing batched model inference")
    parser.add_argument("--worker-python", default=sys.executable,
                        help="model worker interpreter; may differ from the simulator interpreter")
    parser.add_argument("--resume", metavar="EXPERIMENT_ID")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/sim2real")
    args = parser.parse_args(argv)
    registry = load_config(args.config)
    if args.stage == "real":
        if args.task is None or args.model is None:
            parser.error("real requires --task and --model")
        if args.source_dir is not None or args.source_ref is not None or args.source_revision is not None:
            parser.error("source arguments belong to the sim stage")
        task, model = args.task, args.model
    else:
        if args.task is not None or args.model is not None or args.resume is not None:
            parser.error("sim selects its experiment through its source")
        if args.source_dir is not None:
            if args.source_ref is not None or args.source_revision is not None:
                parser.error("--source-dir cannot be combined with remote source arguments")
            output = args.source_dir.resolve()
            (output / "REAL_DONE").read_bytes()
            manifest = read_json(output / "experiment.json")
        else:
            if args.source_ref is None or args.source_revision is None:
                parser.error("sim requires --source-dir or both --source-ref and --source-revision")
            ref = Path(args.source_ref)
            if ref.is_absolute() or ".." in ref.parts:
                parser.error("--source-ref must be a repository-relative path")
            if args.source_revision in {".", ".."} or Path(args.source_revision).name != args.source_revision:
                parser.error("--source-revision must be a single revision identifier")
            args.source_revision = HfApi().repo_info(
                DATASET_REPO, repo_type="dataset", revision=args.source_revision, expand=["sha"],
            ).sha
            output = args.output_root.resolve() / "replays" / ref / args.source_revision
            manifest = download_real_experiment(args.source_ref, args.source_revision, output)
        task, model = manifest["task"], manifest["model"]
        registry = manifest
    mikasa = task == "mikasa_intercept_grab_fast"
    if mikasa != (args.simulator_gpu is not None):
        parser.error("--simulator-gpu is required only for mikasa_intercept_grab_fast")
    if args.simulator_gpu == args.gpu:
        parser.error("InterceptGrab requires distinct simulator and inference GPUs")
    if args.resume is not None and (args.resume in {".", ".."} or Path(args.resume).name != args.resume):
        parser.error("--resume must be an experiment ID, not a path")
    visible = [args.simulator_gpu, args.gpu] if mikasa else [args.gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in visible)
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    if task in {"ant", "inverted_pendulum"}:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    inference_device = "cuda:1" if mikasa else "cuda:0"
    simulator_device = "cuda:0" if mikasa else None
    fingerprint = collect_hardware_fingerprint()
    if args.stage == "sim":
        entry = manifest["assets"]
        base = prepare_model_assets(registry, entry, eval_config_path=output / "eval_template.yaml")
        base["policy"]["worker_python_executable"] = args.worker_python
        result = run_sim(
            base, entry, output=output, manifest=manifest, source_revision=args.source_revision,
            inference_device=inference_device, simulator_device=simulator_device, fingerprint=fingerprint,
            parallel_runs=args.parallel_runs,
        )
        print(f"Completed simulation {result}", flush=True)
        return
    entry = registry["tasks"][task][model]
    deployment = describe_participating_gpus(
        gpus=fingerprint["gpus"], topology=fingerprint["gpu"]["topology"],
        cuda_visible_devices=str(args.gpu),
    )
    deployment["instance_id"] = instance_id_from_gpu_deployment(deployment)
    base, training_profile = prepare_assets(registry, entry)
    base["policy"]["worker_python_executable"] = args.worker_python
    experiment_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if args.resume is None else args.resume
    )
    output = args.output_root.resolve() / task / model / deployment["instance_id"] / experiment_id
    contract = {
        "model": model, "task": task, "assets": entry,
        "model_repo": registry["model_repo"], "model_revision": registry["model_revision"],
        "deployment": deployment, "visible_gpus": visible, "python": sys.executable,
        "archived_eval": base, "training_profile": training_profile,
        "protocol": registry["protocol"],
    }
    result = run_real(
        base, training_profile, entry, output=output, model=model, task=task,
        deployment=deployment, inference_device=inference_device, simulator_device=simulator_device,
        contract=contract, fingerprint=fingerprint, registry=registry, resume=args.resume is not None,
    )
    print(f"Completed {result}", flush=True)


if __name__ == "__main__":
    main()
