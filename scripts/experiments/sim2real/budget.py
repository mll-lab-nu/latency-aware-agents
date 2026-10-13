"""Flappy fit-budget study; one model/GPU per process, explicit real/sim barrier."""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.experiments.sim2real.assets import DATASET_REPO
from scripts.experiments.sim2real import runtime, replay
from latency_bench.executors.factory import build_executor
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.core.config import load_config
from latency_bench.utils.io import load_jsonl_records, read_json, write_json, write_json_atomic


def split_manifest(config):
    ids = list(range(config["total_runs"]))
    random.Random(config["split_seed"]).shuffle(ids)
    count = config["validation_runs"]
    return {
        "original": {
            "valid": config["original_valid"],
            "fit_order": [*config["original_fit"], *(run_id for run_id in range(config["total_runs"])
                                                   if run_id not in config["reused_run_ids"])],
        },
        "random": {"valid": sorted(ids[:count]), "fit_order": ids[count:]},
    }


def prepare_source(output, source):
    destination = output / "source"
    if not (destination / "REAL_DONE").exists():
        snapshot = Path(snapshot_download(
            repo_id=DATASET_REPO, repo_type="dataset", revision=source["revision"],
            allow_patterns=[f"{source['ref']}/{name}" for name in (
                "experiment.json", "eval_template.yaml", "REAL_DONE", "real_summary.json",
                "real_runs.csv", "calibration/calibration.json", "profile/*", "real/*",
            )],
        )) / source["ref"]
        shutil.copytree(snapshot, destination, dirs_exist_ok=True)
    return destination


def initialize(output, model, config):
    """Freeze one model's protocol, sources and report parameters before execution."""
    output.mkdir(parents=True, exist_ok=True)
    valid = config["original_valid"]
    campaign = {
        **config,
        "splits": split_manifest(config),
        "latency_seeds": {str(run_id): run_id - valid[0] if run_id in valid
                          else config["new_latency_seed_base"] + run_id
                          for run_id in range(config["total_runs"])},
    }
    manifest = {
        **campaign, "model": model, "source": campaign["sources"][model],
        "reference_return": campaign["sources"][model]["reference_return"],
        "sim_action_noise": "episode_frame" if model == "qwengr00t" else "deterministic",
        "simulation_protocol": "fixed_valid_runs_shared_reference_means",
    }
    records = ((output.parent / "budget.json", campaign), (output / "budget.json", manifest))
    # Both the shared campaign and this model's existing results own their frozen inputs.
    # Read both contracts before creating either file, including when only one survived.
    for path, expected in records:
        if path.exists() and read_json(path) != expected:
            raise ValueError(f"Budget protocol changed: {path}")
    for path, expected in records:
        if not path.exists():
            write_json_atomic(path, expected)
    return manifest


def measure_real(output, source, model, worker_python, manifest):
    protocol = manifest["protocol"]
    base, original, fps = runtime.runtime_base(source, worker_python)
    runtime.record_execution_contract(output, "real", {"base": base, "env_fps": fps})
    for run_id in range(manifest["total_runs"]):
        path = output / "real" / str(run_id)
        if run_id in manifest["reused_run_ids"]:
            if not (path / "DONE").exists():
                staged = path.with_name(f"{run_id}.importing")
                shutil.copytree(source / "real" / str(run_id), staged, dirs_exist_ok=True)
                staged.rename(path)
            continue
        timing = output / "timings" / f"real_{run_id}.json"
        if not (path / "DONE").exists():
            # Use the same filesystem clock as DONE, including on coarse-mtime filesystems.
            write_json_atomic(timing, {"stage": "real", "run_id": run_id})
            write_json_atomic(timing, {"started_at": timing.stat().st_mtime})
            config = runtime.run_config(base, path, inference_device="cuda:0", simulator_device=None,
                                      env_fps=fps, run_id=run_id, protocol=protocol)
            runtime.execute_stage(config, path, original["assets"])
        started_at = read_json(timing)["started_at"]
        write_json_atomic(timing, {"stage": "real", "model": model, "run_id": run_id, "run_ids": [run_id], "started_at": started_at,
                                   "seconds": (path / "DONE").stat().st_mtime - started_at, "gpu_count": 1})
    (output / "REAL_DONE").touch()


def fit_profiles(output, source, model, *, partitions, fit_sizes, source_ref):
    (output / "REAL_DONE").read_bytes()
    original = read_json(source / "experiment.json")
    fps = read_json(source / "calibration/calibration.json")["selected_env_fps"]
    for split, partition in partitions.items():
        for count in fit_sizes:
            path = output / "profiles" / split / str(count)
            if not (path / "DONE").exists():
                if path.exists():
                    shutil.rmtree(path)
                start = time.perf_counter()
                build_profile_from_traces(
                    sessions=[{"measure_session_id": run_id,
                               "observation_attempts": output / "real" / str(run_id) / "observation_attempts.jsonl"}
                              for run_id in partition["fit_order"][:count]],
                    output_dir=path, model_id=model, workload_id=original["task"], gpu_class=original["target_deployment"]["gpu_class"],
                    instance_id=original["target_deployment"]["instance_id"], source_run_id=output.parent.name,
                    env_fps=fps, metadata={"fit_session_ids": partition["fit_order"][:count],
                                          "validation_session_ids": partition["valid"],
                                          "source": source_ref},
                )
                write_json(path / "timing.json", {"stage": "fit", "model": model, "split": split,
                                                  "fit_runs": count, "run_ids": partition["fit_order"][:count],
                                                  "seconds": time.perf_counter() - start,
                                                  "gpu_count": 0})
                (path / "DONE").touch()


def simulate(output, source, model, worker_python, parallel_runs, *, partitions,
             manifest, combine_settings=False):
    protocol = manifest["protocol"]
    base, _, fps = runtime.runtime_base(source, worker_python)
    base["policy"]["action_noise_seeded"] = model == "qwengr00t"
    contract = {"base": base, "env_fps": fps, "parallel_runs": parallel_runs}
    if combine_settings:
        contract["combine_settings"] = True
    runtime.record_execution_contract(output, "replay", contract)
    first_split = next(iter(partitions))
    first_profile = output / "profiles" / first_split / str(manifest["fit_sizes"][0]) / "profile.json"
    config = runtime.run_config(base, output / "runtime", inference_device="cuda:0", simulator_device=None,
                              env_fps=fps, run_id=0, method="temporal", profile_path=first_profile, latency_seed=0, protocol=protocol)
    config['evaluation'].update(episodes=protocol['episodes'] * parallel_runs)
    config["executor"]["parallel_envs"] = parallel_runs
    config["executor"]["inference_batch_size"] = parallel_runs
    runtime.prepare_replay_runtime(config)
    start = time.perf_counter()
    executor = build_executor(config, inference_devices=["cuda:0"])
    with runtime.warmed_replay_executor(executor):
        setup_seconds = time.perf_counter() - start
        write_json(output / "runtime/config.json", config)
        setup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        setup_path = output / "runtime/setups" / f"{setup_stamp}.json"
        write_json(setup_path, {"stage": "sim_setup", "model": model, "seconds": setup_seconds,
                                                   "gpu_count": 1, "parallel_runs": parallel_runs})
        buffered_jobs = []
        for split, partition in partitions.items():
            for count in manifest["fit_sizes"]:
                profile = output / "profiles" / split / str(count) / "profile.json"
                for method in protocol["methods"]:
                    replay_config = runtime.run_config(base, output, inference_device="cuda:0", simulator_device=None,
                                                     env_fps=fps, run_id=0, method=method,
                                                     profile_path=profile, latency_seed=0, protocol=protocol)
                    run_ids = partition["valid"]
                    path = output / "sim" / split / str(count) / method
                    pending = []
                    for run_id in run_ids:
                        run_path = path / "runs" / str(run_id)
                        if not (run_path / "DONE").exists():
                            pending.append(run_id)
                    if not pending:
                        continue
                    if combine_settings:
                        buffered_jobs.extend((run_id, path / "runs" / str(run_id), replay_config, manifest["latency_seeds"][str(run_id)])
                                             for run_id in pending)
                        while len(buffered_jobs) >= parallel_runs:
                            replay.replay_settings(executor, buffered_jobs[:parallel_runs], output, protocol=protocol)
                            del buffered_jobs[:parallel_runs]
                        continue
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    start = time.perf_counter()
                    replay.replay_group(executor, replay_config, path, pending,
                                        protocol=protocol, latency_seeds=manifest["latency_seeds"])
                    write_json(path / "timings" / f"{stamp}.json", {
                        "stage": "sim", "model": model, "split": split, "fit_runs": count,
                        "method": method, "run_ids": pending, "seconds": time.perf_counter() - start,
                        "gpu_count": 1, "parallel_runs": parallel_runs,
                    })
        if buffered_jobs:
            replay.replay_settings(executor, buffered_jobs, output, protocol=protocol)
    (output / "SIM_DONE").touch()


def verify_parallel(output, source, model, worker_python, manifest):
    """Gate formal replay on an actual GPU serial/batched action-trace comparison."""
    protocol = manifest["protocol"]
    verification = manifest["verification"]
    parallel_runs = verification["parallel_runs"]
    verification_started = time.perf_counter()
    base, _, fps = runtime.runtime_base(source, worker_python)
    base["policy"]["action_noise_seeded"] = model == "qwengr00t"
    runtime.record_execution_contract(output, "replay", {"base": base, "env_fps": fps, "parallel_runs": parallel_runs})
    if (output / "PARALLEL_VERIFIED").exists():
        return
    profile = source / "profile/profile.json"
    run_ids = list(range(parallel_runs))
    for slots in (1, parallel_runs):
        if all((output / "verification" / str(slots) / method / "runs" / str(run_id) / "DONE").exists()
               for method in protocol["methods"] for run_id in run_ids):
            continue
        config = runtime.run_config(base, output / "verification" / str(slots) / "runtime",
                                  inference_device="cuda:0", simulator_device=None, env_fps=fps,
                                  run_id=0, method="temporal", profile_path=profile, latency_seed=0, protocol=protocol)
        config['evaluation'].update(episodes=protocol['episodes'] * parallel_runs, max_env_steps=verification['max_steps'])
        config["executor"]["parallel_envs"] = slots
        config["executor"]["inference_batch_size"] = slots
        runtime.prepare_replay_runtime(config)
        executor = build_executor(config, inference_devices=["cuda:0"])
        with runtime.warmed_replay_executor(executor):
            for method in protocol["methods"]:
                replay_config = runtime.run_config(base, output, inference_device="cuda:0", simulator_device=None,
                                                 env_fps=fps, run_id=0, method=method,
                                                 profile_path=profile, latency_seed=0, protocol=protocol)
                replay_config["evaluation"]["max_env_steps"] = verification["max_steps"]
                path = output / "verification" / str(slots) / method
                pending = [run_id for run_id in run_ids if not (path / "runs" / str(run_id) / "DONE").exists()]
                if pending:
                    replay.replay_group(executor, replay_config, path, pending,
                                        protocol=protocol, latency_seeds=manifest["latency_seeds"])
    rows = []
    for method in protocol["methods"]:
        for run_id in run_ids:
            signatures = []
            for slots in (1, parallel_runs):
                path = output / "verification" / str(slots) / method / "runs" / str(run_id)
                metrics = load_jsonl_records(path / "episode_metrics.jsonl")
                actions = load_jsonl_records(path / "actions.jsonl")
                signatures.append({
                    "returns": [(m["episode_id"], m["episode_return_env"], m["survival_steps"]) for m in metrics],
                    "actions": [(a["metadata"]["episode_id"], a["obs_env_step"], a["action"]["value"],
                                 a["ready_raw_frame"], a["apply_raw_frame"], a["dropped"]) for a in actions],
                    "latencies": [(a["latency"]["latency_ms"], a["latency_raw_frames"]) for a in actions],
                })
            rows.append({"method": method, "run_id": run_id,
                         **{key + "_match": signatures[0][key] == signatures[1][key] for key in signatures[0]}})
    passed = all(r[key] for r in rows for key in ("returns_match", "actions_match", "latencies_match"))
    write_json(output / "verification/result.json", {"passed": passed, "max_steps": verification["max_steps"],
                                                      "parallel_runs": [1, parallel_runs], "runs": rows})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(output / "verification/timings" / f"{stamp}.json", {
        "stage": "verify", "model": model, "seconds": time.perf_counter() - verification_started, "gpu_count": 1,
    })
    if not passed:
        raise RuntimeError(f"Serial/batched replay mismatch: {output / 'verification/result.json'}")
    (output / "PARALLEL_VERIFIED").touch()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "real", "fit", "sim"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--gpu-uuid", required=True, help="Execution GPU UUID")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model = args.model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    output = args.root.resolve() / model
    manifest = initialize(output, model, config)
    source = prepare_source(output, manifest["source"])
    if args.stage == "verify":
        verify_parallel(output, source, model, args.worker_python, manifest)
    elif args.stage == "real":
        measure_real(output, source, model, args.worker_python, manifest)
    elif args.stage == "fit":
        fit_profiles(output, source, model, partitions=manifest["splits"],
                     fit_sizes=manifest["fit_sizes"], source_ref=manifest["source"])
    elif args.stage == "sim":
        # Both cards finish real acquisition before either starts replay.
        for peer in manifest["models"]:
            (args.root / peer / "REAL_DONE").read_bytes()
        (output / "PARALLEL_VERIFIED").read_bytes()
        verified_slots = read_json(output / "verification/result.json")["parallel_runs"][-1]
        simulate(output, source, model, args.worker_python, verified_slots,
                 partitions=manifest["splits"], manifest=manifest)


if __name__ == "__main__":
    main()
