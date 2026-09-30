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

from latency_bench.cross_device import study
from latency_bench.cross_device.grouped_replay import (
    RunTrajectoryLogger,
    _run_grouped_episodes,
    grouped_inputs,
)
from latency_bench.cross_device.study_assets import prepare_model_assets
from latency_bench.cross_device.study_publication import DATASET_REPO, StudyPublication
from latency_bench.executors.factory import build_executor
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.utils.io import load_jsonl_records, read_json, write_json, write_json_atomic

FIT_SIZES = (5, 10, 20, 30, 40, 50)
ALL_RUN_IDS = range(55)
REUSED_RUN_IDS = range(10)
MODELS = ("qwenoft", "qwengr00t")
SPLIT_SEED = 20260929
SOURCES = {
    "qwenoft": {
        "ref": "sim2real/flappy/qwenoft/1x-rtx3090/instance_129d8b35443318ea/20260928T173802629272Z",
        "revision": "b43cfc544b818f55ce70d1f7e929b92831c433c7",
        "gpu": 0, "uuid": "GPU-9091e431-9a0b-168d-a5d2-b5053628d519",
        "reference_return": 439.13,
    },
    "qwengr00t": {
        "ref": "sim2real/flappy/qwengr00t/1x-rtx3090/instance_33621945a15d3b38/20260928T190345050625Z",
        "revision": "fbdc9ed65d2f16630d1d37c3a78467b8b147e262",
        "gpu": 1, "uuid": "GPU-6a1c5dc6-6a38-ff30-cdec-1121a23df467",
        "reference_return": 428.48,
    },
}


def public_source(model):
    """Keep physical GPU selection in the local runner, as in the original study."""
    return {key: SOURCES[model][key] for key in ("ref", "revision", "reference_return")}


def split_manifest():
    ids = list(ALL_RUN_IDS)
    random.Random(SPLIT_SEED).shuffle(ids)
    return {
        "original": {
            "valid": list(study.VALID_RUNS),
            "fit_order": [*study.FIT_RUNS, *(run_id for run_id in ALL_RUN_IDS if run_id not in REUSED_RUN_IDS)],
        },
        "random": {"valid": sorted(ids[:5]), "fit_order": ids[5:]},
    }


def latency_seed(run_id):
    # Retain the original validation streams and allocate distinct new streams.
    return run_id - study.VALID_RUNS[0] if run_id in study.VALID_RUNS else 1000 + run_id



def prepare_source(output, model):
    source = SOURCES[model]
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


def initialize(output, model):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model, "task": "flappy", "source": public_source(model),
        "split_seed": SPLIT_SEED, "splits": split_manifest(), "fit_sizes": list(FIT_SIZES),
        "episodes_per_run": study.EPISODES, "env_seed": study.ENV_SEED,
        "env_seed_stride": study.ENV_SEED_STRIDE, "methods": list(study.METHODS),
        "reference_return": SOURCES[model]["reference_return"],
        "sim_action_noise": "episode_frame" if model == "qwengr00t" else "deterministic",
        "simulation_protocol": "fixed_valid_runs_shared_reference_means",
        "normalizer_source": "paper/exps/sim2real_heldout/references.csv",
    }
    path = output / "budget.json"
    if path.exists():
        # Existing protocol/resume boundary: a directory owns one frozen study.
        if read_json(path) != manifest:
            raise ValueError(f"Budget protocol changed: {path}")
    else:
        write_json_atomic(path, manifest)
    return manifest


def runtime_base(source, worker_python):
    manifest = read_json(source / "experiment.json")
    base = prepare_model_assets(manifest, manifest["assets"], eval_config_path=source / "eval_template.yaml")
    base["policy"]["worker_python_executable"] = worker_python
    fps = read_json(source / "calibration/calibration.json")["selected_env_fps"]
    return base, manifest, fps


def record_execution_contract(output, stage, config, model):
    """Freeze measurement identity before accepting completed runs on resume."""
    hardware = collect_hardware_fingerprint()
    selected = [gpu for gpu in hardware["gpus"]
                if gpu["uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]][0]
    contract = {"config": config, "python": sys.executable, "gpu_uuid": selected["uuid"],
                "driver_version": hardware["gpu"]["driver_version"], "packages": hardware["packages"],
                "cpu": {key: value for key, value in hardware["cpu"].items() if key != "cpu_mhz"},
                "memory": hardware["memory"], "system": hardware["system"]}
    contract_path = output / f"{stage}_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError(f"{stage.title()} execution contract changed: {contract_path}")
    else:
        write_json_atomic(contract_path, contract)
        write_json(output / f"{stage}_hardware.json", {"gpu": selected, "fingerprint": hardware})


def measure_real(output, source, model, worker_python, publication):
    base, manifest, fps = runtime_base(source, worker_python)
    record_execution_contract(output, "real", {"base": base, "env_fps": fps}, model)
    for run_id in ALL_RUN_IDS:
        path = output / "real" / str(run_id)
        if run_id in REUSED_RUN_IDS:
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
            config = study.run_config(base, path, inference_device="cuda:0", simulator_device=None,
                                      env_fps=fps, run_id=run_id)
            study.execute_stage(config, path, manifest["assets"])
        started_at = read_json(timing)["started_at"]
        write_json_atomic(timing, {"stage": "real", "run_id": run_id, "started_at": started_at,
                                   "seconds": (path / "DONE").stat().st_mtime - started_at, "gpu_count": 1})
        publication.publish(f"real/{run_id}", [path, timing])
    (output / "REAL_DONE").touch()
    publication.publish("real_complete", [output / "REAL_DONE", output / "budget.json"])


def fit_profiles(output, source, model, publication, *, partitions):
    (output / "REAL_DONE").read_bytes()
    original = read_json(source / "experiment.json")
    fps = read_json(source / "calibration/calibration.json")["selected_env_fps"]
    for split, partition in partitions.items():
        for count in FIT_SIZES:
            path = output / "profiles" / split / str(count)
            if not (path / "DONE").exists():
                if path.exists():
                    shutil.rmtree(path)
                start = time.perf_counter()
                build_profile_from_traces(
                    sessions=[{"measure_session_id": run_id,
                               "observation_attempts": output / "real" / str(run_id) / "observation_attempts.jsonl"}
                              for run_id in partition["fit_order"][:count]],
                    output_dir=path, model_id=model, workload_id="flappy", gpu_class="1x-rtx3090",
                    instance_id=original["target_deployment"]["instance_id"], source_run_id=output.parent.name,
                    env_fps=fps, metadata={"fit_session_ids": partition["fit_order"][:count],
                                          "validation_session_ids": partition["valid"],
                                          "source": public_source(model)},
                )
                write_json(path / "timing.json", {"stage": "fit", "seconds": time.perf_counter() - start,
                                                  "gpu_count": 0})
                (path / "DONE").touch()
            publication.publish(f"profiles/{split}/{count}", [path])


def replay_group(executor, config, path, run_ids, on_run_complete):
    """Replay missing runs through an already resident model and environment pool."""
    for run_id in run_ids:
        run_path = path / "runs" / str(run_id)
        if run_path.exists():
            shutil.rmtree(run_path)  # Only incomplete runs are passed by the caller.
    executor.logger = RunTrajectoryLogger(
        path, on_run_complete, run_paths={run_id: path / "runs" / str(run_id) for run_id in run_ids},
        env_seed=study.ENV_SEED, env_seed_stride=study.ENV_SEED_STRIDE,
    )
    groups, seeds, sources = grouped_inputs(
        {run_id: {**config["latency"], "seed": latency_seed(run_id)} for run_id in run_ids},
        executor.clock, env_seed=study.ENV_SEED, env_seed_stride=study.ENV_SEED_STRIDE, replica_seed=True,
    )
    return _run_grouped_episodes(
        executor, groups, seeds, sources, config["evaluation"]["eval_max_steps"],
    )


def replay_settings(executor, jobs, output, publication):
    """Fill run slots across settings while preserving each run's public identity and RNG."""
    groups, seeds, sources, identities, paths = [], [], {}, {}, {}
    for session_id, (run_id, path, config) in enumerate(jobs):
        if path.exists():
            shutil.rmtree(path)  # Only incomplete runs enter the job list.
        paths[session_id] = path
        logical, run_seeds, run_sources = grouped_inputs(
            {run_id: {**config["latency"], "seed": latency_seed(run_id)}}, executor.clock,
            env_seed=study.ENV_SEED, env_seed_stride=study.ENV_SEED_STRIDE, replica_seed=True,
        )
        group = [study.ENV_SEED_STRIDE * session_id + episode for episode in range(study.EPISODES)]
        groups.append(group)
        seeds.extend(run_seeds)
        for episode, internal_id in enumerate(group):
            sources[internal_id] = run_sources[logical[0][episode]]
            identities[internal_id] = {"run_id": run_id, "episode_in_run": episode,
                                       "episode_seed": run_seeds[episode]}
    def publish_run(session_id, path):
        publication.publish(path.relative_to(output).as_posix(), [path])
    executor.logger = RunTrajectoryLogger(
        output / "runtime", publish_run, run_paths=paths,
        env_seed=study.ENV_SEED, env_seed_stride=study.ENV_SEED_STRIDE, episode_metadata=identities,
    )
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"PARALLEL_RUNS={len(jobs)} SETTINGS={len({path.parent for path in paths.values()})}", flush=True)
    _run_grouped_episodes(
        executor, groups, seeds, sources, jobs[0][2]["evaluation"]["eval_max_steps"],
    )
    timing = output / "runtime/groups" / f"{stamp}.json"
    write_json(timing, {"stage": "sim", "seconds": time.perf_counter() - started, "gpu_count": 1,
                        "runs": [path.relative_to(output).as_posix() for path in paths.values()]})
    publication.publish(f"runtime/groups/{stamp}", [timing])


def simulate(output, source, model, worker_python, publication, parallel_runs, *, partitions,
             combine_settings=False):
    base, _, fps = runtime_base(source, worker_python)
    base["policy"]["action_noise_seeded"] = model == "qwengr00t"
    contract = {"base": base, "env_fps": fps, "parallel_runs": parallel_runs}
    if combine_settings:
        contract["combine_settings"] = True
    record_execution_contract(output, "replay", contract, model)
    first_split = next(iter(partitions))
    first_profile = output / "profiles" / first_split / "5/profile.json"
    config = study.run_config(base, output / "runtime", inference_device="cuda:0", simulator_device=None,
                              env_fps=fps, run_id=0, method="temporal", profile_path=first_profile, latency_seed=0)
    config["evaluation"].update(
        eval_parallel_envs=parallel_runs, eval_episodes=study.EPISODES * parallel_runs,
    )
    config["executor"]["inference_batch_size"] = parallel_runs
    config["logging"]["simulated_pipeline_profile"] = False
    study._prepare_replay_runtime(config)
    start = time.perf_counter()
    executor = build_executor(config, inference_devices=["cuda:0"])
    try:
        executor.run_warmup()
        setup_seconds = time.perf_counter() - start
        write_json(output / "runtime/config.json", config)
        setup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        setup_path = output / "runtime/setups" / f"{setup_stamp}.json"
        write_json(setup_path, {"stage": "sim_setup", "seconds": setup_seconds,
                                                   "gpu_count": 1, "parallel_runs": parallel_runs})
        buffered_jobs = []
        for split, partition in partitions.items():
            for count in FIT_SIZES:
                profile = output / "profiles" / split / str(count) / "profile.json"
                for method in study.METHODS:
                    replay_config = study.run_config(base, output, inference_device="cuda:0", simulator_device=None,
                                                     env_fps=fps, run_id=0, method=method,
                                                     profile_path=profile, latency_seed=0)
                    run_ids = partition["valid"]
                    path = output / "sim" / split / str(count) / method
                    def publish_run(run_id, run_path):
                        publication.publish(f"{path.relative_to(output)}/runs/{run_id}", [run_path])
                    pending = []
                    for run_id in run_ids:
                        run_path = path / "runs" / str(run_id)
                        if (run_path / "DONE").exists():
                            publish_run(run_id, run_path)
                        else:
                            pending.append(run_id)
                    if not pending:
                        continue
                    if combine_settings:
                        buffered_jobs.extend((run_id, path / "runs" / str(run_id), replay_config)
                                             for run_id in pending)
                        while len(buffered_jobs) >= parallel_runs:
                            replay_settings(executor, buffered_jobs[:parallel_runs], output, publication)
                            del buffered_jobs[:parallel_runs]
                        continue
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    start = time.perf_counter()
                    replay_group(executor, replay_config, path, pending, publish_run)
                    write_json(path / "timings" / f"{stamp}.json", {
                        "stage": "sim", "run_ids": pending, "seconds": time.perf_counter() - start,
                        "gpu_count": 1, "parallel_runs": parallel_runs,
                    })
                    publication.publish(f"{path.relative_to(output)}/timing/{stamp}", [path / "timings" / f"{stamp}.json"])
        if buffered_jobs:
            replay_settings(executor, buffered_jobs, output, publication)
    finally:
        executor.close()
    (output / "SIM_DONE").touch()
    publication.publish("sim_complete", [output / "SIM_DONE", output / "runtime/setups"])


def verify_parallel(output, source, model, worker_python):
    """Gate formal replay on an actual GPU serial/batched action-trace comparison."""
    parallel_runs = 8
    verification_started = time.perf_counter()
    base, _, fps = runtime_base(source, worker_python)
    base["policy"]["action_noise_seeded"] = model == "qwengr00t"
    record_execution_contract(output, "replay", {"base": base, "env_fps": fps, "parallel_runs": parallel_runs}, model)
    if (output / "PARALLEL_VERIFIED").exists():
        return
    profile = source / "profile/profile.json"
    run_ids = list(range(8))
    for slots in (1, parallel_runs):
        if all((output / "verification" / str(slots) / method / "runs" / str(run_id) / "DONE").exists()
               for method in study.METHODS for run_id in run_ids):
            continue
        config = study.run_config(base, output / "verification" / str(slots) / "runtime",
                                  inference_device="cuda:0", simulator_device=None, env_fps=fps,
                                  run_id=0, method="temporal", profile_path=profile, latency_seed=0)
        config["evaluation"].update(
            eval_parallel_envs=slots, eval_episodes=study.EPISODES * parallel_runs, eval_max_steps=64,
        )
        config["executor"]["inference_batch_size"] = slots
        config["logging"]["simulated_pipeline_profile"] = False
        study._prepare_replay_runtime(config)
        executor = build_executor(config, inference_devices=["cuda:0"])
        try:
            executor.run_warmup()
            for method in study.METHODS:
                replay_config = study.run_config(base, output, inference_device="cuda:0", simulator_device=None,
                                                 env_fps=fps, run_id=0, method=method,
                                                 profile_path=profile, latency_seed=0)
                replay_config["evaluation"]["eval_max_steps"] = 64
                path = output / "verification" / str(slots) / method
                pending = [run_id for run_id in run_ids if not (path / "runs" / str(run_id) / "DONE").exists()]
                if pending:
                    replay_group(executor, replay_config, path, pending, lambda run_id, path: None)
        finally:
            executor.close()
    rows = []
    for method in study.METHODS:
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
    write_json(output / "verification/result.json", {"passed": passed, "max_steps": 64,
                                                      "parallel_runs": [1, parallel_runs], "runs": rows})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(output / "verification/timings" / f"{stamp}.json", {
        "stage": "verify", "seconds": time.perf_counter() - verification_started, "gpu_count": 1,
    })
    if not passed:
        raise RuntimeError(f"Serial/batched replay mismatch: {output / 'verification/result.json'}")
    (output / "PARALLEL_VERIFIED").touch()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "real", "fit", "sim"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--gpu-uuid", help="Execution GPU UUID; defaults to the original profiling GPU")
    parser.add_argument("--defer-uploads", action="store_true", help="queue completed stages for a batch publisher")
    args = parser.parse_args(argv)
    model = args.model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = SOURCES[model]["uuid"] if args.gpu_uuid is None else args.gpu_uuid
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    output = args.root.resolve() / model
    initialize(output, model)
    source = prepare_source(output, model)
    publication = StudyPublication(output, f"sim2real-budget/flappy/{args.root.name}/{model}",
                                   defer_uploads=args.defer_uploads)
    publication.publish("context", [output / "budget.json", source / "experiment.json",
                                     source / "eval_template.yaml", source / "calibration/calibration.json"])
    if args.stage == "verify":
        verify_parallel(output, source, model, args.worker_python)
        publication.publish("parallel_verified", [output / "PARALLEL_VERIFIED", output / "verification/result.json", output / "verification/timings"])
    elif args.stage == "real":
        measure_real(output, source, model, args.worker_python, publication)
    elif args.stage == "fit":
        fit_profiles(output, source, model, publication, partitions=split_manifest())
    elif args.stage == "sim":
        # Both cards finish real acquisition before either starts replay.
        for peer in MODELS:
            (args.root / peer / "REAL_DONE").read_bytes()
        (output / "PARALLEL_VERIFIED").read_bytes()
        verified_slots = read_json(output / "verification/result.json")["parallel_runs"][-1]
        simulate(output, source, model, args.worker_python, publication, verified_slots,
                 partitions=split_manifest())


if __name__ == "__main__":
    main()
