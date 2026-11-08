"""Download, audit and compare the complete pinned 30-case calibration cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np

from scripts.experiments.temporal_profiles.compare import (
    CASE, DATASET_REVISION, PROFILE_REVISION, read_sessions, write_json,
)

GPUS = ("a100", "l40s", "rtx3090", "rtx4090", "rtx5090")
TASKS = ("flappy", "inverted_pendulum", "mikasa_intercept_grab_fast")
MODELS = ("qwenoft", "qwengr00t")
CASES = tuple(f"experiments/sim2real-calibration/cases/{gpu}/{task}/{model}"
              for gpu, task, model in product(GPUS, TASKS, MODELS))
MATRIX_HORIZONS = (8, 16)
HMM_MAX_ITERATIONS = 1000


def download(source, reuse):
    source.mkdir(parents=True, exist_ok=True)
    manifest = {"profile_repo": "MLL-Lab/LAGEN-profiles", "profile_revision": PROFILE_REVISION,
                "experiment_repo": "MLL-Lab/LAGEN-datasets", "experiment_revision": DATASET_REVISION,
                "cases": list(CASES), "expected_runs_per_case": 50}
    manifest_path = source / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Source directory belongs to another revision or cohort")
    write_json(manifest_path, manifest)
    if reuse is not None:
        shutil.copytree(reuse / "profiles" / CASE, source / "profiles" / CASE, dirs_exist_ok=True)
        dest = source / "datasets" / CASE / "experiment.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reuse / "experiment.json", dest)
    files = []
    for case in CASES:
        files.append(("datasets", DATASET_REVISION, f"{case}/experiment.json"))
        files.extend(("profiles", PROFILE_REVISION, f"{case}/profile/{name}") for name in
                     ("profile.json", "latency_distribution.json", "latency_burst_model.json"))
        files.extend(("profiles", PROFILE_REVISION, f"{case}/real/{run}/observation_attempts.jsonl")
                     for run in range(50))

    def fetch(item):
        kind, revision, filename = item
        destination = source / kind / filename
        if not destination.exists():
            hf_hub_download(f"MLL-Lab/LAGEN-{kind}", filename, repo_type="dataset",
                            revision=revision, local_dir=source / kind)
        return destination.stat().st_size

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, item) for item in files]
        for count, future in enumerate(as_completed(futures), 1):
            total_bytes += future.result()
            if count % 25 == 0 or count == len(files):
                print(f"Download {count}/{len(files)} files; {total_bytes / 1e6:.1f} MB", flush=True)
    print(f"Sources ready: {source}", flush=True)


def audit(source):
    """Fit-value checks and structural outer checks; no outer timing summaries."""
    rows = []
    for case in CASES:
        case_dir = source / "profiles" / case
        experiment = json.loads((source / "datasets" / case / "experiment.json").read_text())
        profile = json.loads((case_dir / "profile/profile.json").read_text())
        fit_ids, valid_ids = experiment["protocol"]["fit"], experiment["protocol"]["valid"]
        if set(fit_ids) & set(valid_ids) or sorted(fit_ids + valid_ids) != list(range(50)):
            raise ValueError(f"Invalid archived split: {case}")
        if profile["provenance"]["fit_session_ids"] != fit_ids:
            raise ValueError(f"Profile fitting split mismatch: {case}")
        fit, fit_audit = read_sessions(case_dir, fit_ids)
        fitting_values = np.concatenate([s["pairs"][:, 0] for s in fit])
        if len(fitting_values) != profile["summary"]["n_samples"] or not np.isclose(
                fitting_values.mean(), profile["summary"]["mean_ms"], rtol=0, atol=1e-8):
            raise ValueError(f"Fitting data do not reproduce archived profile: {case}")
        valid_counts = []
        for run in valid_ids:
            records = [json.loads(line) for line in (case_dir / f"real/{run}/observation_attempts.jsonl").read_text().splitlines()]
            valid_counts.append({"run_id": run, "accepted": sum(bool(r["submitted"]) for r in records),
                                 "attempts": len(records)})
        row = {"case": case, "fit": fit_audit, "validation_counts": valid_counts,
               "fit_samples": len(fitting_values), "validation_samples": sum(r["accepted"] for r in valid_counts),
               "min_fit_length": min(len(s["pairs"]) for s in fit),
               "min_validation_length": min(r["accepted"] for r in valid_counts),
               "frame_ms": profile["frame_ms"], "env_fps": profile["env_fps"]}
        rows.append(row)
        print(f"Audit {case.rsplit('/cases/', 1)[1]}: fit min={row['min_fit_length']}, "
              f"valid min={row['min_validation_length']}, samples={row['fit_samples'] + row['validation_samples']:,}", flush=True)
    write_json(source / "cohort_audit.json", rows)
    return rows


def run_all(source, output, workers):
    audits = json.loads((source / "cohort_audit.json").read_text())
    if [r["case"] for r in audits] != list(CASES):
        raise ValueError("The audit does not cover the declared cohort")
    minimum = min(min(r["min_fit_length"], r["min_validation_length"]) for r in audits)
    if minimum < max(MATRIX_HORIZONS):
        raise ValueError("At least one session is shorter than the common window horizons")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "cases": list(CASES), "primary_horizons": list(MATRIX_HORIZONS),
        "horizon_reason": "Structural audit minimum is 26 accepted requests. Windows of 8 and 16 retain every session in every case. No padding or concatenation of independent runs.",
        "hmm_max_iterations": HMM_MAX_ITERATIONS,
        "hmm_convergence_policy": "Three restarts per candidate; fixed budget; retain best fitting likelihood and disclose tolerance failures. No case dropped or budget changed based on final scores.",
        "secondary_diagnostics": "ACF through min(64, session length - 1); count-window errors at 8/32/128 requests only when supported. Missing diagnostics do not remove primary-score cases.",
        "inner_split": "20 training + 5 tuning sessions drawn only from 25 original fit sessions, seed 20261108",
        "outer_validation": "25 separate real runs per case, 10 generated repetitions, 5 methods, 2 consecutive episodes per real run",
        "aggregation": "Equal weights per real run within case and per case across the fixed 30-case panel. Cases are not independent random samples of GPUs/models/tasks.",
        "uncertainty": "Conditional crossed bootstrap with common validation run-ID and generation-repetition draws across cases and methods, preserving reused seeds. Fixed equal case weights; fitting and selection uncertainty excluded. Case intervals unadjusted for multiple comparisons.",
        "scope": "Offline accepted-request timing fidelity; no new hardware collection, queue replay, VLA training or game-return evaluation.",
        "sources": json.loads((source / "manifest.json").read_text()),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a fresh output directory")
    write_json(protocol_path, protocol)
    shutil.copy2(source / "cohort_audit.json", output / "cohort_audit.json")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               MPLCONFIGDIR=str(output / ".mplconfig"), XDG_CACHE_HOME=str(output / ".cache"))

    def run_case(case):
        relative = case.split("/cases/")[1]
        destination = output / "cases" / relative
        if (destination / "summary.json").exists() and (destination / "index.html").exists():
            return {"case": case, "status": "complete", "reused_completed_case": True}
        if destination.exists():
            raise ValueError(f"Incomplete case directory requires inspection before restart: {destination}")
        log_path = output / "logs" / f"{relative.replace('/', '__')}.log"
        log_path.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "scripts.experiments.temporal_profiles.study",
                   "--source-dir", str(source), "--output", str(destination), "--case", case,
                   "--horizons", *map(str, MATRIX_HORIZONS), "--hmm-max-iterations", str(HMM_MAX_ITERATIONS)]
        print(f"Starting {relative}", flush=True)
        start = time.monotonic()
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        row = {"case": case, "status": "complete" if result.returncode == 0 else "failed",
               "returncode": result.returncode, "elapsed_seconds": time.monotonic() - start,
               "log": str(log_path.relative_to(output))}
        print(f"{row['status'].upper()} {relative} ({row['elapsed_seconds'] / 60:.1f} minutes)", flush=True)
        return row

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, case) for case in CASES]
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(output / "status.json", completed)
    failed = [row for row in completed if row["status"] != "complete"]
    if failed:
        raise RuntimeError(f"{len(failed)} cases failed; inspect status.json and logs before restarting")
    print(f"All {len(completed)} cases complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "audit", "run"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reuse-pilot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "download":
        download(args.source_dir.resolve(), args.reuse_pilot)
    elif args.command == "audit":
        audit(args.source_dir.resolve())
    else:
        if args.output is None:
            parser.error("run requires --output")
        run_all(args.source_dir.resolve(), args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
