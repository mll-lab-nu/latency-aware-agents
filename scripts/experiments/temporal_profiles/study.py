"""Tune timing generators on fit sessions, then evaluate frozen choices."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from threadpoolctl import threadpool_limits

from latency_bench.core.clock import EnvClock
from latency_bench.latency.samplers import build_sampler
from scripts.experiments.temporal_profiles.compare import (
    CASE, DATASET_REVISION, PROFILE_REVISION, fit_hmm, read_sessions,
    sample_blocks, sample_hmm, write_json,
)
from scripts.experiments.temporal_profiles.scores import (
    HORIZONS, bootstrap_indices, evaluate_windows, interval, timing_diagnostics, windows,
)

SEED = 20261108
METHODS = ("fixed", "iid", "temporal", "hmm", "block")
INNER_REPETITIONS = 5
REPETITIONS = 10


def inner_split(fit_ids, seed=SEED):
    ids = np.random.default_rng(seed).permutation(fit_ids).tolist()
    return ids[:20], ids[20:]


def generate(method, session, sources, seed, *, hmm=None, block_length=None, profile_path=None):
    n = len(session["pairs"])
    if method == "hmm":
        return sample_hmm(*hmm[:3], n, seed)
    if method == "block":
        return sample_blocks([s["pairs"] for s in sources], n, seed, block_length)
    profile = json.loads(profile_path.read_text())
    sampler = build_sampler({"method": method, "seed": seed, "profile_path": str(profile_path),
                             "fixed_latency_ms": profile["summary"]["mean_ms"]},
                            clock=EnvClock(env_fps=profile["env_fps"], obs_fps=profile["env_fps"]))
    result = [sampler.sample({"episode_id": int(episode), "worker_slot": 0}) for episode in session["episodes"]]
    return np.array([[s.latency_ms, s.worker_service_ms] for s in result])


def independent_windows(pairs, seed, horizons=HORIZONS):
    # Exactly one window per horizon from each independently generated session.
    return {length: windows(pairs[:, 0], length, 1, seed + length)[0] for length in horizons}


def fit_candidate(sessions, states, max_iter=500):
    with threadpool_limits(limits=1):
        result = fit_hmm(sessions, n_components=states, max_iter=max_iter)
    selected = next(t for t in result[3]["trials"] if t["seed"] == result[3]["selected_seed"])
    if not selected["reached_tolerance"]:
        print(f"WARNING: selected {states}-state HMM did not reach tolerance within the declared budget", flush=True)
    return result


def select_models(training, tuning, output, horizons=HORIZONS, max_iter=500):
    all_values = np.concatenate([s["pairs"][:, 0] for s in training])
    center, scale = float(all_values.mean()), float(all_values.std())
    candidates = []
    for method, settings in (("hmm", (2, 3, 4)), ("block", (8, 16, 32, 64, 128))):
        for setting in settings:
            name = f"{method}_{setting}"
            print(f"Fitting/tuning {name}", flush=True)
            hmm = fit_candidate(training, setting, max_iter) if method == "hmm" else None
            if hmm is not None:
                write_json(output / f"{name}_fit.json", hmm[3])
            ensembles = {length: [] for length in horizons}
            for repetition in range(INNER_REPETITIONS):
                for session in tuning:
                    seed = SEED + 10000000 + repetition * 100000 + session["run_id"]
                    pairs = generate(method, session, training, seed, hmm=hmm, block_length=setting)
                    for length, vector in independent_windows(pairs, seed + 5000, horizons).items():
                        ensembles[length].append(vector)
            ensembles = {length: np.array(vectors) for length, vectors in ensembles.items()}
            rows = evaluate_windows(tuning, ensembles, center, scale, SEED + 40000000, horizons)
            candidate = {"method": method, "setting": setting, "run_scores": rows,
                         "score": float(np.mean([r["window_score"] for r in rows]))}
            candidates.append(candidate)
            write_json(output / "selection_progress.json", candidates)
            print(f"Inner-only score {name}: {candidate['score']:.6f}", flush=True)
    selected = {method: min((c for c in candidates if c["method"] == method), key=lambda c: (c["score"], c["setting"]))["setting"]
                for method in ("hmm", "block")}
    result = {"selected": selected, "candidates": candidates,
              "normalization": {"center_ms": center, "scale_ms": scale},
              "criterion": f"Equal-weight mean of fair energy and variogram scores at lengths {list(horizons)}; equal tuning-run weights."}
    write_json(output / "selection.json", result)
    return result


def aggregate(rows, run_ids):
    indices = bootstrap_indices(REPETITIONS, len(run_ids))
    matrices, summary = {}, {}
    metrics = [k for k in rows[0] if k not in {"method", "run_id", "repetition"}]
    for method in ("real", *METHODS):
        lookup = {(r["repetition"], r["run_id"]): r for r in rows if r["method"] == method}
        matrices[method], summary[method] = {}, {}
        for metric in metrics:
            values = [[lookup[(rep, run)].get(metric) for run in run_ids] for rep in range(REPETITIONS)]
            if any(value is None for row in values for value in row):
                continue
            matrix = np.array(values)
            matrices[method][metric] = matrix
            summary[method][metric] = interval(matrix, indices)
    differences = {}
    for method in METHODS:
        differences[method] = {metric: interval(matrix - matrices["temporal"][metric], indices)
                               for metric, matrix in matrices[method].items() if metric in matrices["temporal"]}
    return {"methods": summary, "difference_from_temporal": differences,
            "uncertainty": "2000 crossed bootstrap draws over 25 measurement runs and 10 generation repetitions; common draws across methods. Conditional on fitted profiles and selected settings; excludes fitting/model-selection uncertainty."}


def run(source, output, case=CASE, horizons=HORIZONS, max_iter=500):
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection").mkdir()
    (output / "sequences").mkdir()
    case_dir = source / "profiles" / case
    experiment_path = source / "datasets" / case / "experiment.json"
    if case == CASE and not experiment_path.exists():
        experiment_path = source / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    profile_path = case_dir / "profile" / "profile.json"
    profile = json.loads(profile_path.read_text())
    fit_ids, valid_ids = experiment["protocol"]["fit"], experiment["protocol"]["valid"]
    train_ids, tune_ids = inner_split(fit_ids)
    if set(fit_ids) & set(valid_ids) or profile["provenance"]["fit_session_ids"] != fit_ids:
        raise ValueError("Source fitting/validation contracts disagree")
    protocol = {
        "case": case, "source_directory": str(source), "profile_revision": PROFILE_REVISION,
        "experiment_revision": DATASET_REVISION, "model_bundle": experiment["assets"]["bundle"],
        "model_revision": experiment["model_revision"], "inner_training_ids": train_ids,
        "inner_tuning_ids": tune_ids, "refit_ids": fit_ids, "validation_ids": valid_ids,
        "hmm_states": [2, 3, 4], "block_lengths": [8, 16, 32, 64, 128],
        "hmm_restarts": [7, 42, 2026], "hmm_max_iterations": max_iter,
        "hmm_convergence_policy": "Fixed budget per restart; select by fit likelihood, retain and disclose nonconvergence without dropping a case or changing its budget.",
        "inner_repetitions": INNER_REPETITIONS, "final_repetitions": REPETITIONS,
        "horizons": list(horizons), "base_seed": SEED,
        "selection_objective": f"Mean of energy and variogram scores at {list(horizons)} requests; equal-run weights; smaller setting breaks exact ties.",
        "score_definition": "Energy: off-diagonal ensemble-pair U statistic, Euclidean norm / sqrt(window length). Variogram: exponent .5, dyadic lags <=64, each pair weight 1/lag, variance/M finite-ensemble correction, normalized total weight. Both use action latency standardized by fitting mean/std.",
        "window_sampling": "64 uniformly selected observed windows per run/horizon, potentially overlapping. Each generated ensemble member is ONE window from an independent generated session. Inner ensemble pools 5x5 sessions; each final repetition uses 25 independent sessions.",
        "seed_formula": "Inner: base+10000000+100000*rep+run. Final: base+20000000+100000*rep+1000*method_index+run. Window seeds recorded in study.py; observed windows shared across methods/repetitions.",
        "env_fps": profile["env_fps"], "frame_ms": profile["frame_ms"],
        "scope": "Offline accepted-request timing generators. Deadline crossing is measured per accepted request; no queue/admission replay, game evaluation or VLA training.",
        "prior_inspection": ("Flappy/QwenOFT/RTX3090 validation results informed the exploratory pilot. "
                             "The wider cohort was structurally audited for session lengths. No outer validation timing values enter parameter fitting or selection. "
                             "This is an expanded exploratory study, not a preregistered blinded confirmation."),
    }
    write_json(output / "protocol.json", protocol)
    print(f"Frozen inner split: train={train_ids}, tune={tune_ids}", flush=True)
    fit, fit_audit = read_sessions(case_dir, fit_ids)
    fit_lookup = {s["run_id"]: s for s in fit}
    fitting_values = np.concatenate([s["pairs"][:, 0] for s in fit])
    if len(fitting_values) != profile["summary"]["n_samples"] or not np.isclose(
            fitting_values.mean(), profile["summary"]["mean_ms"], rtol=0, atol=1e-8):
        raise ValueError("Fitting traces do not reproduce the archived profile")
    selection = select_models([fit_lookup[i] for i in train_ids], [fit_lookup[i] for i in tune_ids],
                              output / "selection", horizons, max_iter)
    chosen = selection["selected"]
    print(f"Selected using inner runs only: {chosen}; refitting on all 25 fit runs", flush=True)
    hmm = fit_candidate(fit, chosen["hmm"], max_iter)
    write_json(output / "hmm_refit.json", hmm[3])
    # Read outer timing values only after model selection and full refitting.
    valid, valid_audit = read_sessions(case_dir, valid_ids)
    write_json(output / "audit.json", {"fit": fit_audit, "validation": valid_audit})
    center, scale = float(fitting_values.mean()), float(fitting_values.std())
    slow_cutoff = float(np.quantile(fitting_values, .95))
    protocol.update(selected=chosen, normalization={"center_ms": center, "scale_ms": scale},
                    slow_cutoff_ms=slow_cutoff, validation_samples=sum(len(s["pairs"]) for s in valid))
    write_json(output / "protocol.json", protocol)
    np.savez_compressed(output / "sequences" / "real.npz", **{str(s["run_id"]): s["pairs"] for s in valid})
    rows = []
    real_diagnostics = {s["run_id"]: timing_diagnostics(s["pairs"][:, 0], s["pairs"][:, 0], slow_cutoff, profile["frame_ms"]) for s in valid}
    for repetition in range(REPETITIONS):
        generated = {}
        for method_id, method in enumerate(METHODS):
            ensembles = {length: [] for length in horizons}
            diagnostics = {}
            for session in valid:
                run_id = session["run_id"]
                seed = SEED + 20000000 + repetition * 100000 + method_id * 1000 + run_id
                pairs = generate(method, session, fit, seed, hmm=hmm, block_length=chosen["block"], profile_path=profile_path)
                generated[f"{method}_{run_id}"] = pairs
                diagnostics[run_id] = timing_diagnostics(pairs[:, 0], session["pairs"][:, 0], slow_cutoff, profile["frame_ms"])
                window_seed = SEED + 30000000 + repetition * 100000 + run_id
                for length, vector in independent_windows(pairs, window_seed, horizons).items():
                    ensembles[length].append(vector)
            scored = evaluate_windows(valid, {k: np.array(v) for k, v in ensembles.items()}, center, scale, SEED + 40000000, horizons)
            rows.extend({"method": method, "repetition": repetition, **row, **diagnostics[row["run_id"]]} for row in scored)
            print(f"Final repetition {repetition + 1}/{REPETITIONS}: {method} complete", flush=True)
        rows.extend({"method": "real", "repetition": repetition, "run_id": run_id, **values} for run_id, values in real_diagnostics.items())
        np.savez_compressed(output / "sequences" / f"generated_{repetition:02}.npz", **generated)
        write_json(output / "run_metrics.json", rows)
    with (output / "run_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows, valid_ids)
    write_json(output / "summary.json", summary)
    root = Path(__file__).resolve().parents[3]
    protocol["repository_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    protocol["packages"] = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "hmmlearn", "scikit-learn", "matplotlib")}
    write_json(output / "protocol.json", protocol)
    snapshot = output / "code"
    snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.suffix in {".py", ".sh", ".txt", ".md"}:
            shutil.copy2(path, snapshot / path.name)
    from scripts.experiments.temporal_profiles.study_plot import render
    render(output)
    print(f"Complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--hmm-max-iterations", type=int, default=500)
    args = parser.parse_args()
    run(args.source_dir.resolve(), args.output.resolve(), args.case, tuple(args.horizons), args.hmm_max_iterations)


if __name__ == "__main__":
    main()
