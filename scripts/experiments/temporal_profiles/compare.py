"""Offline timing-generator pilot; no policy training or environment execution."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from threadpoolctl import threadpool_limits

from latency_bench.core.clock import EnvClock
from latency_bench.latency.samplers import build_sampler
from latency_bench.latency.sequence_profiles import fit_hmm

PROFILE_REVISION = "8531005becf942264a5c8e3965d1a94b657a90df"
DATASET_REVISION = "0b3b64a5d5f4a52e5b1decf1a30b01ec8637becc"
CASE = "experiments/sim2real-calibration/cases/rtx3090/flappy/qwenoft"
METHODS = ("fixed", "iid", "temporal", "hmm", "block")
SEED = 20261107
BLOCK_LENGTH = 32


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_sessions(case_dir, run_ids):
    sessions, audit = [], []
    for run_id in run_ids:
        path = case_dir / "real" / str(run_id) / "observation_attempts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [row["obs_id"] for row in rows]
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f"Observation order is not strictly increasing: {path}")
        accepted = [row for row in rows if row["submitted"]]
        pairs = np.array([[row["observation_to_action_latency_ms"],
                           row["worker_service_time_ms"]] for row in accepted])
        if not np.isfinite(pairs).all() or np.any(pairs <= 0):
            raise ValueError(f"Nonpositive or nonfinite timing in {path}")
        if {row["worker_id"] for row in accepted} != {0}:
            raise ValueError("This pilot requires exactly one inference worker, slot 0")
        if np.any(pairs[:, 0] <= pairs[:, 1]):
            raise ValueError("Log-overhead HMM requires action latency > worker busy time")
        episodes = np.array([row["episode_id"] for row in accepted], dtype=int)
        sessions.append({"run_id": run_id, "pairs": pairs, "episodes": episodes})
        audit.append({"run_id": run_id, "attempts": len(rows), "accepted": len(accepted),
                      "dropped": len(rows) - len(accepted),
                      "episodes": np.unique(episodes).tolist(),
                      "min_overhead_ms": float(np.min(pairs[:, 0] - pairs[:, 1]))})
    return sessions, audit


def sample_hmm(model, center, scale, n, seed):
    features, _ = model.sample(n_samples=n, random_state=seed)
    busy_and_overhead = np.exp(features * scale + center)
    return np.column_stack((busy_and_overhead.sum(axis=1), busy_and_overhead[:, 0]))


def sample_blocks(sources, n, seed, mean_length):
    """Stationary bootstrap: geometric blocks, circular within a source session."""
    rng = np.random.default_rng(seed)
    lengths = np.array([len(source) for source in sources])
    cumulative = np.cumsum(lengths)
    result = np.empty((n, 2))
    offset = 0
    while offset < n:
        flat_start = int(rng.integers(cumulative[-1]))
        source_id = int(np.searchsorted(cumulative, flat_start, side="right"))
        start = flat_start - (int(cumulative[source_id - 1]) if source_id else 0)
        count = min(int(rng.geometric(1.0 / mean_length)), n - offset)
        source = sources[source_id]
        result[offset:offset + count] = source[(start + np.arange(count)) % len(source)]
        offset += count
    return result


def burst_lengths(values, threshold):
    """Lengths of consecutive exceedances; callers keep sessions separate."""
    mask = np.r_[False, np.asarray(values) > threshold, False].astype(int)
    edges = np.diff(mask)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def session_acf(arrays, max_lag=64):
    """Center within sessions; never fabricate cross-session adjacent samples."""
    centered = [values - values.mean() for values in arrays]
    variance = sum(np.dot(x, x) for x in centered) / sum(len(x) for x in centered)
    if variance < 1e-12:
        return np.full(max_lag, np.nan)
    return np.array([
        sum(np.dot(x[:-lag], x[lag:]) for x in centered if len(x) > lag)
        / sum(len(x) - lag for x in centered if len(x) > lag) / variance
        for lag in range(1, max_lag + 1)
    ])


def describe(arrays, reference, threshold):
    values = np.concatenate(arrays)
    real_values = np.concatenate(reference)
    acf = session_acf(arrays)
    real_acf = session_acf(reference)
    bursts = np.concatenate([burst_lengths(x, threshold) for x in arrays])
    masks = [x > threshold for x in arrays]
    slow_previous = sum(int(mask[:-1].sum()) for mask in masks)
    slow_following = sum(int(np.sum(mask[:-1] & mask[1:])) for mask in masks)
    metrics = {
        "samples": len(values), "mean_ms": float(values.mean()),
        "std_ms": float(values.std()), "p95_ms": float(np.quantile(values, .95)),
        "p99_ms": float(np.quantile(values, .99)),
        "slow_fraction": float(np.mean(values > threshold)),
        "next_slow_given_slow": slow_following / slow_previous if slow_previous else None,
        "slow_spells_per_1000": float(1000 * len(bursts) / len(values)),
        "mean_slow_spell_requests": float(bursts.mean()) if len(bursts) else None,
        "p95_slow_spell_requests": float(np.quantile(bursts, .95)) if len(bursts) else None,
        "acf_lag1": float(acf[0]) if np.isfinite(acf[0]) else None,
        "acf_rmse_vs_real": float(np.sqrt(np.mean((acf - real_acf) ** 2)))
        if np.isfinite(acf).all() else None,
        "wasserstein_ms_vs_real": float(wasserstein_distance(values, real_values)),
        "ks_distance_vs_real": float(ks_2samp(values, real_values, method="asymp").statistic),
    }
    # KS is a descriptive distance here, not an IID hypothesis test.
    return metrics, acf, bursts


def run(output):
    source = output / "source"
    case_dir = source / "profiles" / CASE
    experiment = json.loads((source / "experiment.json").read_text())
    profile_path = case_dir / "profile" / "profile.json"
    profile = json.loads(profile_path.read_text())
    burst = json.loads((case_dir / "profile" / profile["burst_model_path"]).read_text())
    fit_ids, valid_ids = experiment["protocol"]["fit"], experiment["protocol"]["valid"]
    if set(fit_ids) & set(valid_ids):
        raise ValueError("Fitting and validation runs overlap")
    if profile["provenance"]["fit_session_ids"] != fit_ids:
        raise ValueError("Archived Temporal profile uses a different fitting split")
    fit, fit_audit = read_sessions(case_dir, fit_ids)
    valid, valid_audit = read_sessions(case_dir, valid_ids)
    fitted = np.concatenate([s["pairs"] for s in fit])
    if len(fitted) != profile["summary"]["n_samples"] or not np.isclose(
            fitted[:, 0].mean(), profile["summary"]["mean_ms"], atol=1e-8, rtol=0):
        raise ValueError("Downloaded fitting traces do not reproduce the archived profile summary")
    threshold = float(np.quantile(fitted[:, 0], .95))
    audit = {"fit": fit_audit, "validation": valid_audit,
             "fitting_samples": len(fitted),
             "validation_samples": sum(len(s["pairs"]) for s in valid),
             "slow_threshold_action_ms": threshold,
             "archived_worker_burst_threshold_ms": burst["spike_threshold_ms_by_worker_slot"],
             "profile_schema": burst["schema_version"]}
    write_json(output / "audit.json", audit)
    print(f"Verified {len(fit)} fit and {len(valid)} validation sessions; "
          f"{len(fitted):,} fitting samples reproduce archived mean.", flush=True)
    with threadpool_limits(limits=1):
        model, center, scale, parameters = fit_hmm(fit)
    write_json(output / "hmm_fit.json", parameters)

    clock = EnvClock(env_fps=profile["env_fps"], obs_fps=profile["env_fps"])
    generated = {method: [] for method in METHODS}
    for session in valid:
        run_id, n = session["run_id"], len(session["pairs"])
        for method_id, method in enumerate(METHODS):
            seed = SEED + 1000 * method_id + run_id
            if method == "hmm":
                pairs = sample_hmm(model, center, scale, n, seed)
            elif method == "block":
                pairs = sample_blocks([s["pairs"] for s in fit], n, seed, BLOCK_LENGTH)
            else:
                sampler = build_sampler({"method": method, "seed": seed,
                                         "fixed_latency_ms": profile["summary"]["mean_ms"],
                                         "profile_path": str(profile_path)}, clock=clock)
                samples = [sampler.sample({"episode_id": int(episode), "worker_slot": 0})
                           for episode in session["episodes"]]
                pairs = np.array([[s.latency_ms, s.worker_service_ms] for s in samples])
            generated[method].append(pairs)
        print(f"Generated five methods for validation run {run_id} ({n:,} requests)", flush=True)

    series = {"real": [s["pairs"] for s in valid], **generated}
    archive = {f"{method}_{run_id}": array
               for method, arrays in series.items() for run_id, array in zip(valid_ids, arrays)}
    np.savez_compressed(output / "sequences.npz", **archive)
    metrics, acfs, bursts = {}, {}, {}
    reference = [s["pairs"][:, 0] for s in valid]
    for method, pairs in series.items():
        metrics[method], acfs[method], bursts[method] = describe(
            [x[:, 0] for x in pairs], reference, threshold)
    fit_metrics, _, _ = describe([s["pairs"][:, 0] for s in fit], reference, threshold)
    write_json(output / "metrics.json", {"methods": metrics, "fit_data": fit_metrics})
    worker_threshold = burst["spike_threshold_ms_by_worker_slot"]["0"]
    worker_metrics = {
        method: describe([x[:, 1] for x in pairs], [s["pairs"][:, 1] for s in valid], worker_threshold)[0]
        for method, pairs in series.items() if method != "fixed"
    }
    write_json(output / "worker_spike_metrics.json", {
        "threshold_ms": worker_threshold,
        "note": "Archived Temporal worker-spike threshold; Fixed excluded because it rounds busy time to frames.",
        "methods": worker_metrics,
    })
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *metrics["real"]])
        writer.writeheader()
        writer.writerows({"method": method, **values} for method, values in metrics.items())

    root = Path(__file__).resolve().parents[3]
    provenance = {
        "case": CASE, "profile_repo": "MLL-Lab/LAGEN-profiles", "profile_revision": PROFILE_REVISION,
        "experiment_repo": "MLL-Lab/LAGEN-datasets", "experiment_revision": DATASET_REVISION,
        "model_bundle": experiment["assets"]["bundle"], "model_revision": experiment["model_revision"],
        "seed": SEED, "stationary_bootstrap_mean_block_length": BLOCK_LENGTH,
        "fit_run_ids": fit_ids, "validation_run_ids": valid_ids,
        "display_run": valid_ids[0], "display_start_request": 0, "display_requests": 1000,
        "env_fps": profile["env_fps"], "temporal_schema": burst["schema_version"],
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scipy", "hmmlearn", "scikit-learn", "matplotlib")},
        "scope": "Offline admitted-request timing sequences; no scheduler, game replay, or policy training.",
        "sampling": "One synthetic sequence per held-out run, matched in accepted count and episode boundaries.",
        "fixed_busy_time": "Existing Fixed sampler rounds action latency to frames for busy time; main plots use action latency only.",
        "threshold": "Fitting action-latency p95 for every method; diagnostic slow spells are distinct from internal Temporal burst labels.",
        "acf": "Within-session centering; no cross-session transitions. Undefined for constant Fixed.",
        "block_baseline": "Mean block length 32 fixed before validation; circular within a source session; pairs stay together.",
        "limitations": "One hardware/model/task case; untuned two-state HMM and block length; descriptive statistics, no significance claim.",
    }
    write_json(output / "provenance.json", provenance)
    snapshot = output / "code"
    snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in {".py", ".sh", ".txt", ".md"}:
            shutil.copy2(path, snapshot / path.name)
    from scripts.experiments.temporal_profiles.plot import render
    render(output, series, metrics, acfs, bursts, audit, provenance)
    print(f"Results and figures: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
