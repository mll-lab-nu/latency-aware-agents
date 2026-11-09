"""Compare frozen timing profiles with reserved, contended hardware sessions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import wasserstein_distance

from latency_bench.core.clock import EnvClock
from latency_bench.latency.samplers import build_sampler
from latency_bench.utils.io import load_jsonl_records, read_json, write_json
from scripts.experiments.temporal_profiles.compare import burst_lengths
from scripts.experiments.temporal_profiles.scores import timing_diagnostics


def read_session(source, session_id):
    folder = source / "sessions" / str(session_id)
    (folder / "DONE").read_bytes()
    records = load_jsonl_records(folder / "observation_attempts.jsonl")
    if any(b["obs_id"] <= a["obs_id"] for a, b in zip(records, records[1:])):
        raise ValueError(f"Unordered observations in session {session_id}")
    rows = [row for row in records if row["submitted"]]
    if len(rows) < 32 or {row["worker_id"] for row in rows} != {0}:
        raise ValueError(f"Session {session_id} requires >=32 completed requests from worker 0")
    pairs = np.array([[r["observation_to_action_latency_ms"], r["worker_service_time_ms"]] for r in rows])
    if not np.isfinite(pairs).all() or np.any(pairs[:, 1] <= 0) or np.any(pairs[:, 0] <= pairs[:, 1]):
        raise ValueError(f"Session {session_id} requires finite action latency > busy time > 0")
    return {"pairs": pairs, "episodes": [r["episode_id"] for r in rows]}


def diagnostics(values, real, threshold, frame_ms):
    metrics = timing_diagnostics(values, real, threshold, frame_ms)
    mask, truth = values > threshold, real > threshold
    count = int(mask[:-1].sum())
    true_count = int(truth[:-1].sum())
    persistence = float((mask[:-1] & mask[1:]).sum() / count) if count else None
    true_persistence = float((truth[:-1] & truth[1:]).sum() / true_count) if true_count else None
    lengths, real_lengths = burst_lengths(values, threshold), burst_lengths(real, threshold)
    metrics.update(std_ms=float(values.std()), threshold_ms=float(threshold),
                   slow_fraction_error=float(abs(mask.mean() - truth.mean())),
                   next_slow_given_slow=persistence,
                   persistence_error=(abs(persistence - true_persistence)
                                      if persistence is not None and true_persistence is not None else None),
                   mean_burst_requests=float(lengths.mean()) if len(lengths) else None,
                   burst_length_error=(float(wasserstein_distance(lengths, real_lengths))
                                       if len(lengths) and len(real_lengths) else None),
                   p95_error_ms=float(abs(np.quantile(values, .95) - np.quantile(real, .95))),
                   p99_error_ms=float(abs(np.quantile(values, .99) - np.quantile(real, .99))))
    return metrics


def compare_profiles(output):
    protocol = read_json(output / "protocol.json")
    target = output / "comparison"
    if (target / "DONE").exists():
        if read_json(target / "protocol.json") != protocol:
            raise ValueError("Comparison protocol changed; use a new output directory")
        return
    if target.exists():
        raise FileExistsError(target)
    source = Path(read_json(output / "measurement.json")["RUN_DIR"])
    fit_ids, test_ids = protocol["fit_session_ids"], protocol["held_out_session_ids"]
    if not fit_ids or not test_ids or set(fit_ids) & set(test_ids):
        raise ValueError("Comparison requires nonempty, disjoint fitting and held-out sessions")
    methods, settings = protocol["methods"], protocol["comparison"]
    for method in methods:
        profile = read_json(output / "profiles" / method / "profile.json")
        if profile["provenance"]["fit_options"]["session_ids"] != fit_ids:
            raise ValueError(f"{method} was not fitted on the shared training partition")
    training = np.concatenate([read_session(source, i)["pairs"] for i in fit_ids])
    cutoffs = {str(q): np.quantile(training, q, axis=0).tolist() for q in settings["slow_quantiles"]}
    cfg = protocol["measurement_config"]
    clock = EnvClock(env_fps=cfg["env"]["env_fps"], obs_fps=cfg["env"]["obs_fps"])
    rows, traces, load_evidence = [], {}, {}
    for session_id in test_ids:
        session = read_session(source, session_id)
        real = session["pairs"]
        traces[f"real_s{session_id}"] = real
        for column, channel in enumerate(("action", "worker")):
            for q, thresholds in cutoffs.items():
                rows.append({"method": "real", "session_id": session_id, "repeat": -1,
                             "channel": channel, "quantile": q,
                             **diagnostics(real[:, column], real[:, column], thresholds[column], clock.frame_ms)})
        for method_index, method in enumerate(methods):
            for repeat in range(settings["generation_repeats"]):
                seed = settings["seed"] + method_index * 1000000 + session_id * 1000 + repeat
                sampler = build_sampler({"method": method, "seed": seed,
                    "profile_path": str(output / "profiles" / method / "profile.json")}, clock=clock)
                samples = [sampler.sample({"worker_slot": 0, "episode_id": int(episode)})
                           for episode in session["episodes"]]
                generated = np.array([[s.latency_ms, s.worker_service_ms] for s in samples])
                traces[f"{method}_s{session_id}_r{repeat}"] = generated
                for column, channel in enumerate(("action", "worker")):
                    for q, thresholds in cutoffs.items():
                        rows.append({"method": method, "session_id": session_id, "repeat": repeat,
                                     "channel": channel, "quantile": q,
                                     **diagnostics(generated[:, column], real[:, column],
                                                   thresholds[column], clock.frame_ms)})
        print(f"Compared all profiles on held-out session {session_id}: {len(real)} requests", flush=True)
    for session_id in fit_ids + test_ids:
        path = source / "sessions" / str(session_id) / "contention/report.json"
        if path.exists():
            load_evidence[str(session_id)] = read_json(path)
    warnings = []
    if "memory_mock" in cfg["profiling"]:
        if len(load_evidence) != len(fit_ids + test_ids):
            raise ValueError("Missing contention reports for the profiling sessions")
        if cfg["profiling"]["memory_mock"]["condition"] == "irregular":
            for session_id, evidence in load_evidence.items():
                count = evidence["on_intervals_intersecting_observed_span"]
                if count < 10:
                    warnings.append(f"Session {session_id}: only {count} observed ON events; collect longer sessions before drawing burst conclusions.")
    fields = ("distribution_error_ms", "acf_error", "p95_error_ms", "p99_error_ms", "slow_fraction_error",
              "persistence_error", "burst_length_error", "slow_count_error_32")
    aggregates = []
    for method in methods:
        for channel in ("action", "worker"):
            for q in cutoffs:
                selected = [r for r in rows if (r["method"], r["channel"], r["quantile"]) == (method, channel, q)]
                result = {"method": method, "channel": channel, "quantile": q}
                for field in fields:
                    per_session = []
                    for session_id in test_ids:
                        values = [r[field] for r in selected if r["session_id"] == session_id and r[field] is not None]
                        per_session.append({"session_id": session_id, "available_draws": len(values),
                                            "mean": float(np.mean(values)) if values else None,
                                            "generation_sd": float(np.std(values)) if values else None})
                    complete = all(s["available_draws"] == settings["generation_repeats"] for s in per_session)
                    result[field] = {"mean": float(np.mean([s["mean"] for s in per_session])) if complete else None,
                                     "sessions": per_session}
                aggregates.append(result)
    with tempfile.TemporaryDirectory(prefix=".compare-", dir=output) as temporary:
        staged = Path(temporary) / "comparison"
        staged.mkdir()
        write_json(staged / "protocol.json", protocol)
        write_json(staged / "summary.json", {"fit_session_ids": fit_ids, "held_out_session_ids": test_ids,
            "cutoffs_from_fit_sessions_ms": cutoffs, "aggregates": aggregates,
            "contention_evidence": load_evidence, "warnings": warnings,
            "interpretation": "Equal-session means; generation SD is simulation variability, not uncertainty over hardware sessions. "
                              "Burst lengths count consecutive admitted requests, preserving ordinary episode boundaries. "
                              "ON labels are diagnostics only and are never supplied to fitting models. "
                              "No alignment to the realized test trace, significance test, or automatic best-method claim."})
        with (staged / "metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(staged / "sequences.npz", **traces)
        render_report(staged, traces, aggregates, test_ids, methods, warnings)
        (staged / "DONE").touch()
        staged.rename(target)


def render_report(output, traces, aggregates, sessions, methods, warnings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["real", *methods]
    figure, axes = plt.subplots(len(labels), 1, figsize=(12, 2 * len(labels)), sharex=True, sharey=True)
    for axis, method in zip(axes, labels):
        key = f"real_s{sessions[0]}" if method == "real" else f"{method}_s{sessions[0]}_r0"
        axis.plot(traces[key][:1000, 0], color="black", linewidth=.8)
        axis.set_ylabel(f"{method}\nms")
    axes[-1].set_xlabel("Admitted request index (independent generated sequences; not event-aligned)")
    figure.tight_layout()
    figure.savefig(output / "traces.png", dpi=150)
    plt.close(figure)
    table = []
    for row in aggregates:
        if row["channel"] == "action" and row["quantile"] == "0.95":
            values = [row[name]["mean"] for name in
                      ("distribution_error_ms", "acf_error", "persistence_error", "burst_length_error")]
            table.append("<tr><td>" + row["method"] + "</td>" + "".join(
                f"<td>{value:.4g}</td>" if value is not None else "<td>Unavailable</td>" for value in values) + "</tr>")
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Contended profiling comparison</title>'
        '<style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}td,th{border:1px solid;padding:8px}table{border-collapse:collapse}img{width:100%}</style>'
        '<h1>Temporal / AR timing-profile comparison</h1>'
        '<p>Profiles fitted on shared training sessions and compared with reserved hardware sessions. '
        'This table uses action latency and the training p95 threshold. Lower errors indicate closer agreement. '
        'Missing burst statistics remain unavailable rather than being treated as zero error.</p>'
        '<table><tr><th>Method</th><th>Distribution W1 (ms)</th><th>ACF RMSE</th>'
        '<th>Slow persistence error</th><th>Burst length W1 (requests)</th></tr>' + ''.join(table) + '</table>'
        '<p>See metrics.csv and summary.json for each session, worker timing, p99 thresholds, tail errors and draw variability. '
        'Repeated synthetic draws are not additional independent hardware measurements. No policy-return comparison is performed.</p>'
        + ''.join('<p>' + warning + '</p>' for warning in warnings)
        + '<img src="traces.png" alt="First held-out latency trace and independent generated samples">', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    compare_profiles(parser.parse_args().output_dir)
