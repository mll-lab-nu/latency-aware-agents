"""Aggregate the fixed 30-case panel and save a portable timing-study report."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.temporal_profiles.compare import write_json
from scripts.experiments.temporal_profiles.matrix import CASES, GPUS, MODELS, TASKS
from scripts.experiments.temporal_profiles.plot import COLORS
from scripts.experiments.temporal_profiles.scores import bootstrap_indices, evaluate_windows, windows
from scripts.experiments.temporal_profiles.study import METHODS, REPETITIONS, SEED

LABELS = {"real": "Measured hardware", "fixed": "Fixed", "iid": "IID", "temporal": "Our Temporal",
          "hmm": "Tuned HMM", "block": "Tuned blocks"}
TASK_LABELS = {"flappy": "Flappy Bird", "inverted_pendulum": "Inverted Pendulum",
               "mikasa_intercept_grab_fast": "MIKASA InterceptGrabFast"}
MODEL_LABELS = {"qwenoft": "QwenOFT", "qwengr00t": "QwenGR00T"}
COMPARISONS = (("hmm", "temporal"), ("block", "temporal"), ("hmm", "block"))


def estimate(mean, samples):
    return {"mean": float(mean), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def bootstrap(matrix, indices):
    repetitions, runs = indices
    return matrix[repetitions[:, :, None], runs[:, None, :]].mean(axis=(1, 2))


def bootstrap_supported(matrix, eligible, run_ids, indices):
    """Retain the same run-ID clusters when secondary horizons omit short runs."""
    full = np.full((matrix.shape[0], len(run_ids)), np.nan)
    full[:, [run_ids.index(run) for run in eligible]] = matrix
    repetitions, runs = indices
    samples = np.nanmean(full[repetitions[:, :, None], runs[:, None, :]], axis=(1, 2))
    if not np.isfinite(samples).all():
        raise ValueError("Bootstrap draw has no supported run; this horizon lacks enough coverage")
    return samples


def long_window_scores(case_dir, protocol):
    """Secondary 32/128-request scores use only supported whole sessions."""
    path = case_dir / "secondary_windows.json"
    if path.exists():
        return json.loads(path.read_text())
    with np.load(case_dir / "sequences/real.npz") as real:
        sessions = [{"run_id": run, "pairs": real[str(run)]} for run in protocol["validation_ids"]]
    results = {}
    for length in (32, 128):
        eligible = [s for s in sessions if len(s["pairs"]) >= length]
        rows = []
        if len(eligible) >= 2:
            for repetition in range(REPETITIONS):
                with np.load(case_dir / "sequences" / f"generated_{repetition:02}.npz") as generated:
                    for method in METHODS:
                        ensemble = np.array([windows(generated[f"{method}_{s['run_id']}"][:, 0], length, 1,
                                                    SEED + 30000000 + repetition * 100000 + s["run_id"] + length)[0]
                                             for s in eligible])
                        scored = evaluate_windows(eligible, {length: ensemble},
                                                  protocol["normalization"]["center_ms"],
                                                  protocol["normalization"]["scale_ms"],
                                                  SEED + 40000000, horizons=(length,))
                        rows.extend({"method": method, "repetition": repetition, **row} for row in scored)
        results[str(length)] = {"eligible_run_ids": [s["run_id"] for s in eligible], "rows": rows}
    write_json(path, results)
    return results


def collect(output):
    records = []
    common_run_ids = None
    for case in CASES:
        relative = case.split("/cases/")[1]
        gpu, task, model = relative.split("/")
        directory = output / "cases" / relative
        protocol = json.loads((directory / "protocol.json").read_text())
        rows = json.loads((directory / "run_metrics.json").read_text())
        if protocol["horizons"] != [8, 16] or protocol["final_repetitions"] != REPETITIONS:
            raise ValueError(f"Inconsistent primary protocol: {case}")
        run_ids = protocol["validation_ids"]
        if common_run_ids is None:
            common_run_ids = run_ids
        if run_ids != common_run_ids:
            raise ValueError("Panel bootstrap requires matched archived run IDs")
        lookup = {(r["method"], r["repetition"], r["run_id"]): r for r in rows}
        raw_metrics = ("window_score", "energy_8", "variogram_8", "energy_16", "variogram_16",
                       "distribution_error_ms", "acf_error", "slow_count_error_8",
                       "slow_count_error_32", "slow_count_error_128", "lag1", "miss_1_frame", "miss_2_frames")
        # Same run and repetition draws preserve common environment/generation
        # seeds across hardware/model cases, not only across methods in a case.
        indices = bootstrap_indices(REPETITIONS, len(run_ids), seed=SEED)
        data = {}
        for method in METHODS:
            matrices = {}
            for metric in raw_metrics:
                values = [[lookup[method, rep, run].get(metric) for run in run_ids] for rep in range(REPETITIONS)]
                if not any(value is None for row in values for value in row):
                    matrices[metric] = np.array(values)
            matrices["distribution_error_sd"] = matrices["distribution_error_ms"] / protocol["normalization"]["scale_ms"]
            real_miss = np.array([lookup["real", 0, run]["miss_1_frame"] for run in run_ids])
            matrices["deadline_rate_error_pp"] = 100 * np.abs(matrices["miss_1_frame"] - real_miss[None, :])
            data[method] = {metric: {"mean": float(matrix.mean()), "samples": bootstrap(matrix, indices),
                                      "run_count": len(run_ids)} for metric, matrix in matrices.items()}
        secondary = long_window_scores(directory, protocol)
        for length, result in secondary.items():
            eligible = result["eligible_run_ids"]
            if len(eligible) < 2:
                continue
            secondary_lookup = {(r["method"], r["repetition"], r["run_id"]): r for r in result["rows"]}
            for method in METHODS:
                matrix = np.array([[secondary_lookup[method, rep, run]["window_score"] for run in eligible]
                                   for rep in range(REPETITIONS)])
                data[method][f"secondary_score_{length}"] = {
                    "mean": float(matrix.mean()), "samples": bootstrap_supported(matrix, eligible, run_ids, indices), "run_count": len(eligible)}
        fits = [directory / "selection" / f"hmm_{states}_fit.json" for states in (2, 3, 4)] + [directory / "hmm_refit.json"]
        convergence = []
        for path in fits:
            fit = json.loads(path.read_text())
            selected = next(t for t in fit["trials"] if t["seed"] == fit["selected_seed"])
            convergence.append({"stage": path.stem, "selected_converged": selected["reached_tolerance"],
                                "selected_iterations": selected["iterations"],
                                "failed_restart_count": sum(not t["reached_tolerance"] for t in fit["trials"])})
        record = {"case": case, "relative": relative, "gpu": gpu, "task": task, "model": model,
                  "protocol": protocol, "data": data, "convergence": convergence,
                  "real_lag1": float(np.mean([lookup["real", 0, run]["lag1"] for run in run_ids])),
                  "real_miss_1_frame": float(np.mean([lookup["real", 0, run]["miss_1_frame"] for run in run_ids])),
                  "secondary_coverage": {h: len(v["eligible_run_ids"]) for h, v in secondary.items()}}
        records.append(record)
        print(f"Aggregated {len(records)}/30: {relative}", flush=True)
    return records


def group_summary(records, metric):
    result = {"methods": {}, "paired": {}}
    for method in METHODS:
        supported = [r["data"][method][metric] for r in records if metric in r["data"][method]]
        if supported:
            result["methods"][method] = {**estimate(np.mean([v["mean"] for v in supported]),
                                                    np.mean([v["samples"] for v in supported], axis=0)),
                                          "cases": len(supported), "runs": sum(v["run_count"] for v in supported)}
    for a, b in COMPARISONS:
        supported = [r for r in records if metric in r["data"][a] and metric in r["data"][b]]
        if supported:
            deltas = [r["data"][a][metric]["mean"] - r["data"][b][metric]["mean"] for r in supported]
            samples = np.mean([r["data"][a][metric]["samples"] - r["data"][b][metric]["samples"] for r in supported], axis=0)
            result["paired"][f"{a}_minus_{b}"] = {**estimate(np.mean(deltas), samples),
                                                      "cases": len(supported), "lower_mean_cases": sum(d < 0 for d in deltas)}
    return result


def write_tables(output, records):
    metrics = sorted(set().union(*(r["data"]["temporal"] for r in records)))
    summaries = {"all": {m: group_summary(records, m) for m in metrics}}
    for dimension, values in (("task", TASKS), ("model", MODELS), ("gpu", GPUS)):
        for value in values:
            selected = [r for r in records if r[dimension] == value]
            summaries[f"{dimension}:{value}"] = {m: group_summary(selected, m) for m in metrics}
    write_json(output / "summary.json", summaries)
    case_rows, case_json = [], []
    for r in records:
        serialized = {key: value for key, value in r.items() if key not in {"data", "protocol"}}
        serialized["selected"] = r["protocol"]["selected"]
        serialized["methods"] = {}
        for method in METHODS:
            stats = {m: {**estimate(v["mean"], v["samples"]), "runs": v["run_count"]}
                     for m, v in r["data"][method].items()}
            a, b = r["data"][method]["window_score"], r["data"]["temporal"]["window_score"]
            delta = estimate(a["mean"] - b["mean"], a["samples"] - b["samples"])
            serialized["methods"][method] = {"metrics": stats, "primary_difference_from_temporal": delta}
            case_rows.append({"task": r["task"], "model": r["model"], "gpu": r["gpu"], "method": method,
                              "window_score": a["mean"], "ci95_low": stats["window_score"]["ci95"][0],
                              "ci95_high": stats["window_score"]["ci95"][1], "difference_from_temporal": delta["mean"],
                              "difference_ci95_low": delta["ci95"][0], "difference_ci95_high": delta["ci95"][1],
                              "hmm_states": r["protocol"]["selected"]["hmm"], "block_length": r["protocol"]["selected"]["block"],
                              "validation_runs": 25, "validation_samples": r["protocol"]["validation_samples"],
                              "secondary_32_runs": r["secondary_coverage"]["32"],
                              "secondary_128_runs": r["secondary_coverage"]["128"],
                              "hmm_refit_converged": r["convergence"][-1]["selected_converged"]})
        case_json.append(serialized)
    write_json(output / "case_results.json", case_json)
    with (output / "case_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    group_rows = []
    for group, grouped in summaries.items():
        for metric, summary in grouped.items():
            for method, value in summary["methods"].items():
                group_rows.append({"group": group, "metric": metric, "method": method, "mean": value["mean"],
                                   "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1],
                                   "cases": value["cases"], "runs": value["runs"]})
    with (output / "group_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    return summaries


def save_figure(fig, output, name, pdf):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def draw_reports(output, records, summaries, explanation):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#26364a", "axes.titleweight": "bold"})
    with PdfPages(output / "report.pdf") as pdf:
        fig = plt.figure(figsize=(12, 12))
        fig.text(.07, .94, "Temporal profiles across the full archive", size=23, weight="bold")
        fig.text(.07, .89, "30 cases · 3 tasks · 2 models · 5 GPUs", size=15, color="#587287")
        y = .82
        for paragraph in explanation:
            wrapped = textwrap.fill(paragraph, width=112)
            fig.text(.07, y, wrapped, fontsize=11, va="top", linespacing=1.5)
            y -= .0195 * len(wrapped.splitlines()) + .022
        save_figure(fig, output, "procedure_and_findings", pdf)

        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        fig.subplots_adjust(left=.07, right=.98, top=.76, bottom=.22, wspace=.4)
        fig.text(.07, .93, "Average timing fidelity across all 30 cases", fontsize=22, weight="bold")
        fig.text(.07, .865, "Every case receives equal weight. Lower is better on all three panels.", fontsize=12)
        for ax, metric, title in zip(axes, ("window_score", "distribution_error_sd", "deadline_rate_error_pp"),
                                     ("Primary ordered-window score", "Delay-distribution error / fitting SD", "One-frame miss-rate error (pp)")):
            for i, method in enumerate(METHODS):
                value = summaries["all"][metric]["methods"][method]
                low, high = value["ci95"]
                ax.vlines(i, low, high, color=COLORS[method], lw=2)
                ax.scatter(i, value["mean"], color=COLORS[method], s=65, zorder=3)
            ax.set(title=title, xlim=(-.5, 4.5))
            ax.set_xticks(range(5), [LABELS[m] for m in METHODS], rotation=32, ha="right")
            ax.grid(axis="y", alpha=.2)
        fig.text(.07, .045, "95% intervals jointly resample matched run IDs and generation repeats across all cases. Case weights stay fixed.\nTiming fidelity does not establish which method produces the best trained agent or game return.", color="#5c6c7e", fontsize=10)
        save_figure(fig, output, "overall_comparison", pdf)

        ordered = sorted(records, key=lambda r: (TASKS.index(r["task"]), MODELS.index(r["model"]), GPUS.index(r["gpu"])))
        relative = np.array([[100 * (r["data"][m]["window_score"]["mean"] / r["data"]["temporal"]["window_score"]["mean"] - 1)
                              for m in METHODS] for r in ordered])
        fig, ax = plt.subplots(figsize=(12, 13))
        fig.subplots_adjust(left=.40, right=.88, top=.87, bottom=.11)
        fig.text(.05, .965, "Every environment / model / GPU combination", fontsize=22, weight="bold")
        fig.text(.05, .925, "Primary score change relative to Our Temporal (%) · negative favors the alternative", fontsize=12)
        bound = max(5, float(np.max(np.abs(relative))))
        im = ax.imshow(relative, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_yticks(range(len(ordered)), [f"{TASK_LABELS[r['task']]} / {MODEL_LABELS[r['model']]} / {r['gpu'].upper()}" for r in ordered], fontsize=9)
        ax.set_xticks(range(5), [LABELS[m] for m in METHODS], rotation=20, ha="right")
        for i in range(len(ordered)):
            for j in range(5):
                ax.text(j, i, f"{relative[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(relative[i, j]) > .55 * bound else "#26364a")
        for row in (9.5, 19.5):
            ax.axhline(row, color="white", lw=3)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.04, label="Change in score (%)")
        fig.text(.05, .025, "Descriptive case means; paired uncertainty is available in case_results.csv. No case is selected or omitted based on its score.", fontsize=10)
        save_figure(fig, output, "all_cases_heatmap", pdf)

        groups = [(f"task:{t}", TASK_LABELS[t]) for t in TASKS] + [(f"model:{m}", MODEL_LABELS[m]) for m in MODELS] + [(f"gpu:{g}", g.upper()) for g in GPUS]
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(left=.30, right=.96, top=.79, bottom=.15)
        fig.text(.06, .94, "Does the conclusion hold across settings?", fontsize=22, weight="bold")
        fig.text(.06, .885, "Primary score difference from Our Temporal · negative favors the alternative", fontsize=12)
        for method, offset in (("hmm", -.13), ("block", .13)):
            for i, (key, label) in enumerate(groups):
                v = summaries[key]["window_score"]["paired"][f"{method}_minus_temporal"]
                ax.hlines(i + offset, *v["ci95"], color=COLORS[method], lw=2)
                ax.scatter(v["mean"], i + offset, color=COLORS[method], s=40, label=LABELS[method] if i == 0 else None)
        ax.axvline(0, color="#677889", lw=1, ls="--")
        ax.set_yticks(range(len(groups)), [label for _, label in groups])
        ax.invert_yaxis()
        ax.set_xlabel("Paired difference in standardized window score")
        ax.grid(axis="x", alpha=.2)
        ax.legend(loc="best", frameon=False)
        fig.text(.06, .04, "Equal case weights within each group. Conditional 95% intervals; overlapping groups are not independent experiments.\nThese subgroup comparisons are exploratory, with no multiplicity correction.", fontsize=10, color="#5c6c7e")
        save_figure(fig, output, "subgroup_comparisons", pdf)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.23, wspace=.35)
        fig.text(.08, .94, "Secondary checks on longer timing sequences", fontsize=21, weight="bold")
        fig.text(.08, .875, "Frozen settings selected on 8/16-request windows; longer windows do not affect model selection.", fontsize=11)
        for ax, length in zip(axes, (32, 128)):
            result = summaries["all"][f"secondary_score_{length}"]
            for i, method in enumerate(("hmm", "block")):
                value = result["paired"][f"{method}_minus_temporal"]
                ax.vlines(i, *value["ci95"], color=COLORS[method], lw=2)
                ax.scatter(i, value["mean"], color=COLORS[method], s=65)
            coverage = result["methods"]["temporal"]
            ax.axhline(0, color="#677889", ls="--", lw=1)
            ax.set(title=f"{length} requests · {coverage['cases']}/30 cases · {coverage['runs']}/750 runs",
                   ylabel="Window score difference from Our Temporal", xlim=(-.5, 1.5))
            ax.set_xticks([0, 1], [LABELS["hmm"], LABELS["block"]])
            ax.grid(axis="y", alpha=.2)
        fig.text(.08, .06, "Negative differences favor the alternative. Only runs at least as long as the window contribute.\nCoverage differs between horizons, so these are separate diagnostic comparisons, not directly comparable overall rankings.", color="#5c6c7e", fontsize=10)
        save_figure(fig, output, "long_window_checks", pdf)

    with PdfPages(output / "timing_traces.pdf") as pdf:
        for r in records:
            directory = output / "cases" / r["relative"]
            run_id = r["protocol"]["validation_ids"][0]
            with np.load(directory / "sequences/real.npz") as archive:
                series = {"real": archive[str(run_id)][:300, 0]}
            with np.load(directory / "sequences/generated_00.npz") as archive:
                series.update({m: archive[f"{m}_{run_id}"][:300, 0] for m in METHODS})
            fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True, sharey=True)
            fig.subplots_adjust(left=.21, right=.97, top=.84, bottom=.11, hspace=.25)
            fig.text(.07, .95, TASK_LABELS[r["task"]], fontsize=22, weight="bold")
            fig.text(.07, .90, f"{MODEL_LABELS[r['model']]} · {r['gpu'].upper()} · first listed validation run {run_id} · generation repeat 0", fontsize=12)
            for ax, (method, values) in zip(axes, series.items()):
                ax.plot(np.arange(len(values)), values, color=COLORS[method], lw=1)
                ax.set_ylabel(LABELS[method], rotation=0, ha="right", va="center", fontsize=10)
                ax.grid(axis="y", alpha=.2)
            axes[-1].set_xlabel("Consecutive accepted requests (first 300, or the entire shorter session)")
            fig.text(.07, .035, "Response delay in milliseconds; identical y scale within this case. Synthetic spikes need not line up with measured spikes.\nThis display rule is fixed for every case; it does not select a favorable segment.", color="#5c6c7e", fontsize=10)
            save_figure(fig, directory, "timing_traces", pdf)


def report(output):
    write_json(output / "aggregation_protocol.json", {
        "bootstrap_draws": 2000, "seed": SEED,
        "rule": "Common run-ID and repetition draws across every case and method; equal fixed case weights. Secondary horizons renormalize the same run-cluster draws over eligible runs.",
        "scope": "Conditional on fitted generators and selected settings; no uncertainty over fitting or future tasks/models/GPUs."
    })
    records = collect(output)
    summaries = write_tables(output, records)
    primary = summaries["all"]["window_score"]
    best = min(METHODS, key=lambda m: primary["methods"][m]["mean"])
    failed_refits = [r["relative"] for r in records if not r["convergence"][-1]["selected_converged"]]
    failed_stages = sum(not stage["selected_converged"] for r in records for stage in r["convergence"])
    audits = json.loads((output / "cohort_audit.json").read_text())
    fit_samples = sum(r["fit_samples"] for r in audits)
    validation_samples = sum(r["validation_samples"] for r in audits)
    def paired_text(key):
        v = primary["paired"][key]
        return f"{v['mean']:+.5f} (95% interval {v['ci95'][0]:+.5f} to {v['ci95'][1]:+.5f})"
    explanation = [
        f"Completed all 30 archived combinations: Flappy Bird, Inverted Pendulum and MIKASA InterceptGrabFast; QwenOFT and QwenGR00T; A100, L40S, RTX 3090, RTX 4090 and RTX 5090. These are the complete calibration-cohort combinations, not every task or checkpoint elsewhere on Hugging Face.",
        f"Evidence: 1,500 hardware runs, two episodes per run. The fitting half contains {fit_samples:,} accepted requests; the separate validation half contains {validation_samples:,}. Archived fitting counts match, and means agree within 1e-8 ms. No new hardware measurements or agent training were performed.",
        "Procedure: split each case's 25 fitting runs into 20 for learning and 5 for choosing HMM states (2/3/4) or mean block length (8/16/32/64/128). Refit the chosen method on all 25 fitting runs. Evaluate against 25 separate runs using 10 generation repeats. Fixed, IID and the archived v13 Temporal sampler provide reference methods.",
        "The primary score averages energy and variogram scores at 8 and 16 consecutive requests, standardized with fitting statistics. Lower means closer timing-window distributions. The shortest session has 26 accepted requests; these common horizons retain every run. Longer 32/128-request scores are separate checks with explicit coverage.",
        f"Result: {LABELS[best]} has the lowest equal-case mean primary score. HMM minus Temporal: {paired_text('hmm_minus_temporal')}. Blocks minus Temporal: {paired_text('block_minus_temporal')}. Negative differences favor the alternative.",
        f"HMM has a lower case mean than Temporal in {primary['paired']['hmm_minus_temporal']['lower_mean_cases']}/30 cases; blocks in {primary['paired']['block_minus_temporal']['lower_mean_cases']}/30. HMM minus blocks: {paired_text('hmm_minus_block')}. These counts describe means, not 30 corrected significance tests.",
        "There is a tradeoff: Temporal has lower average delay-distribution error than both alternatives, with paired intervals favoring Temporal. One-frame miss-rate error differences are uncertain. On the L40S subgroup, the primary-score differences also remain uncertain. Neither the sequence result nor these diagnostics establish a universal winner.",
        f"Limits: conditional intervals exclude fitting/selection uncertainty and generalization to unmeasured hardware. {len(failed_refits)}/30 selected final HMM fits missed the declared convergence tolerance; {failed_stages}/120 selected HMM fitting stages did so. All cases remain included. Timing fidelity alone cannot establish better game return or justify claiming Temporal is universally best.",
    ]
    (output / "explanation.txt").write_text("\n\n".join(explanation) + "\n")
    draw_reports(output, records, summaries, explanation)
    headers = "".join(f"<th>{LABELS[m]}</th>" for m in METHODS)
    rows = []
    for r in sorted(records, key=lambda r: (TASKS.index(r["task"]), MODELS.index(r["model"]), GPUS.index(r["gpu"]))):
        cells = "".join(f"<td>{r['data'][m]['window_score']['mean']:.4f}</td>" for m in METHODS)
        winner = min(METHODS, key=lambda m: r["data"][m]["window_score"]["mean"])
        rows.append(f"<tr data-task='{r['task']}' data-model='{r['model']}' data-gpu='{r['gpu']}'>"
                    f"<td><a href='cases/{r['relative']}/index.html'>{TASK_LABELS[r['task']]}</a></td>"
                    f"<td>{MODEL_LABELS[r['model']]}</td><td>{r['gpu'].upper()}</td>{cells}"
                    f"<td>{LABELS[winner]}</td><td>{r['protocol']['selected']['hmm']} / {r['protocol']['selected']['block']}</td></tr>")
    filter_html = ""
    for dimension, values, names in (("task", TASKS, TASK_LABELS), ("model", MODELS, MODEL_LABELS), ("gpu", GPUS, {})):
        options = "".join(f"<option value='{v}'>{names.get(v, v.upper())}</option>" for v in values)
        filter_html += f"<label>{dimension.capitalize()} <select id='{dimension}' onchange='filterRows()'><option value=''>All</option>{options}</select></label> "
    trace_cards = "".join(f"<details><summary>{TASK_LABELS[r['task']]} / {MODEL_LABELS[r['model']]} / {r['gpu'].upper()}</summary>"
                          f"<img loading='lazy' src='cases/{r['relative']}/timing_traces.png' alt='Measured and generated response delays'>"
                          f"<p><a href='cases/{r['relative']}/index.html'>Full case report, fits and paired results</a></p></details>" for r in records)
    source_protocol = json.loads((output / "protocol.json").read_text())
    html_text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal-profile comparison — all 30 cases</title>
<style>body{{font:16px/1.65 system-ui;color:#26364a;max-width:1280px;margin:40px auto;padding:0 24px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:2em}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dce3ea;text-align:right}}td:first-child,th:first-child{{text-align:left}}thead{{background:#edf3f6}}a{{color:#007e86}}.note{{padding:20px;background:#edf5f6;border-left:4px solid #007e86}}.scroll{{overflow-x:auto}}select{{padding:7px;margin:0 14px 12px 4px}}details{{padding:10px;border-bottom:1px solid #dce3ea}}summary{{cursor:pointer;font-weight:600}}small{{color:#617386}}</style>
<h1>Temporal profiles across all 30 archived cases</h1>
<p>3 environments × 2 models × 5 GPUs · 1,500 recorded runs · 10 generation repeats per method and case</p>
<p class="note"><b>{LABELS[best]} has the lowest average primary timing score.</b> This is an offline comparison of generated delay sequences against hardware recordings. It does not measure agent training quality or game return.</p>
<p><a href="report.pdf">Main PDF</a> · <a href="timing_traces.pdf">All 30 timing plots</a> · <a href="case_results.csv">Case results CSV</a> · <a href="group_results.csv">All metrics CSV</a> · <a href="explanation.txt">Plain-language explanation</a> · <a href="protocol.json">Protocol</a> · <a href="report_bundle.zip">Portable report bundle</a></p>
<h2>What we did and found</h2>{''.join('<p>'+html.escape(p)+'</p>' for p in explanation)}
<img src="overall_comparison.png" alt="Overall timing fidelity on three metrics">
<img src="all_cases_heatmap.png" alt="Relative primary scores for every case">
<h2>Inspect every case</h2><p>Primary score: lower is better. H/B gives the selected HMM state count and mean block length. Follow an environment link for per-case plots and evidence.</p>
{filter_html}<div class="scroll"><table><thead><tr><th>Environment</th><th>Model</th><th>GPU</th>{headers}<th>Lowest mean</th><th>H/B</th></tr></thead><tbody id="cases">{''.join(rows)}</tbody></table></div>
<img src="subgroup_comparisons.png" alt="Paired HMM and block differences within task, model and GPU groups">
<img src="long_window_checks.png" alt="Secondary longer-window scores and coverage">
<h2>Side-by-side measured and generated timing</h2><p>Every case uses the first listed validation run (39), generation repeat 0, and the first 300 accepted requests or the entire shorter run. All six traces within a case share one delay axis. Exact spike alignment is not an objective for stochastic profile generators.</p>{trace_cards}
<h2>How to interpret the evidence</h2>
<p>The score uses the distribution of ordered response-delay windows. Energy describes the overall window distribution; variogram emphasizes differences between positions. Their equal-weight combination is a declared objective, not a universal definition of a good deployment profile. All generators retain paired action latency and worker busy time; the primary score evaluates action latency.</p>
<p>Uncertainty uses 2,000 crossed bootstrap draws over whole validation run IDs and generation repetitions. All cases and methods share the same draws to preserve the archive's reused environment and generation seeds. Aggregation keeps equal fixed case weights. Secondary horizons renormalize these same cluster draws over eligible runs. Intervals condition on fitted generators and selected settings. They exclude fitting uncertainty and do not imply that these tasks, models or GPU runs are random independent samples of future deployments. Subgroup and per-case intervals are exploratory and unadjusted for multiplicity. Details are recorded in <a href="aggregation_protocol.json">the aggregation protocol</a>.</p>
<p>The Flappy/QwenOFT/RTX3090 pilot informed the design. A structural audit of all cases preceded fitting and revealed the need for shorter common horizons. No outer validation timing values enter parameter fitting or hyperparameter selection. The primary score here uses 8/16-request windows, so its numerical values must not be spliced into the earlier 8/32/128-window study. Longer windows are supplementary, use only eligible sessions, and never join independent runs.</p>
<p>The HMM has full-covariance Gaussian emissions on standardized log busy time and log positive overhead. Three restarts (7/42/2026) have a fixed 1,000-iteration budget; the best fitting likelihood wins. Nonconvergence is retained and disclosed rather than dropping unfavorable cases. Stationary block resampling uses geometric block lengths, paired timing values, and circular continuation inside a fitting session. It may wrap and replay short source runs; it cannot generate unseen latency magnitudes.</p>
<p>Finite-ensemble score corrections assume independent identically distributed ensemble members. Members here are independent but stratified by recorded session length and episode schedule, so the correction is approximate. Deadline diagnostics count accepted requests. They do not simulate worker capacity, observation admission, dropped requests or policy behavior. Raw hardware replay remains a useful later control-performance reference.</p>
<p>Selected final HMM fits missing tolerance: {html.escape(', '.join(failed_refits) or 'none')}. Full convergence evidence: <a href="case_results.json">case_results.json</a>. All selected-stage failures: {failed_stages}/120.</p>
<h2>Artifacts and reproducibility</h2><p>Profiles: <a href="https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles/tree/{source_protocol['sources']['profile_revision']}">MLL-Lab/LAGEN-profiles @ {source_protocol['sources']['profile_revision']}</a>.<br>Experiment metadata: <a href="https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/{source_protocol['sources']['experiment_revision']}">MLL-Lab/LAGEN-datasets @ {source_protocol['sources']['experiment_revision']}</a>.</p>
<p><a href="cohort_audit.json">Audit of all 1,500 runs</a> · <a href="summary.json">Group summaries and paired differences</a> · <a href="case_results.json">Case summaries and convergence</a>. Each case retains raw generated sequences, run-level metrics, parameter choices, fitted HMM parameters, package versions, checkpoint identity and a code snapshot. The portable bundle excludes the large raw/generated sequence archives; the complete local output retains them.</p>
<script>function filterRows(){{const keys=['task','model','gpu'];document.querySelectorAll('#cases tr').forEach(row=>{{row.hidden=!keys.every(key=>!document.getElementById(key).value||row.dataset[key]===document.getElementById(key).value);}});}}</script></html>"""
    (output / "report.html").write_text(html_text)
    snapshot = output / "code"
    snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in {".py", ".sh", ".txt", ".md"}:
            shutil.copy2(path, snapshot / path.name)
    with zipfile.ZipFile(output / "report_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if not path.is_file() or path.suffix not in {".html", ".json", ".csv", ".txt", ".png", ".pdf", ".py", ".sh", ".md"}:
                continue
            if any(part.startswith(".") for part in relative.parts) or "sequences" in relative.parts or "logs" in relative.parts:
                continue
            bundle.write(path, relative)
    print(f"Saved complete report: {output / 'report.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
