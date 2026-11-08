"""Offline report for the inner-tuned temporal generator comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.temporal_profiles.plot import COLORS, save


def render(output):
    protocol = json.loads((output / "protocol.json").read_text())
    selection = json.loads((output / "selection" / "selection.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    results = summary["methods"]
    horizon_text = ", ".join(str(h) for h in protocol["horizons"])
    case_label = protocol["case"].split("/cases/")[-1].replace("/", " · ")
    labels = {"fixed": "Fixed", "iid": "IID", "temporal": "Our Temporal",
              "hmm": f"HMM · {protocol['selected']['hmm']} states",
              "block": f"Blocks · L={protocol['selected']['block']}"}
    methods = list(labels)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "text.color": "#283445",
                         "grid.alpha": .18})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=.08, right=.97, top=.75, bottom=.22, wspace=.3)
    fig.text(.08, .94, "Choose settings using fitting data only", fontsize=21, weight="bold")
    fig.text(.08, .875, "20 sessions learn parameters · 5 separate sessions select settings · lower window score is better", fontsize=11)
    for ax, method in zip(axes, ("hmm", "block")):
        candidates = [c for c in selection["candidates"] if c["method"] == method]
        x = [c["setting"] for c in candidates]
        y = [c["score"] for c in candidates]
        ax.plot(x, y, "o-", color=COLORS[method], lw=2)
        for c in candidates:
            ax.scatter([c["setting"]] * len(c["run_scores"]), [r["window_score"] for r in c["run_scores"]],
                       color=COLORS[method], alpha=.25, s=22)
        chosen = next(c for c in candidates if c["setting"] == protocol["selected"][method])
        ax.scatter(chosen["setting"], chosen["score"], marker="*", s=200, color=COLORS[method], edgecolor="white", zorder=4)
        ax.set(title="HMM state count" if method == "hmm" else "Mean resampled block length",
               xlabel="Number of states" if method == "hmm" else "Requests per block", ylabel="Inner tuning score")
        if method == "block":
            ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.grid()
    fig.text(.08, .07, f"Faint dots: five tuning-session scores. Stars: selected settings.\nSelection uses the mean of standardized energy and variogram scores at {horizon_text} requests.", color="#647084", fontsize=9)
    save(fig, output, "model_selection")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.08, right=.97, top=.68, bottom=.22, wspace=.3)
    fig.text(.08, .94, "Do whole sequences resemble real timing?", fontsize=21, weight="bold")
    fig.text(.08, .88, "25 validation sessions · 10 generated repetitions · lower scores are better", fontsize=11)
    for ax, metric, title in zip(axes, ("energy", "variogram"), ("Energy score: overall window distribution", "Variogram score: temporal dependence")):
        for method in methods:
            vals = [results[method][f"{metric}_{h}"] for h in protocol["horizons"]]
            x, y = np.array(protocol["horizons"]), np.array([v["mean"] for v in vals])
            low, high = np.array([v["ci95"] for v in vals]).T
            ax.plot(x, y, "o-", label=labels[method], color=COLORS[method], lw=1.8)
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.10)
        ax.set(title=title, xlabel="Consecutive requests per window", ylabel="Standardized score", xscale="log")
        ax.set_xticks(protocol["horizons"], [str(h) for h in protocol["horizons"]])
        ax.grid()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.07, .84), ncol=3, frameon=False)
    fig.text(.08, .065, "Shading: 95% bootstrap intervals over whole real runs and generation repetitions, conditional on fitted models.\nGenerated windows come from independent simulated timing sequences. No game or queue simulation is involved.", color="#647084", fontsize=9)
    save(fig, output, "window_scores")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.13, wspace=.31, hspace=.6)
    fig.text(.09, .955, "Timing accuracy and action deadlines", fontsize=21, weight="bold")
    fig.text(.09, .91, f"One frame = {protocol['frame_ms']:.2f} ms · all method settings frozen before final evaluation", fontsize=11)
    panels = [("window_score", "Whole-window score", "Lower is better", 1),
              ("lag1", "Dependence between neighboring requests", "Lag-1 correlation", 1),
              ("miss_1_frame", "Responses taking more than one frame", "% of accepted requests", 100),
              ("slow_count_error_8", "Slow-response counts in 8-request windows", "Distribution error (requests); lower is better", 1)]
    for ax, (metric, title, ylabel, factor) in zip(axes.flat, panels):
        if metric in results["real"] and not metric.startswith("slow_count_error"):
            real = results["real"][metric]
            ax.axhspan(real["ci95"][0] * factor, real["ci95"][1] * factor, color=COLORS["real"], alpha=.10)
            ax.axhline(real["mean"] * factor, color=COLORS["real"], ls="--", lw=1.2)
        for i, method in enumerate(methods):
            if metric not in results[method]:
                continue
            value = results[method][metric]
            low, high = value["ci95"]
            ax.vlines(i, low * factor, high * factor, color=COLORS[method], lw=2)
            ax.plot(i, value["mean"] * factor, "o", color=COLORS[method], ms=7)
        ax.set(title=title, ylabel=ylabel, xlim=(-.5, 4.5))
        ax.set_xticks(range(5), [labels[m] for m in methods], rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y")
    fig.text(.09, .035, "Dark dashed line / band: measured hardware mean / interval. Fixed has undefined correlation.\nDeadline checks count accepted requests only; they do not predict dropped requests or game return.", color="#647084", fontsize=9)
    save(fig, output, "fidelity_and_deadlines")

    metric_names = {"window_score": "Combined window score", "distribution_error_ms": "Delay-distribution error (ms)",
                    "acf_error": "Autocorrelation error", "slow_count_error_32": "Slow-count error, 32 requests",
                    "slow_count_error_128": "Slow-count error, 128 requests", "miss_1_frame": "More than one frame (%)",
                    "miss_2_frames": "More than two frames (%)", "max_miss_streak": "Longest miss streak per run"}
    table = []
    for metric, name in metric_names.items():
        cells = []
        for method in methods:
            if metric not in results[method]:
                cells.append("—")
            else:
                v = results[method][metric]
                factor = 100 if metric.startswith("miss_") else 1
                cells.append(f"{factor*v['mean']:.4f}<br><small>[{factor*v['ci95'][0]:.4f}, {factor*v['ci95'][1]:.4f}]</small>")
        table.append(f"<tr><th>{name}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    deltas = []
    for method in methods:
        if method == "temporal":
            continue
        d = summary["difference_from_temporal"][method]["window_score"]
        deltas.append(f"<tr><th>{labels[method]} minus Our Temporal</th><td>{d['mean']:.5f}</td><td>[{d['ci95'][0]:.5f}, {d['ci95'][1]:.5f}]</td></tr>")
    html = f"""<!doctype html><html><meta charset="utf-8"><title>Tuned timing profile comparison</title>
<style>body{{font:16px/1.6 system-ui;color:#283445;max-width:1200px;margin:40px auto;padding:0 24px}}h1{{line-height:1.2}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dde2e8;text-align:right}}th:first-child{{text-align:left}}small{{color:#647084}}a{{color:#007d82}}.note{{background:#f0f5f6;padding:16px;border-left:4px solid #007d82}}</style>
<h1>Stronger temporal-profile comparison</h1>
<p>{case_label} · existing Hugging Face measurements.</p>
<p class="note">20 fitting sessions learn parameters; 5 fitting sessions select settings. Selected alternatives are refitted on all 25 fitting sessions, then evaluated against 25 separate real sessions using 10 generated repetitions. This is an offline timing study, not VLA training or game evaluation.</p>
<p><a href="protocol.json">Frozen protocol and sources</a> · <a href="selection/selection.json">Selection results</a> · <a href="hmm_refit.json">Final HMM fit</a> · <a href="summary.json">Summary and paired comparisons</a> · <a href="run_metrics.csv">All run-level results</a></p>
<p>Selected HMM: <b>{protocol['selected']['hmm']} states</b>. Selected mean block length: <b>{protocol['selected']['block']} requests</b>. The original v13 Temporal profile remains unchanged.</p>
<img src="model_selection.png" alt="Inner-only HMM and block-length selection">
<img src="window_scores.png" alt="Energy and variogram scores across three window lengths">
<p>Energy scores assess the distribution of whole ordered windows. Variogram scores focus on differences between positions. The selection objective gives equal weight to both scores at each of {horizon_text} requests. Normalization uses fitting data only. This declared objective is not a universal ranking for every deployment purpose.</p>
<img src="fidelity_and_deadlines.png" alt="Fidelity and deadline statistics with uncertainty">
<table><thead><tr><th>Metric</th>{''.join('<th>'+labels[m]+'</th>' for m in methods)}</tr></thead><tbody>{''.join(table)}</tbody></table>
<p>Brackets show 95% intervals. Scores/error metrics favor lower values; deadline rates and streak lengths should match hardware, rather than simply be minimized. Each real run receives equal weight.</p>
<h2>Paired difference in the declared window score</h2>
<table><thead><tr><th>Comparison</th><th>Difference</th><th>95% interval</th></tr></thead><tbody>{''.join(deltas)}</tbody></table>
<p>Negative values favor the alternative for this score. Bootstrap draws pair methods on the same real runs and generation repetitions. Intervals condition on the fitted models and selection, and do not include fitting uncertainty or a correction for multiple comparisons.</p>
<h2>Interpretation and limits</h2>
<p>{protocol['prior_inspection']} This page covers one model, task and hardware case.</p>
<p>Each generated ensemble window comes from its own independent generated session. Observed windows may overlap within a real session; uncertainty resamples whole sessions, not individual windows. The reported intervals also resample generation repetitions. They are not bounds across different trained VLAs.</p>
<p>The finite-ensemble corrections use the standard IID-ensemble formulas. Generated sessions here retain different recorded length and episode schedules, so the corrections are approximate for this stratified setup. Zero observed deadline misses do not establish zero underlying miss probability.</p>
<p>Slow-response cutoff: {protocol['slow_cutoff_ms']:.3f} ms, the fitting p95. Deadline durations use the archived calibrated clock ({protocol['env_fps']:.4f} Hz). The analysis uses accepted-request order and does not run a worker-capacity scheduler or measure policy return. Both observation-to-action latency and worker busy time remain stored as paired values in the generated artifacts.</p>
<p>PDFs: <a href="model_selection.pdf">Selection</a> · <a href="window_scores.pdf">Window scores</a> · <a href="fidelity_and_deadlines.pdf">Fidelity/deadlines</a>.</p>
</html>"""
    (output / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)


if __name__ == "__main__":
    main()
