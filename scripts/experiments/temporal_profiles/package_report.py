"""Package an existing tuned timing study as a plain-language visual report."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.experiments.temporal_profiles.plot import COLORS

METHODS = ("real", "fixed", "iid", "temporal", "hmm", "block")
INK = "#202c3d"
MUTED = "#647084"


def page(title, subtitle, number):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(.07, .93, title, fontsize=22, weight="bold")
    fig.text(.07, .885, subtitle, fontsize=11, color=MUTED)
    fig.text(.07, .035, "LAGEN  /  Offline timing-profile study  /  Flappy Bird · QwenOFT · RTX 3090", fontsize=8, color=MUTED)
    fig.text(.93, .035, str(number), ha="right", fontsize=9, color=MUTED)
    return fig


def paragraph(fig, y, heading, text, *, width=112, fontsize=11):
    fig.text(.07, y, heading, fontsize=13, weight="bold", va="top")
    fig.text(.07, y - .035, textwrap.fill(text, width), fontsize=fontsize,
             va="top", linespacing=1.5)


def overview(protocol, summary, labels):
    results = summary["methods"]
    fig = page("What do the results show?", "Better timing resemblance depends on what we measure. These are timing scores, not game scores.", 1)
    fig.text(.07, .813, "Both alternatives improve the window score. HMM misses rare long delays.",
             fontsize=13, weight="bold", color=COLORS["temporal"])
    ax = fig.add_axes([.18, .54, .73, .20])
    for i, method in enumerate(("iid", "hmm", "block")):
        value = summary["difference_from_temporal"][method]["window_score"]
        ax.hlines(i, *value["ci95"], color=COLORS[method], lw=3)
        ax.plot(value["mean"], i, "o", color=COLORS[method], ms=8)
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.set_yticks(range(3), [labels[m] for m in ("iid", "hmm", "block")])
    ax.invert_yaxis()
    ax.set_ylim(2.5, -.5)
    ax.set_title("Window-score difference from our Temporal profile", loc="left", pad=12)
    ax.set_xlabel("Negative values favor the alternative; zero means equal score")
    ax.grid(axis="x")
    for metric, rect, title, ylabel, factor in (
        ("lag1", [.10, .18, .34, .23], "Do neighboring delays vary together?", "Adjacent-delay correlation", 1),
        ("miss_1_frame", [.59, .18, .34, .23], "How often is a response very late?", "Responses per 10,000 taking >1 frame", 10000),
    ):
        ax = fig.add_axes(rect)
        reference = results["real"][metric]
        ax.axhspan(*(factor * np.array(reference["ci95"])), color=INK, alpha=.08)
        ax.axhline(factor * reference["mean"], color=INK, ls="--", lw=1)
        for i, method in enumerate(METHODS):
            if metric not in results[method]:
                ax.text(i, .04, "N/A", ha="center", fontsize=9, color=MUTED)
                continue
            value = results[method][metric]
            ax.vlines(i, *(factor * np.array(value["ci95"])), color=COLORS[method], lw=2)
            ax.plot(i, value["mean"] * factor, "o", color=COLORS[method], ms=7)
        ax.set_xticks(range(6), [labels[m] for m in METHODS], rotation=22, ha="right", fontsize=9)
        ax.set(title=title, ylabel=ylabel, xlim=(-.5, 5.5))
        ax.grid(axis="y")
    fig.text(.07, .077, f"Match the measured rate, rather than minimize it. One frame = {protocol['frame_ms']:.2f} ms.\n"
             "Bars: 95% bootstrap intervals. Dashed lines / shaded bands: measured reference. Fixed has no defined correlation.",
             fontsize=9, color=MUTED, linespacing=1.4)
    return fig


def trace_figure(output, protocol, labels):
    run = protocol["validation_ids"][0]
    with np.load(output / "sequences" / "real.npz") as real, np.load(output / "sequences" / "generated_00.npz") as generated:
        series = {"real": real[str(run)][:1000, 0]}
        series.update({m: generated[f"{m}_{run}"][:1000, 0] for m in METHODS[1:]})
    n = min(map(len, series.values()))
    low = min(v[:n].min() for v in series.values())
    high = max(max(v[:n].max() for v in series.values()), protocol["frame_ms"])
    fig = page("What do the profiles look like?", f"First {n:,} accepted requests of validation run {run} · generated repetition 0 · identical axes", 3)
    axes = fig.subplots(6, 1, sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.96, top=.82, bottom=.16, hspace=.55)
    x = np.arange(1, n + 1)
    for ax, method in zip(axes, METHODS):
        ax.plot(x, series[method][:n], color=COLORS[method], lw=.7)
        ax.axhline(protocol["slow_cutoff_ms"], color=MUTED, ls=":", lw=.7)
        ax.axhline(protocol["frame_ms"], color=INK, ls="--", lw=.8)
        ax.set_title(labels[method], loc="left", fontsize=10, pad=3, color=COLORS[method])
        ax.set_ylim(low - 2, high + 2)
        ax.set_ylabel("ms", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].set(xlim=(1, n), xlabel="Consecutive accepted request number (not wall-clock time)")
    fig.text(.07, .075, f"Dotted line: fitting-data p95 ({protocol['slow_cutoff_ms']:.2f} ms). Dashed line: one frame ({protocol['frame_ms']:.2f} ms).\n"
             "This is one illustration, chosen by a fixed rule. Generated spikes need not occur at the same positions as real spikes.",
             fontsize=9, color=MUTED, linespacing=1.4)
    return fig


def write_report(output):
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    results = summary["methods"]
    states, length = protocol["selected"]["hmm"], protocol["selected"]["block"]
    labels = {"real": "Measured", "fixed": "Fixed", "iid": "IID", "temporal": "Our Temporal",
              "hmm": f"HMM ({states})", "block": f"Blocks ({length})"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "text.color": INK,
                         "axes.labelcolor": INK, "grid.alpha": .15})
    fit_count = sum(r["accepted"] for r in audit["fit"])
    valid_count = sum(r["accepted"] for r in audit["validation"])
    steps = [
        ("1. Start with actual hardware measurements",
         f"We use 50 recorded sessions: 25 for fitting ({fit_count:,} accepted responses) and 25 for validation ({valid_count:,}). Each session contains two episodes. These are ordered response times measured while QwenOFT plays Flappy Bird on an RTX 3090."),
        ("2. Choose alternative settings using fitting sessions only",
         f"Split the 25 fitting sessions into 20 for learning and 5 for choosing settings. Try HMMs with 2, 3 or 4 hidden states and resampled blocks averaging 8, 16, 32, 64 or 128 requests. The selected settings are {states} states and {length} requests per block."),
        ("3. Refit, generate, then compare with separate real sessions",
         "Refit the selected alternatives using all 25 fitting sessions. Keep the original Fixed, IID and v13 Temporal profiles unchanged. Generate 10 repetitions of 25 session lengths per method: 1,250 synthetic sequences across five generators."),
        ("4. Judge patterns of delays, not the position of each spike",
         "Compare windows of 8, 32 and 128 consecutive requests using energy and variogram scores. Also check adjacent-delay correlation, slow-spell counts and frame-deadline crossings. Average sessions equally; bootstrap whole sessions and generation repetitions 2,000 times for uncertainty."),
    ]
    methods = [
        ("Fixed", "Use the same fitted mean response time for every request."),
        ("IID", "Draw fresh timing pairs independently from the fitted distribution; no temporal memory."),
        ("Our Temporal", "Use the existing v13 fitted model of ordinary periods and rare slow bursts."),
        (f"HMM ({states} states)", "Learn hidden timing regimes and their transitions; generate positive worker times and overhead jointly."),
        (f"Blocks (mean {length})", "Copy random stretches from fitting traces, preserving their local order; concatenate them into new traces."),
    ]
    real_rate = results["real"]["miss_1_frame"]["mean"] * 10000
    temporal_rate = results["temporal"]["miss_1_frame"]["mean"] * 10000
    hmm_rate = results["hmm"]["miss_1_frame"]["mean"] * 10000
    block_rate = results["block"]["miss_1_frame"]["mean"] * 10000
    findings = [
        ("Both alternatives improve the declared window score",
         f"Temporal scores {results['temporal']['window_score']['mean']:.4f}; HMM scores {results['hmm']['window_score']['mean']:.4f}; blocks score {results['block']['window_score']['mean']:.4f}. Lower is better. Both paired differences from Temporal have 95% intervals below zero. This supports an advantage for this score in this case, conditional on the fitted models."),
        ("Blocks reproduce adjacent-delay dependence well",
         f"Measured correlation is {results['real']['lag1']['mean']:.3f}; blocks give {results['block']['lag1']['mean']:.3f}, HMM {results['hmm']['lag1']['mean']:.3f}, and Temporal {results['temporal']['lag1']['mean']:.3f}. Blocks retain real stretches of timing variation. Their lowest window-score estimate does not establish a significant advantage over HMM."),
        ("Our Temporal captures rare deadline crossings better than HMM",
         f"Per 10,000 accepted responses, hardware produces {real_rate:.2f} responses longer than one frame. Temporal gives {temporal_rate:.2f}, blocks {block_rate:.2f}, and HMM only {hmm_rate:.2f}. HMM therefore produces about 1/{real_rate/hmm_rate:.0f} of the measured rate. A simulator should reproduce this rate; making it near zero is not automatically better."),
    ]
    limits = [
        "One task, model and GPU case. No policy training, game-return evaluation or request-admission replay was performed.",
        "Validation timings were excluded from fitting and tuning but had already been viewed in the earlier pilot. This is exploratory evidence, not a fresh blinded confirmation.",
        "The 95% intervals condition on fitted models and selected settings. They exclude fitting/selection uncertainty and have no multiple-comparison correction.",
        "Finite-ensemble score corrections assume IID ensemble members. Here independently generated sessions have different recorded lengths and episode schedules, so those corrections are approximate.",
        "Block length 128 was the largest candidate tested; it is not proven optimal. Resampling observed traces also limits extrapolation to unseen timing behavior.",
        "Only accepted requests are scored. Zero observed two-frame crossings does not imply zero underlying risk. Ten generated repetitions do not create ten new sets of real hardware measurements.",
    ]
    paper_note = ("The paper's main sim-to-real comparison measures error in normalized game returns across 30 cases. "
                  "This study instead asks how closely generated timing sequences resemble measured timings in one case. "
                  "These answer different questions. The current results do not establish which generator trains the best policy, "
                  "and they do not support claiming that our Temporal profile beats every alternative.")
    recommendation = ("Before making a broad paper claim, repeat the frozen procedure across more existing hardware/model/task cases. "
                      "A later replay experiment can test game-return calibration. If the claim is about training a better agent, "
                      "train matched agents with the chosen profiles and evaluate them under the same measured timing conditions. "
                      "Those experiments have not been run here.")

    with PdfPages(output / "report.pdf") as pdf:
        fig = overview(protocol, summary, labels)
        pdf.savefig(fig)
        fig.savefig(output / "results_overview.png", dpi=180)
        fig.savefig(output / "results_overview.pdf")
        plt.close(fig)

        fig = page("What did we do?", "Fit timing generators to existing data; keep a separate set of real sessions for evaluation.", 2)
        for y, (heading, text) in zip((.82, .66, .50, .34), steps):
            paragraph(fig, y, heading, text)
        fig.text(.07, .15, "What the score means", fontsize=13, weight="bold", va="top")
        fig.text(.07, .115, "Energy: resemblance of whole windows. Variogram: differences between positions within a window.\n"
                 "The combined score gives equal weight to both at all three lengths. It has no unit and is not game reward.",
                 fontsize=11, linespacing=1.5, va="top")
        pdf.savefig(fig)
        fig.savefig(output / "report_procedure.png", dpi=150)
        plt.close(fig)

        fig = trace_figure(output, protocol, labels)
        pdf.savefig(fig)
        fig.savefig(output / "tuned_timing_sequences.png", dpi=180)
        fig.savefig(output / "tuned_timing_sequences.pdf")
        plt.close(fig)

        fig = page("Interpretation and limits", "What this evidence supports, and how it relates to the paper.", 4)
        for y, (heading, text) in zip((.82, .68, .54), findings):
            paragraph(fig, y, heading, text, width=125, fontsize=10)
        paragraph(fig, .38, "Relationship to the paper", paper_note, width=125, fontsize=10)
        paragraph(fig, .24, "Scope of the evidence", " ".join(limits[:4]), width=138, fontsize=9)
        fig.text(.07, .078, "Detailed limitations, exact source revisions, all tables and reproducibility links are in report.html and protocol.json.",
                 fontsize=9, color=MUTED)
        pdf.savefig(fig)
        fig.savefig(output / "report_interpretation.png", dpi=150)
        plt.close(fig)

    with (output / "summary_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "method", "metric", "mean", "ci95_low", "ci95_high", "generation_sd"])
        for comparison in ("methods", "difference_from_temporal"):
            for method, metrics in summary[comparison].items():
                for metric, value in metrics.items():
                    writer.writerow([comparison, method, metric, value["mean"], *value["ci95"], value["generation_sd"]])

    rows = []
    for method in METHODS:
        cells = []
        for metric, factor, digits in (("window_score", 1, 4), ("lag1", 1, 3), ("miss_1_frame", 10000, 2)):
            if metric not in results[method]:
                cells.append("—")
            else:
                value = results[method][metric]
                cells.append(f"<b>{value['mean']*factor:.{digits}f}</b><br><small>[{value['ci95'][0]*factor:.{digits}f}, {value['ci95'][1]*factor:.{digits}f}]</small>")
        rows.append(f"<tr><th>{labels[method]}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    source_url = f"https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles/tree/{protocol['profile_revision']}/{protocol['case']}"
    experiment_url = f"https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/{protocol['experiment_revision']}"
    sections = lambda items: "".join(f"<h3>{escape(title)}</h3><p>{escape(body)}</p>" for title, body in items)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timing profiles: visual report</title>
<style>
body{{font:17px/1.65 system-ui,sans-serif;color:{INK};max-width:1120px;margin:40px auto;padding:0 24px}}
h1{{font-size:40px;line-height:1.15;max-width:900px}}h2{{margin-top:2.4em}}h3{{margin-bottom:.3em;font-size:20px}}
p{{max-width:950px}}a{{color:#007d82}}img{{width:100%;height:auto}}small,.meta{{color:{MUTED}}}
.takeaway{{padding:20px 26px;background:#eef6f5;border-left:5px solid #007d82;font-size:20px}}
table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{text-align:right;padding:12px;border-bottom:1px solid #dde2e8}}
th:first-child,td:first-child{{text-align:left}}.table-scroll{{overflow-x:auto}}code{{font-size:13px;overflow-wrap:anywhere}}
details{{margin:20px 0;padding:16px;background:#f5f7f9}}summary{{cursor:pointer;font-weight:600}}
@media print{{body{{max-width:none;font-size:11pt}}details{{display:block}}img{{break-inside:avoid}}}}
</style></head><body>
<p class="meta">LAGEN · Flappy Bird · QwenOFT · RTX 3090 · offline timing study</p>
<h1>Can other temporal profiles reproduce real delays?</h1>
<p class="takeaway">Yes, in this case both tuned alternatives improve the whole-window score over our Temporal profile.
Block resampling also captures neighboring-delay dependence well. HMM seriously underestimates rare long delays.</p>
<p><a href="report.pdf">Read the 4-page PDF</a> · <a href="report_bundle.zip">Download the report bundle</a> ·
<a href="summary_results.csv">Summary data (CSV)</a> · <a href="index.html">Detailed statistical report</a></p>
<p>This report packages the completed stronger study. It does not refit models or change the saved results.
Fixed and IID are context controls; the new temporal alternatives are the HMM and block resampling.</p>
<img src="results_overview.png" alt="Paired window-score differences, adjacent-delay correlation and rare deadline-crossing rates with confidence intervals">
<h2>The numbers in plain language</h2>
<div class="table-scroll"><table><thead><tr><th>Profile</th><th>Window score<br><small>lower is better</small></th>
<th>Adjacent-delay correlation<br><small>match measured value</small></th><th>Responses longer than 1 frame<br><small>per 10,000; match measured rate</small></th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p>Brackets show 95% bootstrap intervals. Every real session has equal weight. Rates per 10,000 are scaled frequencies,
not counts from a separate 10,000-request experiment. A constant Fixed sequence has undefined correlation.
The whole-window score compares a generator to real data, so no real-vs-itself score was estimated.</p>
{sections(findings)}
<h2>What we actually did</h2>{sections(steps)}
<p>Normalization and tuning use fitting data only. The original Temporal artifact uses schema v13.
The HMM models log worker time and log pre-worker overhead jointly, preserving positive timing values.
Three fitting restarts use seeds 7, 42 and 2026. The study seed is {protocol['base_seed']}.
Measured worker times and response times remain paired in all saved sequence arrays.</p>
<h2>What each profile means</h2>{sections(methods)}
<img src="tuned_timing_sequences.png" alt="Six traces on identical axes: measured, fixed, IID, Temporal, three-state HMM and mean-128 block resampling">
<p>The measured sequence is our reference sample, not a deterministic future to predict. Generated spikes need not
line up request by request. We compare their distribution, persistence and frequency instead of pointwise error.</p>
<h2>How settings were selected</h2>
<img src="model_selection.png" alt="Inner tuning scores selecting a three-state HMM and mean block length 128">
<p>Energy scores compare distributions of entire windows. Variogram scores emphasize differences between positions.
The objective averages both scores at 8, 32 and 128 requests with equal weights. This chosen objective is not a universal
ranking for every deployment purpose. The common slow-response cutoff is {protocol['slow_cutoff_ms']:.3f} ms (fitting p95);
the deadline is {protocol['frame_ms']:.3f} ms at {protocol['env_fps']:.4f} frames/second. These are different cutoffs.</p>
<h2>What this means for the paper</h2><p>{escape(paper_note)}</p><p>{escape(recommendation)}</p>
<h2>Limits to keep with these results</h2><ul>{''.join('<li>'+escape(item)+'</li>' for item in limits)}</ul>
<details><summary>Scoring and uncertainty details</summary>
<p>Each real session supplies 64 sampled windows per length; these may overlap. Each generated ensemble window comes
from an independent synthetic session. Final scoring uses 25 ensemble members per repetition, with 10 repetitions.
The bootstrap resamples whole real sessions and generation repetitions, using shared draws for paired comparisons.</p>
<p>Paired HMM-minus-Temporal and block-minus-Temporal intervals lie below zero. The IID-minus-Temporal interval crosses
zero. No paired HMM-versus-block significance claim is made. See <a href="summary.json">summary.json</a> for exact intervals
and <a href="protocol.json">protocol.json</a> for score definitions and seed rules.</p></details>
<h2>Where everything is saved</h2>
<p><code>{escape(str(output))}</code></p>
<ul><li><a href="report.pdf">report.pdf</a>, this HTML and PNG/PDF figures: explanation and visual results.</li>
<li><a href="summary_results.csv">summary_results.csv</a> and <a href="summary.json">summary.json</a>: means, uncertainty and paired comparisons.</li>
<li><a href="run_metrics.csv">run_metrics.csv</a>: all session/repetition measurements. Real statistics repeat across generation repetitions; they represent 25 real sessions.</li>
<li><a href="protocol.json">protocol.json</a>, <a href="audit.json">audit.json</a>, <a href="selection/selection.json">selection/selection.json</a>,
<a href="hmm_refit.json">hmm_refit.json</a>: settings, data counts, model selection and learned parameters.</li>
<li><code>sequences/real.npz</code> and <code>sequences/generated_00.npz</code> through <code>generated_09.npz</code>:
all saved real validation and generated timing pairs. Retained in the full local output folder, excluded from the lightweight ZIP.</li>
<li><a href="code/README.md">code/README.md</a> and source snapshot: reproduction commands and implementation.
<a href="analysis.log">analysis.log</a>: original execution log.</li></ul>
<p>The ZIP includes reports, figures, metrics, protocol, selection, code and original small fitted-profile files.
It excludes raw request JSONL files and sequence arrays. It is a review bundle; complete source measurements are retained locally at
<code>{escape(protocol['source_directory'])}</code> and available from the pinned source below.</p>
<h2>Sources and artifact identity</h2>
<p><a href="{source_url}">Hugging Face measured traces and fitted profile</a><br>
<code>MLL-Lab/LAGEN-profiles@{protocol['profile_revision']}</code></p>
<p><a href="{experiment_url}">Hugging Face archived experiment</a><br>
<code>MLL-Lab/LAGEN-datasets@{protocol['experiment_revision']}</code></p>
<p>Measured model bundle: <code>{escape(protocol['model_bundle'])}</code><br>
Archived model revision: <code>{protocol['model_revision']}</code><br>
Benchmark base commit: <code>{protocol['repository_commit']}</code>. Experiment additions are preserved in <code>code/</code>.</p>
<p>Paper comparison source: <code>overleaf_latest_paper/sections/appendix/sim2real/main.tex</code> and
<code>figures/fig4_sim2real_calibration/heldout_30cases_20261002/README.md</code> in the local paper checkout.</p>
</body></html>'''
    (output / "report.html").write_text(html)
    for name in ("package_report.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, output / "code" / name)
    with ZipFile(output / "report_bundle.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for directory in (output, output / "selection", output / "code"):
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in {".html", ".png", ".pdf", ".csv", ".json", ".log", ".py", ".sh", ".txt", ".md"}:
                    bundle.write(path, Path(output.name) / path.relative_to(output))
        source = Path(protocol["source_directory"])
        bundle.write(source / "experiment.json", Path(output.name) / "source" / "experiment.json")
        for path in sorted((source / "profiles" / protocol["case"] / "profile").glob("*.json")):
            bundle.write(path, Path(output.name) / "source" / "profile" / path.name)
    print(f"Saved report.html, report.pdf, figures, summary_results.csv and report_bundle.zip in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Existing stronger-study output directory")
    write_report(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
