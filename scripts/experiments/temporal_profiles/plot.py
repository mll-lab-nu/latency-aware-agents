"""Render shared-axis trace panels and held-out distribution diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"real": "Real hardware", "fixed": "Fixed mean", "iid": "IID",
          "temporal": "Our Temporal · v13", "hmm": "Two-state HMM", "block": "Block resampling · L=32"}
COLORS = {"real": "#202C3D", "fixed": "#8B929B", "iid": "#BD7B2D",
          "temporal": "#007D82", "hmm": "#8B5AA3", "block": "#C25A56"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def render(output, series, metrics, acfs, bursts, audit, provenance):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#283445", "text.color": "#283445",
                         "axes.titleweight": "bold", "grid.alpha": .18})
    methods = list(series)
    threshold = audit["slow_threshold_action_ms"]
    n = min(1000, min(len(arrays[0]) for arrays in series.values()))
    displayed = [arrays[0][:n, 0] for arrays in series.values()]
    low, high = min(x.min() for x in displayed), max(x.max() for x in displayed)
    padding = .07 * (high - low)
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.875, bottom=.115, hspace=.40)
    fig.text(.09, .965, "What do the timing profiles generate?", fontsize=22, weight="bold")
    fig.text(.09, .935, "Flappy Bird  /  QwenOFT  /  RTX 3090", fontsize=13)
    fig.text(.09, .912, f"First {n:,} accepted requests of validation run {provenance['display_run']} · identical axes", color="#647084")
    for ax, method, values in zip(axes, methods, displayed):
        x = np.arange(1, n + 1)
        ax.plot(x, values, color=COLORS[method], lw=.85)
        ax.axhline(threshold, color="#647084", ls=(0, (3, 3)), lw=.8)
        ax.scatter(x[values > threshold], values[values > threshold], s=7, color=COLORS[method], zorder=3)
        ax.set_title(LABELS[method], loc="left", fontsize=11, color=COLORS[method], pad=5)
        ax.set_ylim(low - padding, high + padding)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.grid(axis="y")
    axes[-1].set_xlabel("Consecutive accepted request number")
    axes[-1].set_xlim(1, n)
    fig.text(.09, .025, f"Dashed line: fitting-data 95th percentile ({threshold:.2f} ms). Dots: responses above that cutoff.\n"
             "Synthetic bursts need not line up with real bursts. The window was fixed before fitting; all-run diagnostics are separate.",
             fontsize=9, color="#647084", linespacing=1.5)
    save(fig, output, "timing_sequences")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.12, wspace=.28, hspace=.45)
    fig.text(.08, .96, "Do the generated patterns match held-out hardware?", fontsize=20, weight="bold")
    fig.text(.08, .921, f"25 validation runs · {audit['validation_samples']:,} real accepted requests · action-response latency", fontsize=11)
    for method in methods:
        values = np.concatenate([x[:, 0] for x in series[method]])
        color = COLORS[method]
        lw = 2.2 if method == "real" else 1.4
        q = np.linspace(0, 1, 801)
        axes[0, 0].plot(np.quantile(values, q), q, color=color, lw=lw, label=LABELS[method])
        tail_q = np.linspace(.90, 1, 401)
        axes[0, 1].plot(100 * tail_q, np.quantile(values, tail_q), color=color, lw=lw)
        if np.isfinite(acfs[method]).all():
            axes[1, 0].plot(np.arange(1, len(acfs[method]) + 1), acfs[method], color=color, lw=lw)
        if len(bursts[method]):
            x = np.unique(bursts[method])
            y = [np.mean(bursts[method] >= length) for length in x]
            axes[1, 1].step(x, y, where="post", color=color, lw=lw)
    axes[0, 0].set(title="Typical delays and their distribution", xlabel="Response latency (ms)", ylabel="Fraction of responses at or below")
    axes[0, 1].set(title="The slowest 10% of responses", xlabel="Percentile", ylabel="Response latency (ms)")
    axes[1, 0].set(title="Dependence between successive responses", xlabel="Request separation (lag)", ylabel="Within-run autocorrelation")
    axes[1, 0].axhline(0, color="#8B929B", lw=.8)
    axes[1, 1].set(title="How long do slow spells last?", xlabel="Consecutive responses above fitting p95", ylabel="Fraction of spells at least this long", xscale="log", yscale="log")
    for ax in axes.flat:
        ax.grid()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .885), ncol=3, frameon=False, fontsize=10)
    fig.text(.08, .035, "All methods use the same fitting traces; real curves use reserved validation sessions. Fixed has no slow spells and undefined autocorrelation.\n"
             "Slow spells use one common diagnostic cutoff, not the internal burst labels of any model. These are descriptive timing checks, not game scores.",
             fontsize=9, color="#647084", linespacing=1.5)
    save(fig, output, "timing_diagnostics")

    columns = [("mean_ms", "Mean (ms)"), ("p99_ms", "p99 (ms)"),
               ("slow_fraction", "Slow %"), ("next_slow_given_slow", "Next slow after slow (%)"),
               ("wasserstein_ms_vs_real", "Distribution error (ms)"),
               ("acf_rmse_vs_real", "Correlation error")]
    rows = []
    for method in methods:
        cells = []
        for key, _ in columns:
            value = metrics[method][key]
            cells.append("—" if value is None else f"{value * (100 if key in {'slow_fraction', 'next_slow_given_slow'} else 1):.3f}")
        rows.append("<tr><th>" + LABELS[method] + "</th>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
    source_url = f"https://huggingface.co/datasets/MLL-Lab/LAGEN-profiles/tree/{provenance['profile_revision']}/{provenance['case']}"
    closest_acf = min((m for m in methods if m not in {"real", "fixed"}),
                      key=lambda m: metrics[m]["acf_rmse_vs_real"])
    worker_spikes = json.loads((output / "worker_spike_metrics.json").read_text())
    worker_rows = []
    for method, values in worker_spikes["methods"].items():
        next_slow = values["next_slow_given_slow"]
        conditional = "—" if next_slow is None else f"{100 * next_slow:.2f}%"
        worker_rows.append(f"<tr><th>{LABELS[method]}</th><td>{100 * values['slow_fraction']:.3f}%</td><td>{conditional}</td></tr>")
    html = f"""<!doctype html><html><meta charset="utf-8"><title>Temporal profile pilot</title>
<style>body{{font:16px/1.6 system-ui;color:#283445;max-width:1200px;margin:45px auto;padding:0 24px}}h1{{line-height:1.2}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #dde2e8}}th:first-child{{text-align:left}}.note{{background:#f0f5f6;padding:18px;border-left:4px solid #007d82}}a{{color:#007d82}}</style>
<h1>Temporal profile comparison: first offline pilot</h1>
<p>Flappy Bird · QwenOFT · RTX 3090. {audit['fitting_samples']:,} accepted fitting requests and {audit['validation_samples']:,} accepted validation requests, in 25 independent sessions per split.</p>
<p class="note">This checks timing generators. No VLA weights were changed, no game evaluation was run, and no method is assumed to win. Timing fidelity alone does not establish better policy learning.</p>
<p>The lowest autocorrelation error among these generators is {LABELS[closest_acf]} ({metrics[closest_acf]['acf_rmse_vs_real']:.4f}); Our Temporal is {metrics['temporal']['acf_rmse_vs_real']:.4f}. Real lag-1 correlation is {metrics['real']['acf_lag1']:.3f}, versus {metrics['temporal']['acf_lag1']:.3f} for Our Temporal. This is evidence for one case and one generated realization per run, not a conclusion across deployments.</p>
<p><a href="{source_url}">Pinned Hugging Face measurement source</a> · <a href="provenance.json">Protocol</a> · <a href="audit.json">Data audit</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="hmm_fit.json">HMM fit</a></p>
<img src="timing_sequences.png" alt="Six timing sequence panels with identical axes">
<p>Each synthetic method generates a fresh sequence. Burst positions are not expected to align with the measured trace. The displayed run and window were chosen by a fixed rule, before fitting.</p>
<img src="timing_diagnostics.png" alt="Distribution, tail, autocorrelation and slow-spell diagnostics">
<table><thead><tr><th>Method</th>{''.join('<th>'+label+'</th>' for _, label in columns)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Distribution error is the Wasserstein distance from held-out response times; correlation error is RMSE over lags 1–64. Lower values mean closer agreement for that diagnostic. No composite winner or significance test is used.</p>
<p>HMM: two full-covariance Gaussian states on log busy time and log pre-worker overhead, three fitting restarts. Block resampling: geometric blocks with mean length 32, resampling timing pairs together within fitting sessions. Both are initial baselines, not exhaustively tuned.</p>
<p>The common slow cutoff is the fitting response-time p95 ({threshold:.3f} ms). Spells touching session boundaries are included as observed lengths; no spell or correlation pair crosses a session boundary. The simulation samplers preserve their original episode-state behavior.</p>
<h2>Rarer worker spikes: the sampler's own threshold</h2>
<p>This separate check uses the archived Temporal definition of a slow worker response: busy time above {worker_spikes['threshold_ms']:.3f} ms. Fixed is excluded here because its existing busy-time rounding is a different capacity model.</p>
<table><thead><tr><th>Method</th><th>Worker spikes (%)</th><th>Next spike after spike (%)</th></tr></thead><tbody>{''.join(worker_rows)}</tbody></table>
<p><a href="worker_spike_metrics.json">Full worker-spike diagnostics</a>. This threshold identifies rarer events than the main plots' p95 cutoff.</p>
<p>These plots use accepted-request order. They do not replay request admission or worker capacity. The archived Fixed sampler uses rounded busy time, so response latency is the primary comparison here.</p>
<p>Exports: <a href="timing_sequences.pdf">Sequence PDF</a> · <a href="timing_diagnostics.pdf">Diagnostics PDF</a>. Full generated pairs are in sequences.npz.</p>
</html>"""
    (output / "index.html").write_text(html)


def main():
    import argparse
    from scripts.experiments.temporal_profiles.compare import describe
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    provenance = json.loads((output / "provenance.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())["methods"]
    with np.load(output / "sequences.npz") as archive:
        series = {method: [archive[f"{method}_{run}"] for run in provenance["validation_run_ids"]] for method in LABELS}
    acfs, bursts = {}, {}
    for method in LABELS:
        _, acfs[method], bursts[method] = describe([x[:, 0] for x in series[method]], [x[:, 0] for x in series["real"]], audit["slow_threshold_action_ms"])
    render(output, series, metrics, acfs, bursts, audit, provenance)


if __name__ == "__main__":
    main()
