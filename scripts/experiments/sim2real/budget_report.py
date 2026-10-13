"""Run-level conditional uncertainty, latency diagnostics and fit-budget plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latency_bench.core.clock import ms_to_raw_frames
from latency_bench.latency.burst_profile import build_burst_samples, build_hidden_regime_burst_model_from_sessions
from latency_bench.utils.io import load_jsonl_records, read_json, write_csv, write_json


def run_mean(path):
    return np.mean([row["episode_return_env"] for row in load_jsonl_records(path / "episode_metrics.jsonl")])


def normalized_mean_error(real_mean, simulated_mean, reference):
    """Signed percentage-point error against the frozen reference return."""
    return (simulated_mean - real_mean) * (100 / reference)


def comparison(real, simulated, reference, *, paired, seed, repetitions):
    """Compare means, preserving valid pairs or independently resampling fit references."""
    real, simulated = np.asarray(real), np.asarray(simulated)
    sim_indices = np.random.default_rng(seed).integers(0, len(simulated), size=(repetitions, len(simulated)))
    real_indices = sim_indices if paired else np.random.default_rng(seed + 1).integers(
        0, len(real), size=(repetitions, len(real)),
    )
    scale = 100 / reference
    errors = normalized_mean_error(real[real_indices].mean(axis=1), simulated[sim_indices].mean(axis=1), reference)
    difference = normalized_mean_error(real.mean(), simulated.mean(), reference)
    return {
        "real_mean": real.mean(), "sim_mean": simulated.mean(),
        "sim_mean_pct": simulated.mean() * scale,
        "signed_error_pp": difference, "absolute_error_pp": abs(difference),
        "signed_ci_low_pp": np.quantile(errors, .025), "signed_ci_high_pp": np.quantile(errors, .975),
        "absolute_ci_low_pp": np.quantile(abs(errors), .025),
        "absolute_ci_high_pp": np.quantile(abs(errors), .975),
    }, abs(errors)


def latency_diagnostics(root, manifest):
    rows = []
    for model in manifest["models"]:
        output = root / model
        fps = read_json(output / "source/calibration/calibration.json")["selected_env_fps"]
        for run_id in range(manifest["total_runs"]):
            path = output / "real" / str(run_id)
            attempts = load_jsonl_records(path / "observation_attempts.jsonl")
            samples = build_burst_samples(attempts, measure_session_id=run_id)
            burst = build_hidden_regime_burst_model_from_sessions([samples])
            values = np.array([s["observation_to_action_latency_ms"] for s in samples])
            frames = np.array([ms_to_raw_frames(v, fps) for v in values])
            dwells = burst["burst_dwell_lengths"]
            rows.append({
                "model": model, "run_id": run_id,
                "collection": "original" if run_id in manifest["reused_run_ids"] else "new",
                "return": run_mean(path), "admitted_count": len(values),
                "mean_ms": values.mean(), "p95_ms": np.quantile(values, .95), "p99_ms": np.quantile(values, .99),
                "above_one_frame_pct": 100 * np.mean(frames > 1),
                "above_two_frames_pct": 100 * np.mean(frames > 2),
                "burst_count": len(dwells), "burst_dwell_mean_records": np.mean(dwells) if dwells else None,
                "burst_dwell_max_records": max(dwells) if dwells else None,
            })
    return rows


def _timing_paths(output):
    return [*output.glob("timings/*.json"), *output.glob("profiles/*/*/timing.json"),
            *output.glob("sim/*/*/*/timings/*.json"), *output.glob("runtime/setups/*.json"),
            *output.glob("verification/timings/*.json")]


def make_report(root, *, timing_root=None):
    manifest = read_json(root / "budget.json")
    rows, draws = [], {}
    for model in manifest["models"]:
        (root / model / "SIM_DONE").read_bytes()
        for split, partition in manifest["splits"].items():
            for count in manifest["fit_sizes"]:
                for method in manifest["protocol"]["methods"]:
                    path = root / model / "sim" / split / str(count) / method / "runs"
                    sim = [run_mean(path / str(i)) for i in partition["valid"]]
                    for reference_split, ids in (("fit", partition["fit_order"][:count]), ("valid", partition["valid"])):
                        real = [run_mean(root / model / "real" / str(i)) for i in ids]
                        # Both curves share the same sim estimates and bootstrap draws.
                        # Only validation has matching real/sim environment seeds.
                        result, bootstrap = comparison(
                            real, sim, manifest["sources"][model]["reference_return"],
                            paired=reference_split == "valid", seed=manifest["split_seed"], repetitions=manifest["bootstrap_repetitions"],
                        )
                        row = {"model": model, "split": split, "fit_runs": count,
                               "method": method, "reference_split": reference_split,
                               "real_runs": len(ids), "sim_runs": len(sim), **result}
                        rows.append(row)
                        draws[model, split, count, method, reference_split] = bootstrap
    aggregate = []
    for split in manifest["splits"]:
        for count in manifest["fit_sizes"]:
            for method in manifest["protocol"]["methods"]:
                for target in ("fit", "valid"):
                    selected = [r for r in rows if (r["split"], r["fit_runs"], r["method"], r["reference_split"]) == (split, count, method, target)]
                    bootstrap = np.mean([draws[m, split, count, method, target] for m in manifest["models"]], axis=0)
                    aggregate.append({"split": split, "fit_runs": count, "method": method, "reference_split": target,
                                      "n_cases": len(manifest["models"]), "nmae_pp": np.mean([r["absolute_error_pp"] for r in selected]),
                                      "ci_low_pp": np.quantile(bootstrap, .025), "ci_high_pp": np.quantile(bootstrap, .975)})
    report = root / "report"
    report.mkdir(exist_ok=True)
    write_csv(report / "case_errors.csv", rows)
    write_csv(report / "nmae.csv", aggregate)
    write_csv(report / "real_latency_by_run.csv", latency_diagnostics(root, manifest))
    costs = []
    timing_root = root if timing_root is None else timing_root
    for model in manifest["models"]:
        for path in _timing_paths(timing_root / model):
            data = read_json(path)
            coordinates = {"real": ("run_ids",), "fit": ("split", "fit_runs",),
                           "sim": ("split", "fit_runs", "method", "run_ids"),
                           "sim_setup": (), "verify": ()}[data["stage"]]
            costs.append({"model": data["model"], "stage": data["stage"], "seconds": data["seconds"],
                          "gpu_hours": data["seconds"] * data["gpu_count"] / 3600,
                          "source": path.relative_to(timing_root).as_posix(),
                          **{key: data[key] for key in coordinates}})
    write_csv(report / "costs.csv", costs)
    cumulative = []
    for model in manifest["models"]:
        own = [c for c in costs if c["model"] == model]
        collected_cost = sum(c["gpu_hours"] for c in own if c["stage"] == "real")
        for split in manifest["splits"]:
            for count in manifest["fit_sizes"]:
                prefix_cost = sum(c["gpu_hours"] for c in own if c["stage"] == "real"
                                  and c["run_ids"][0] in manifest["splits"]["original"]["fit_order"][:count])
                replay_cost = sum(c["gpu_hours"] for c in own if c["stage"] == "sim"
                                  and c["split"] == split and c["fit_runs"] <= count)
                cumulative.append({"model": model, "split": split, "fit_runs": count,
                                   "collected_real_runs": len(range(manifest["total_runs"])), "full_collection_gpu_hours": collected_cost,
                                   "original_prefix_equivalent_gpu_hours": prefix_cost if split == "original" else None,
                                   "cumulative_replay_gpu_hours": replay_cost})
    write_csv(report / "cumulative_costs.csv", cumulative)
    stability = []
    for model in manifest["models"]:
        for split in manifest["splits"]:
            for method in manifest["protocol"]["methods"]:
                selected = sorted([r for r in rows if (r["model"], r["split"], r["method"], r["reference_split"]) == (model, split, method, "valid")], key=lambda r: r["fit_runs"])
                final = selected[-1]
                candidate = next((r["fit_runs"] for i, r in enumerate(selected[:-1])
                                  if all(abs(v["absolute_error_pp"] - final["absolute_error_pp"]) <= manifest["tolerance_pp"]
                                         and abs(v["sim_mean_pct"] - final["sim_mean_pct"]) <= manifest["tolerance_pp"]
                                         for v in selected[i:])), None)
                precise = all((r["signed_ci_high_pp"] - r["signed_ci_low_pp"]) / 2 <= manifest["tolerance_pp"]
                              for r in selected if candidate is not None and r["fit_runs"] >= candidate)
                stability.append({"model": model, "split": split, "method": method,
                                  "descriptive_plateau_from_runs": candidate,
                                  "budget_supported_at_2pp_precision": candidate is not None and precise})
    write_json(report / "stability.json", stability)
    write_json(report / "protocol.json", {
        "experiment": manifest,
        "bootstrap_repetitions": manifest["bootstrap_repetitions"], "bootstrap_unit": "run means; valid real/sim paired, fit real/sim independently resampled; common draws across models and methods",
        "interval_scope": "conditional on fitted profile and fixed zero-latency reference",
        "random_splits": 1, "tolerance_pp": manifest["tolerance_pp"],
        "fit_curves": "shared sim mean versus all real fit runs at each size; a mean-calibration diagnostic",
        "cost_scope": "new successful real acquisition and simulated replay, including setup and warmup",
    })
    plot_curves(report, aggregate, manifest)
    (report / "README.md").write_text(
        "# Flappy fit-budget results\n\n"
        f"Models: {', '.join(manifest['models'])}; fit runs: {manifest['fit_sizes']}; "
        f"{manifest['validation_runs']} validation runs; {manifest['protocol']['episodes']} episodes/run.\n\n"
        "See `case_errors.csv` for signed errors, `nmae.csv` for equally weighted two-case NMAE, "
        "`real_latency_by_run.csv` for collection drift and tails, `costs.csv` and `cumulative_costs.csv` for measured costs, "
        "and `stability.json` for descriptive 2 pp plateau candidates.\n\n"
        "`valid_original.png/pdf` and `valid_random.png/pdf` plot validation NMAE against fit runs, "
        "with four method curves and equal OFT/GR00T weights.\n\n"
        "Each point uses one fixed five-run simulation for both real references. "
        "Validation bootstrap is paired; fit-reference bootstrap independently resamples its real and sim runs. "
        "Intervals condition on each fitted profile. "
        "Five validation runs and one random partition do not establish a universal budget. "
        "An uncertain plateau is not evidence that more fitting data cannot help.\n",
        encoding="utf-8",
    )
    (root / "REPORT_DONE").touch()
    return report


def plot_curves(report, aggregate, manifest):
    colors = {"fixed": "#777777", "normal": "#3474a4", "iid": "#c48530", "temporal": "#a64768"}
    labels = {"fixed": "Fixed", "normal": "Normal", "iid": "IID", "temporal": "Temporal"}
    titles = {"original": "Fixed validation runs", "random": "Random validation runs"}
    for split in manifest["splits"]:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for method in manifest["protocol"]["methods"]:
            selected = sorted(
                [row for row in aggregate if row["split"] == split
                 and row["method"] == method and row["reference_split"] == "valid"],
                key=lambda row: row["fit_runs"],
            )
            ax.plot([row["fit_runs"] for row in selected], [row["nmae_pp"] for row in selected],
                    color=colors[method], marker="o", markersize=4, linewidth=1.8, label=labels[method])
        ax.set(title=titles[split], xlabel="Fit runs", ylabel="NMAE (pp)", ylim=(0, None))
        ax.set_xticks(manifest["fit_sizes"])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        fig.text(.5, .015, "OFT + GR00T · equal model weights", ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(report / f"valid_{split}.png", dpi=240)
        fig.savefig(report / f"valid_{split}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path)
    args = parser.parse_args()
    report = make_report(args.root, timing_root=args.timing_root)
    manifest = read_json(args.root / "budget.json")
