"""Four additional validation partitions and the five-split Flappy NMAE figure."""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latency_bench.cross_device import budget
from latency_bench.cross_device.budget_report import run_mean
from latency_bench.cross_device.study import EPISODES, METHODS
from latency_bench.cross_device.study_publication import StudyPublication
from latency_bench.utils.io import read_json, write_json_atomic, write_csv, write_json

RANDOM_SEEDS = (budget.SPLIT_SEED, 20260930, 20261001, 20261002, 20261003)


def random_partitions():
    """Vary only validation selection, filtering a common frozen fit ordering."""
    fit_priority = list(budget.ALL_RUN_IDS)
    random.Random(budget.SPLIT_SEED).shuffle(fit_priority)
    partitions = {}
    for seed in RANDOM_SEEDS:
        ids = list(budget.ALL_RUN_IDS)
        random.Random(seed).shuffle(ids)
        valid = sorted(ids[:5])
        partitions[str(seed)] = {
            "valid": valid, "fit_order": [run_id for run_id in fit_priority if run_id not in valid],
        }
    return partitions


def prepare_inputs(root, base_root):
    """Reuse the completed real pool and pin the original random result as repeat one."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "repeats.json"
    if not manifest_path.exists():
        manifest = {
            "previous_campaign": base_root.name,
            "previous_report_revision": read_json(base_root / "uploads/report.json")["revision"],
            "previous_sim_revisions": {
                model: read_json(base_root / model / "uploads/sim_complete.json")["revision"]
                for model in budget.MODELS
            },
            "seeds": list(RANDOM_SEEDS), "partitions": random_partitions(),
            "fit_sizes": list(budget.FIT_SIZES), "methods": list(METHODS),
            "models": list(budget.MODELS), "episodes_per_run": EPISODES,
            "fit_priority_seed": budget.SPLIT_SEED,
            "new_episodes": 1920, "random_episodes_including_previous": 2400,
            "parallel_runs": 10, "combine_settings": True,
            "metric": "mean across splits of equally weighted OFT/GR00T validation NMAE",
            "band": "sample standard deviation across five splits; descriptive, not a confidence interval",
        }
        write_json_atomic(manifest_path, manifest)
    manifest = read_json(manifest_path)
    for model in budget.MODELS:
        source = base_root / model
        (source / "SIM_DONE").read_bytes()
        output = root / model
        output.mkdir(exist_ok=True)
        for name in ("real", "source"):
            destination = output / name
            if not destination.exists():
                destination.symlink_to(source / name, target_is_directory=True)
        shutil.copy2(source / "REAL_DONE", output / "REAL_DONE")
    return manifest


def repeated_report(root, base_root):
    """Average per-split NMAE after absolute error and equal model weighting."""
    manifest = read_json(root / "repeats.json")
    for model in manifest["models"]:
        (root / model / "SIM_DONE").read_bytes()
    cases, by_seed = [], []
    for seed in manifest["seeds"]:
        partition = manifest["partitions"][str(seed)]
        for count in manifest["fit_sizes"]:
            for method in manifest["methods"]:
                errors = []
                for model in manifest["models"]:
                    parent, split = (base_root, "random") if seed == manifest["seeds"][0] else (root, str(seed))
                    path = parent / model / "sim" / split / str(count) / method / "runs"
                    simulated = np.mean([run_mean(path / str(run_id)) for run_id in partition["valid"]])
                    real = np.mean([run_mean(base_root / model / "real" / str(run_id))
                                    for run_id in partition["valid"]])
                    error = 100 * abs(simulated - real) / budget.SOURCES[model]["reference_return"]
                    errors.append(error)
                    cases.append({"seed": seed, "model": model, "fit_runs": count, "method": method,
                                  "real_mean": real, "sim_mean": simulated, "absolute_error_pp": error})
                by_seed.append({"seed": seed, "fit_runs": count, "method": method,
                                "nmae_pp": np.mean(errors)})
    aggregate = []
    for count in manifest["fit_sizes"]:
        for method in manifest["methods"]:
            values = [row["nmae_pp"] for row in by_seed
                      if row["fit_runs"] == count and row["method"] == method]
            aggregate.append({"fit_runs": count, "method": method, "n_splits": len(values),
                              "nmae_mean_pp": np.mean(values), "nmae_sd_pp": np.std(values, ddof=1)})
    report = root / "report"
    report.mkdir(exist_ok=True)
    write_csv(report / "case_errors.csv", cases)
    write_csv(report / "random_nmae_by_seed.csv", by_seed)
    write_csv(report / "random_nmae.csv", aggregate)
    write_json(report / "protocol.json", manifest)
    colors = {"fixed": "#777777", "normal": "#3474a4", "iid": "#c48530", "temporal": "#a64768"}
    labels = {"fixed": "Fixed", "normal": "Normal", "iid": "IID", "temporal": "Temporal"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in manifest["methods"]:
        points = [r for r in aggregate if r["method"] == method]
        xs = [r["fit_runs"] for r in points]
        means = np.array([r["nmae_mean_pp"] for r in points])
        sd = np.array([r["nmae_sd_pp"] for r in points])
        ax.plot(xs, means, color=colors[method], label=labels[method], marker="o", markersize=4, linewidth=1.8)
        ax.fill_between(xs, np.maximum(0, means - sd), means + sd, color=colors[method], alpha=.15)
    ax.set(title="Random validation · 5 splits", xlabel="Fit runs", ylabel="NMAE (pp)", ylim=(0, None))
    ax.set_xticks(manifest["fit_sizes"])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5, .015, "Mean ± SD across splits · OFT + GR00T equally weighted", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(report / f"valid_random.{extension}", dpi=240)
    plt.close(fig)
    (root / "REPORT_DONE").touch()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "sim", "report"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--model", choices=budget.MODELS)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--worker-python", default=sys.executable)
    args = parser.parse_args(argv)
    root, base_root = args.root.resolve(), args.base_root.resolve()
    manifest = read_json(root / "repeats.json")
    prefix = f"sim2real-budget/flappy/{root.name}"
    publication = StudyPublication(root, prefix, defer_uploads=True)
    publication.publish("context", [root / "repeats.json"])
    if args.stage == "report":
        report = repeated_report(root, base_root)
        publication.publish("report", [report, root / "REPORT_DONE"])
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    output = root / args.model
    publication = StudyPublication(output, f"{prefix}/{args.model}", defer_uploads=True)
    partitions = {str(seed): manifest["partitions"][str(seed)] for seed in manifest["seeds"][1:]}
    if args.stage == "fit":
        budget.fit_profiles(output, output / "source", args.model, publication, partitions=partitions)
    else:
        budget.simulate(output, output / "source", args.model, args.worker_python, publication,
                        manifest["parallel_runs"], partitions=partitions,
                        combine_settings=manifest["combine_settings"])


if __name__ == "__main__":
    main()
