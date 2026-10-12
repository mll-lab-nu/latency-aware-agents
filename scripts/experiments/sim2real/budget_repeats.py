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

from scripts.experiments.sim2real import budget
from scripts.experiments.sim2real.budget_report import run_mean, normalized_mean_error
from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json, write_json_atomic, write_csv, write_json

def random_partitions(campaign, seeds):
    """Vary only validation selection, filtering a common frozen fit ordering."""
    fit_priority = list(range(campaign["total_runs"]))
    random.Random(campaign["split_seed"]).shuffle(fit_priority)
    partitions = {}
    for seed in seeds:
        ids = list(range(campaign["total_runs"]))
        random.Random(seed).shuffle(ids)
        valid = sorted(ids[:campaign["validation_runs"]])
        partitions[str(seed)] = {
            "valid": valid, "fit_order": [run_id for run_id in fit_priority if run_id not in valid],
        }
    return partitions


def prepare_inputs(root, base_root, config):
    """Reuse the completed real pool and pin the original random result as repeat one."""
    campaign = read_json(base_root / "budget.json")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "repeats.json"
    if not manifest_path.exists():
        per_split = (len(campaign["models"]) * len(campaign["fit_sizes"]) *
                     len(campaign["protocol"]["methods"]) * campaign["validation_runs"] * campaign["protocol"]["episodes"])
        manifest = {
            **campaign, **config,
            "previous_campaign": base_root.name,
            "previous_output": str(base_root.resolve()),
            "partitions": random_partitions(campaign, config["seeds"]),
            "fit_priority_seed": campaign["split_seed"],
            "new_episodes": per_split * (len(config["seeds"]) - 1),
            "random_episodes_including_previous": per_split * len(config["seeds"]),
            "metric": "mean across splits of equally weighted OFT/GR00T validation NMAE",
            "band": "sample standard deviation across five splits; descriptive, not a confidence interval",
        }
        write_json_atomic(manifest_path, manifest)
    manifest = read_json(manifest_path)
    for model in manifest["models"]:
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
            for method in manifest["protocol"]["methods"]:
                errors = []
                for model in manifest["models"]:
                    parent, split = (base_root, "random") if seed == manifest["seeds"][0] else (root, str(seed))
                    path = parent / model / "sim" / split / str(count) / method / "runs"
                    simulated = np.mean([run_mean(path / str(run_id)) for run_id in partition["valid"]])
                    real = np.mean([run_mean(base_root / model / "real" / str(run_id))
                                    for run_id in partition["valid"]])
                    error = abs(normalized_mean_error(real, simulated, manifest["sources"][model]["reference_return"]))
                    errors.append(error)
                    cases.append({"seed": seed, "model": model, "fit_runs": count, "method": method,
                                  "real_mean": real, "sim_mean": simulated, "absolute_error_pp": error})
                by_seed.append({"seed": seed, "fit_runs": count, "method": method,
                                "nmae_pp": np.mean(errors)})
    aggregate = []
    for count in manifest["fit_sizes"]:
        for method in manifest["protocol"]["methods"]:
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
    for method in manifest["protocol"]["methods"]:
        points = [r for r in aggregate if r["method"] == method]
        xs = [r["fit_runs"] for r in points]
        means = np.array([r["nmae_mean_pp"] for r in points])
        sd = np.array([r["nmae_sd_pp"] for r in points])
        ax.plot(xs, means, color=colors[method], label=labels[method], marker="o", markersize=4, linewidth=1.8)
        ax.fill_between(xs, np.maximum(0, means - sd), means + sd, color=colors[method], alpha=.15)
    ax.set(title=f"Random validation · {len(manifest['seeds'])} splits", xlabel="Fit runs", ylabel="NMAE (pp)", ylim=(0, None))
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
    parser.add_argument("--stage", choices=("prepare", "fit", "sim", "report"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--worker-python", default=sys.executable)
    args = parser.parse_args(argv)
    root, base_root = args.root.resolve(), args.base_root.resolve()
    if args.stage == "prepare":
        config = load_config(args.config)
        prepare_inputs(root, base_root, config)
    manifest = read_json(root / "repeats.json")
    if args.stage == "prepare":
        return
    if args.stage == "report":
        report = repeated_report(root, base_root)
        return
    if args.stage == "sim" and args.gpu_uuid is None:
        parser.error("sim requires --gpu-uuid")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.stage == "sim":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    output = root / args.model
    partitions = {str(seed): manifest["partitions"][str(seed)] for seed in manifest["seeds"][1:]}
    if args.stage == "fit":
        budget.fit_profiles(output, output / "source", args.model, partitions=partitions,
                            fit_sizes=manifest["fit_sizes"], source_ref=manifest["sources"][args.model])
    else:
        budget.simulate(output, output / "source", args.model, args.worker_python, manifest["parallel_runs"], partitions=partitions, manifest=manifest,
                        combine_settings=manifest["combine_settings"])


if __name__ == "__main__":
    main()
