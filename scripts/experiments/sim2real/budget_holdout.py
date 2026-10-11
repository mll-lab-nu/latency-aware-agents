"""Evaluate a 28-session profile against a frozen 27-session holdout."""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.sim2real import budget, runtime, replay
from scripts.experiments.sim2real.budget_report import run_mean
from scripts.experiments.sim2real.publication import StudyPublication
from latency_bench.core.config import load_config
from latency_bench.executors.factory import build_executor
from latency_bench.utils.io import read_json, write_csv, write_json_atomic


def prepare(root, base_root, config):
    """Freeze the split and reference existing real data without copying traces."""
    campaign = read_json(base_root / "budget.json")
    ids = list(range(campaign["total_runs"]))
    random.Random(config["split_seed"]).shuffle(ids)
    count = config["fit_runs"]
    manifest = {
        **campaign, **config,
        "partition": {"fit_order": ids[:count], "valid": sorted(ids[count:])},
        "expected_sim_episodes": (len(ids[count:]) * campaign["protocol"]["episodes"] *
                                  len(campaign["models"]) * len(campaign["protocol"]["methods"])),
        "base_root": str(base_root),
        "inference": "conditional on frozen split and profiles; previously inspected real pool",
    }
    path = root / "holdout.json"
    if path.exists():
        manifest = read_json(path)
    else:
        write_json_atomic(path, manifest)
    for model in manifest["models"]:
        output = root / model
        output.mkdir(exist_ok=True)
        source = base_root / model
        (source / "REAL_DONE").read_bytes()
        for name in ("real", "source"):
            destination = output / name
            if not destination.exists():
                destination.symlink_to(source / name, target_is_directory=True)
        shutil.copy2(source / "REAL_DONE", output / "REAL_DONE")
    return manifest


def run_path(output, method, run_id, fit_runs):
    return output / "sim" / str(fit_runs) / method / "runs" / str(run_id)


def simulation_jobs(output, base, fps, manifest):
    """Evaluate each method once on all held-out environment seeds."""
    for method in manifest["protocol"]["methods"]:
        config = runtime.run_config(
            base, output, inference_device="cuda:0", simulator_device=None, env_fps=fps,
            run_id=0, method=method, profile_path=output / "profiles/holdout" / str(manifest["fit_runs"]) / "profile.json",
            latency_seed=0, protocol=manifest["protocol"],
        )
        for run_id in manifest["partition"]["valid"]:
            seed = manifest["latency_seed_base"] + run_id
            yield run_id, run_path(output, method, run_id, manifest["fit_runs"]), config, seed


def simulate(output, model, manifest, publication, worker_python):
    """Replay the held-out sessions at the configured parallelism and inference batch cap."""
    base, _, fps = runtime.runtime_base(output / "source", worker_python)
    base["policy"]["action_noise_seeded"] = model == "qwengr00t"
    slots = manifest["parallel_runs"]
    runtime.record_execution_contract(output, "replay", {"base": base, "env_fps": fps, "protocol": manifest})
    config = runtime.run_config(
        base, output / "runtime", inference_device="cuda:0", simulator_device=None, env_fps=fps,
        run_id=0, method="temporal", profile_path=output / "profiles/holdout" / str(manifest["fit_runs"]) / "profile.json", latency_seed=0, protocol=manifest["protocol"],
    )
    config['evaluation'].update(episodes=manifest['protocol']['episodes'] * slots)
    config["executor"]["parallel_envs"] = slots
    config["executor"]["inference_batch_size"] = manifest["inference_batch_size"][model]
    runtime.prepare_replay_runtime(config)
    write_json_atomic(output / "runtime/config.json", config)
    executor = build_executor(config, inference_devices=["cuda:0"])
    with runtime.warmed_replay_executor(executor):
        jobs = []
        for job in simulation_jobs(output, base, fps, manifest):
            path = job[1]
            if (path / "DONE").exists():
                publication.publish(path.relative_to(output).as_posix(), [path])
                continue
            jobs.append(job)
            if len(jobs) == slots:
                replay.replay_settings(executor, jobs, output, publication, protocol=manifest["protocol"])
                jobs.clear()
        if jobs:
            replay.replay_settings(executor, jobs, output, publication, protocol=manifest["protocol"])
    (output / "SIM_DONE").touch()
    publication.publish("sim_complete", [output / "SIM_DONE"])


def summarize(real, simulated, reference, indices):
    """Estimate return agreement by resampling whole real/sim run pairs."""
    scale = 100 / reference
    predictions = simulated
    differences = (predictions - real) * scale
    bootstrap = differences[indices].mean(axis=1)
    runs = len(real)
    return {
        "real_mean": real.mean(), "sim_mean": predictions.mean(),
        "real_mean_pct": real.mean() * scale, "sim_mean_pct": predictions.mean() * scale,
        "real_se_pp": real.std(ddof=1) / np.sqrt(runs) * scale,
        "sim_run_se_pp": predictions.std(ddof=1) / np.sqrt(runs) * scale,
        "signed_error_pp": differences.mean(), "absolute_error_pp": abs(differences.mean()),
        "signed_ci_low_pp": np.quantile(bootstrap, .025),
        "signed_ci_high_pp": np.quantile(bootstrap, .975),
    }, abs(bootstrap)


def report(root, manifest):
    """Export held-out calibration and equal-model normalized return errors."""
    valid = manifest["partition"]["valid"]
    indices = np.random.default_rng(manifest["bootstrap_seed"]).integers(
        0, len(valid), size=(manifest["bootstrap_samples"], len(valid)),
    )
    cases, runs, draws = [], [], {}
    for model in manifest["models"]:
        output = root / model
        (output / "SIM_DONE").read_bytes()
        real = np.array([run_mean(output / "real" / str(run_id)) for run_id in valid])
        for method in manifest["protocol"]["methods"]:
            simulated = np.array([run_mean(run_path(output, method, run_id, manifest["fit_runs"])) for run_id in valid])
            summary, boot = summarize(real, simulated, manifest["sources"][model]["reference_return"], indices)
            cases.append({"model": model, "fit_runs": manifest["fit_runs"], "method": method, **summary})
            draws[model, method] = boot
            for run_id, real_value, sim_value in zip(valid, real, simulated):
                runs.append({"model": model, "method": method, "run_id": run_id,
                             "real_mean": real_value, "sim_mean": sim_value})
    aggregate = []
    for method in manifest["protocol"]["methods"]:
        values = [r["absolute_error_pp"] for r in cases if r["method"] == method]
        boot = np.mean([draws[model, method] for model in manifest["models"]], axis=0)
        aggregate.append({"method": method, "fit_runs": manifest["fit_runs"], "nmae_pp": np.mean(values),
                          "bootstrap_se_pp": boot.std(ddof=1)})
    destination = root / "report"
    write_csv(destination / "case_errors.csv", cases)
    write_csv(destination / "run_returns.csv", runs)
    write_csv(destination / "nmae.csv", aggregate)
    write_json_atomic(destination / "protocol.json", manifest)
    colors = {"fixed": "#777777", "normal": "#3474a4", "iid": "#c48530", "temporal": "#a64768"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharex=True, sharey=True)
    primary = cases
    limits = [value for row in primary for value in (
        row["real_mean_pct"] - row["real_se_pp"], row["real_mean_pct"] + row["real_se_pp"],
        row["sim_mean_pct"] - row["sim_run_se_pp"], row["sim_mean_pct"] + row["sim_run_se_pp"],
    )]
    lower, upper = min(0, min(limits) - 5), max(100, max(limits) + 5)
    for ax, method in zip(axes, manifest["protocol"]["methods"]):
        ax.plot([lower, upper], [lower, upper], color="#aaaaaa", linestyle="--")
        for model, marker in zip(manifest["models"], ("o", "s")):
            row = next(row for row in primary if row["model"] == model and row["method"] == method)
            ax.errorbar(row["real_mean_pct"], row["sim_mean_pct"], xerr=row["real_se_pp"],
                        yerr=row["sim_run_se_pp"], marker=marker, color=colors[method], capsize=3, label=model)
        ax.set(title=method.title(), xlabel="Held-out real return (%)", xlim=(lower, upper), ylim=(lower, upper))
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Simulated return (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{manifest['fit_runs']} fit / {len(valid)} held-out runs · mean ± SE · one simulation per real run")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"heldout_calibration.{extension}", dpi=220)
    plt.close(fig)
    (root / "REPORT_DONE").touch()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "sim", "report"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--worker-python", default=sys.executable)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        config = load_config(args.config)
        prepare(root, args.base_root.resolve(), config)
    manifest = read_json(root / "holdout.json")
    prefix = f"sim2real-budget/{manifest['task']}/{root.name}"
    if args.stage == "prepare":
        StudyPublication(root, prefix, defer_uploads=True).publish("context", [root / "holdout.json"])
        return
    if args.stage == "report":
        destination = report(root, manifest)
        StudyPublication(root, prefix, defer_uploads=True).publish("report", [destination, root / "REPORT_DONE"])
        return
    if args.stage == "sim" and args.gpu_uuid is None:
        parser.error("sim requires --gpu-uuid")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.stage == "sim":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["LATENCY_BENCH_ENV_BACKEND_START_METHOD"] = "spawn"
    output = root / args.model
    publication = StudyPublication(output, f"{prefix}/{args.model}", defer_uploads=True)
    if args.stage == "fit":
        budget.fit_profiles(output, output / "source", args.model, publication,
                            partitions={"holdout": manifest["partition"]}, fit_sizes=[manifest["fit_runs"]],
                            source_ref=manifest["sources"][args.model])
    else:
        simulate(output, args.model, manifest, publication, args.worker_python)


if __name__ == "__main__":
    main()
