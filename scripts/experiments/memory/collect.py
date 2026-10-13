"""Collect memory teacher datasets and merge probe shards."""

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import subprocess
import sys

from latency_bench.core.config import load_config, merge_dicts
from latency_bench.data.rollout_probe import merge_probe_states, write_probe_state
from latency_bench.utils.io import read_json, write_yaml
from latency_bench.utils.paths import REPO_ROOT


def _run_config(recipe):
    experiment = recipe["dataset_experiment"]
    config = {key: value for key, value in recipe.items() if key != "dataset_experiment"}
    job = experiment["teacher_job"]
    if job is not None:
        with Path(job["manifest"]).open(newline="") as stream:
            row = next(row for row in csv.DictReader(stream, delimiter="\t")
                       if row["latency_fixed_frames"] == str(job["latency_fixed_frames"])
                       and row["seed"] == str(job["seed"]))
        config = merge_dicts(load_config(row["eval_config"]), config)
        config["policy"].update(source="local", root=row["train_artifact_dir"], checkpoint=None)
    return config


def _collect_command(config, path):
    write_yaml(path, config)
    return [sys.executable, "-m", "latency_bench", "collect", "--config", str(path)]


def _collect_shards(config, shards, directory):
    processes = []
    output_dirs = []
    try:
        for index in range(shards["count"]):
            shard = deepcopy(config)
            output = directory / "shards" / str(index)
            output_dirs.append(output)
            shard["logging"]["output_dir"] = str(output)
            shard["experiment"]["seed"] += index * shards["seed_stride"]
            shard["executor"]["parallel_envs"] = shards["parallel_envs"]
            shard["evaluation"]["episodes"] = shards["episodes"]
            shard["collection"]["probe_only"] = True
            command = _collect_command(shard, directory / "configs" / f"probe-{index}.yaml")
            with (directory / f"probe-{index}.log").open("w") as log:
                processes.append(subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=log))
        for process in processes:
            result = process.wait()
            if result:
                raise subprocess.CalledProcessError(result, process.args)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
    state = merge_probe_states(
        [read_json(output / "probe_state.json") for output in output_dirs],
        max_episodes=config["evaluation"]["episodes"],
        val_fraction=config["collection"]["val_fraction"],
        split_seed=config["collection"]["split_seed"],
    )
    write_probe_state(Path(config["logging"]["output_dir"]), state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = load_config(args.config)
    experiment = recipe["dataset_experiment"]
    config = _run_config(recipe)
    output = Path(config["logging"]["output_dir"]).expanduser().absolute()
    directory = output.with_name(output.name + "_collection")
    directory.mkdir(parents=True, exist_ok=True)
    config["logging"]["output_dir"] = str(output)
    config_path = directory / "collect.yaml"
    shards = experiment["probe_shards"]
    if shards is not None:
        _collect_shards(config, shards, directory)
    subprocess.run(_collect_command(config, config_path), cwd=REPO_ROOT, check=True)
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
