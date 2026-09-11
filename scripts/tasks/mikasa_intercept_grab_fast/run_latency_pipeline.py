"""Collect issued-action data and train QwenOFT with the shared H1 recipe."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import hf_hub_download, snapshot_download
import yaml

from latency_bench.core.config import load_config
from scripts.starvla.collect import _write_mikasa_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mikasa = root / "third_party/MIKASA-Robo"
    environment = os.environ | config["environment"]
    environment["PYTHONPATH"] = os.pathsep.join([str(root), str(mikasa)])
    environment["VLA_PY"] = str(root / ".venv/bin/python")

    def run(stage, command):
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=root, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    source = config["teacher_source"]
    teacher = hf_hub_download(source["repo_id"], source["path"], revision=source["revision"],
                              local_dir=root / "outputs/published-models")
    latency = load_config(config["latency_config"])
    source = latency["profile_source"]
    snapshot = snapshot_download(source["repo_id"], repo_type="dataset", revision=source["revision"],
                                 allow_patterns=[source["path"] + "/*"])
    profile_dir = output / "profile"
    shutil.copytree(Path(snapshot) / source["path"], profile_dir, dirs_exist_ok=True)
    latency["latency"]["profile_path"] = str(profile_dir / "profile.json")
    latency_path = output / "latency.yaml"
    latency_path.write_text(yaml.safe_dump(latency, sort_keys=False))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data = output / "dataset"
    run("collect", ["bash", "scripts/tasks/mikasa_intercept_grab_fast/collect_latency.sh",
        "--latency-config", str(latency_path), "--checkpoint", teacher,
        "--path-to-save-data", str(data), "--num-train-data", str(config["collection"]["episodes"]),
        "--num-envs", str(config["collection"]["num_envs"]),
        *(["--keep-failed"] if config["collection"]["keep_failed"] else [])])
    builder = mikasa / "utils/convert_npz_to_rlds/rlds_dataset_builder/.venv/bin/python"
    converter = mikasa / "utils/convert_rlds_to_lerobot/.venv/bin/python"
    run("rlds", [str(builder), str(mikasa / "utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py"),
        "--task", "InterceptGrabFast-VLA-v0", "--repo-root", str(mikasa),
        "--npz-root", str(data / "data_npz"), "--rlds-root", str(data / "data_rlds")])
    for split in ("train", "val"):
        run(f"lerobot_{split}", [str(converter),
            str(mikasa / "utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py"),
            "--task", "InterceptGrabFast-VLA-v0", "--repo-root", str(mikasa),
            "--rlds-root", str(data / "data_rlds"), "--lerobot-root", str(data / "data_lerobot"),
            "--repo-id-template", f"local/{{task}}_h1_{split}", "--fps", "20",
            "--split", config["collection"][f"{split}_split"]])
    for split, name in (("train", "lerobot"), ("val", "lerobot__val")):
        shutil.move(str(data / "data_lerobot/local" / f"intercept_grab_fast_vla_v0_h1_{split}"),
                    str(data / name))
    _write_mikasa_contract(data / "lerobot", data / "lerobot__val")
    (data / "provenance.json").write_text(json.dumps({
        "teacher": config["teacher_source"], "profile": source,
        "latency": latency["latency"], "collection": config["collection"],
        "action_labels": "issued actions at admitted request observations",
    }, indent=2) + "\n")
    vla = config["vla"]
    training = output / "vla"
    run("vla", ["bash", "scripts/tasks/mikasa_intercept_grab_fast/train_latency_vla.sh",
        str(data), teacher, "--run-id", vla["run_id"], "--output-dir", str(training),
        "--num-gpus", str(vla["num_gpus"]), "--micro-batch", str(vla["micro_batch"]),
        "--vla-python", vla["python"],
        "--no-publish", *(["--smoke"] if vla["smoke"] else [])])


if __name__ == "__main__":
    main()
