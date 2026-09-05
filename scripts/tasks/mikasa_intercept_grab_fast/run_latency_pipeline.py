"""Run MIKASA profiling, command PPO, distillation and closed-loop evaluation."""

import argparse
import os
from pathlib import Path
import subprocess

import yaml

from latency_bench.core.config import load_config
from latency_bench.cross_device.profile_resolver import resolve_model_artifacts
from latency_bench.cross_device.spec import load_profile_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-path", type=Path, help="Explicit existing profile; skip measuring a new one.")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mikasa = root / "third_party/MIKASA-Robo"
    python = str(mikasa / ".venv/bin/python")
    environment = os.environ | config["environment"]
    environment["PYTHONPATH"] = os.pathsep.join([str(root), str(mikasa)])

    def run(stage, command, *, cwd=root):
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    profile = args.profile_path
    if profile is None:
        run("profile", ["bash", "scripts/profile.sh", "run", "--model", "qwenoft",
            "--workload", "mikasa_intercept_grab_fast", "--session-python", python,
            *config["profile_arguments"]])
        profile = Path(next(line.removeprefix("PROFILE_PATH=") for line in
            (output / "profile.log").read_text().splitlines() if line.startswith("PROFILE_PATH=")))
    latency = load_config(config["latency_config"])
    latency["latency"]["profile_path"] = str(profile.resolve())
    latency_path = output / "latency.yaml"
    latency_path.write_text(yaml.safe_dump(latency, sort_keys=False))
    teacher = output / "teacher"
    run("teacher", [python, str(mikasa / "mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py"),
        "--env-id", "InterceptGrabFast-VLA-v0", "--latency-config", str(latency_path),
        "--checkpoint", config["teacher_checkpoint"], "--latency-output-dir", str(teacher), "--no-track"])
    data = output / "dataset"
    run("collect", ["bash", "scripts/tasks/mikasa_intercept_grab_fast/collect_latency.sh",
        "--latency-config", str(latency_path), "--checkpoint", str(teacher / "teacher.pt"),
        "--path-to-save-data", str(data), "--num-train-data", str(config["collection"]["episodes"]),
        "--num-envs", str(config["collection"]["num_envs"]),
        *(["--keep-failed"] if config["collection"]["keep_failed"] else [])])
    run("rlds", ["uv", "run", "--project", str(mikasa / "utils/convert_npz_to_rlds/rlds_dataset_builder"),
        "python", str(mikasa / "utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py"),
        "--task", "InterceptGrabFast-VLA-v0", "--repo-root", str(mikasa),
        "--npz-root", str(data / "data_npz"), "--rlds-root", str(data / "data_rlds")])
    for split in ("train", "val"):
        run(f"lerobot_{split}", ["uv", "run", "--project", str(mikasa / "utils/convert_rlds_to_lerobot"),
            "python", str(mikasa / "utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py"),
            "--task", "InterceptGrabFast-VLA-v0", "--repo-root", str(mikasa),
            "--rlds-root", str(data / "data_rlds"), "--lerobot-root", str(data / "data_lerobot"),
            "--repo-id-template", f"local/{{task}}_h1_{split}", "--fps", "20",
            "--split", config["collection"][f"{split}_split"]])
    run("modalities", [python, "scripts/mikasa/prepare_starvla_dataset.py", "--data-root", str(data / "data_lerobot"),
        "--train-name", "local/intercept_grab_fast_vla_v0_h1_train", "--val-name", "local/intercept_grab_fast_vla_v0_h1_val"])
    vla = config["vla"]
    run("vla", ["bash", "scripts/tasks/mikasa_intercept_grab_fast/train_latency_vla.sh",
        str(data / "data_lerobot"), vla["initial_checkpoint"], "--run_id", vla["run_id"],
        "--trainer.max_train_steps", str(vla["steps"]), "--trainer.save_interval", str(vla["steps"]),
        *vla["arguments"]])
    checkpoint_dir = root / "third_party/starVLA/results/Checkpoints" / vla["run_id"]
    baseline = resolve_model_artifacts(
        load_profile_spec("configs/profiles.yaml", "qwenoft", "mikasa_intercept_grab_fast"),
        hf_artifacts=True,
    )
    models = {
        "baseline": {key: baseline[key] for key in ("checkpoint_path", "model_config_path")},
        "latency": {
            "checkpoint_path": str(checkpoint_dir / f"checkpoints/steps_{vla['steps']}_pytorch_model.pt"),
            "model_config_path": str(checkpoint_dir / "config.full.yaml"),
        },
    }
    for model, policy_paths in models.items():
        for method in ("zero", latency["latency"]["method"]):
            evaluation = load_config(config["evaluation_config"])
            evaluation["latency"] = latency["latency"] | {"method": method}
            evaluation["scheduler"] = latency["scheduler"]
            evaluation["executor"]["simulated_worker_capacity"] = latency["executor"]["simulated_worker_capacity"]
            evaluation["policy"].update(policy_paths)
            stage = f"eval_{model}_{method}"
            evaluation["logging"]["output_dir"] = str(output / stage)
            evaluation_path = output / f"{stage}.yaml"
            evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
            # Native model configs resolve their pretrained VLM path from starVLA.
            run(stage, [python, str(root / "scripts/mikasa/evaluate.py"), "latency-eval",
                "--eval-config", str(evaluation_path)], cwd=root / "third_party/starVLA")


if __name__ == "__main__":
    main()
