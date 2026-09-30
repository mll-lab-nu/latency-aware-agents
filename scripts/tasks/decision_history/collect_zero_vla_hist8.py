"""Collect one shared zero-latency H8 dataset per game from its published teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from huggingface_hub import snapshot_download

from latency_bench.utils.io import write_json_atomic
from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot
from latency_bench.data.sf_teacher_probe import write_probe_state, merge_probe_states
from scripts.tasks.decision_history.collect_vla_hist8 import (
    ROOT, TASK_SETTINGS, HISTORY_DIMS, _file_sha256, _export_command, _quality_report,
)


def collect_rollouts(args, catalog: dict, teacher: dict, output: Path) -> None:
    """Run probe/replay only, releasing its GPU before conversion and transfer."""
    raw = output / "raw"
    initial, increment, horizon, probe_envs, replay_envs, device = TASK_SETTINGS[args.task]
    if args.smoke:
        initial, horizon = (20, 256) if args.task in {"atlantis", "asterix"} else (10, 32)
    snapshot_download(catalog["repo_id"], revision=catalog["revision"],
                      allow_patterns=[teacher["path"] + "/**"], local_dir=output / "models")
    bundle = output / "models" / teacher["path"]
    runtime_bundle = output / "teacher_runtime"
    runtime_bundle.mkdir(exist_ok=True)
    checkpoint = runtime_bundle / teacher["checkpoint"]
    if not checkpoint.exists():
        os.link(bundle / teacher["checkpoint"], checkpoint)
    assert _file_sha256(checkpoint) == teacher["checkpoint_sha256"]
    source_config = json.loads((bundle / "config.json").read_text())
    write_json_atomic(output / "teacher_source_config.json", source_config)
    overrides = {"env_fps": teacher["env_fps"], "obs_fps": teacher["obs_fps"],
                 "latency_type": "fixed", "fixed_latency_ms": 0.0, "seed": 42}
    write_json_atomic(runtime_bundle / "config.json", source_config | overrides)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / path) for path in (
        ".", "third_party/sample-factory", "third_party/flappy-bird-gymnasium", "third_party/cule"))
    cuda_home = Path(sys.executable).parents[1]
    env["CUDA_HOME"] = str(cuda_home)
    env["CXX"] = str(cuda_home / "bin/x86_64-conda-linux-gnu-g++")
    env["NVCC_PREPEND_FLAGS"] = f"-ccbin={env['CXX']}"
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env["OPENBLAS_NUM_THREADS"] = "1"
    episodes = initial
    if (raw / "probe_state.json").exists():
        episodes = max(episodes, len(json.loads((raw / "probe_state.json").read_text())["accepted_specs"]))

    def export(probe_only):
        command = _export_command(checkpoint=checkpoint, output_dir=raw, episodes=episodes,
                                  horizon=horizon, probe_envs=probe_envs, replay_envs=replay_envs,
                                  device=device, probe_only=probe_only)
        if args.task not in {"atlantis", "asterix"}:
            command.extend(["--export-zero-action-history", "8"])
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    while True:
        export(True)
        state = json.loads((raw / "probe_state.json").read_text())
        if device == "gpu":
            state["accepted_specs"].sort(key=lambda item: item["seed"])
        merged = merge_probe_states([state], max_episodes=episodes, val_fraction=.1, split_seed=0)
        labels = sum(item["episode_length"] for item in merged["accepted_specs"] if item["split"] == "train")
        if args.smoke or labels >= 320_000:
            break
        episodes += increment
    write_probe_state(raw, merged)
    write_json_atomic(output / "accepted_episodes.json", {"episodes": merged["accepted_specs"]})
    export(False)
    metadata = json.loads((raw / "metadata.json").read_text())
    provenance = {"teacher_repo": catalog["repo_id"], "teacher_revision": catalog["revision"],
                  "teacher_path": teacher["path"], "teacher_checkpoint": teacher["checkpoint"],
                  "teacher_sha256": teacher["checkpoint_sha256"], "rollout_overrides": overrides,
                  "training_latency_condition": "zero", "history": "decision_v1", "history_decisions": 8,
                  "history_source": "teacher_transport" if args.task in {"atlantis", "asterix"}
                  else "zero_latency_issued_equals_applied", "student_image_frames": 1,
                  "max_episode_raw_frames": horizon,
                  "episodes": episodes, "train_labels": metadata["train_rows"], "filter_preset": "none"}
    write_json_atomic(output / "rollout.complete.json", provenance)


def materialize_rollouts(task: str, output: Path, mirror_host: str | None) -> None:
    """Convert the completed rollout and publish the training-ready marker last."""
    raw = output / "raw"
    provenance = json.loads((output / "rollout.complete.json").read_text())
    metadata = json.loads((raw / "metadata.json").read_text())
    specs = json.loads((output / "accepted_episodes.json").read_text())["episodes"]
    manifest = convert_canonical_rollout_to_lerobot(raw, output / "lerobot")
    assert manifest["state_dim"] == HISTORY_DIMS[task]
    write_json_atomic(output / "quality_report.json", _quality_report(raw, specs, metadata))
    write_json_atomic(output / "provenance.json", provenance)
    if mirror_host:
        target = f"ubuntu@{mirror_host}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", target,
                        f"mkdir -p {shlex.quote(str(output / 'lerobot'))} {shlex.quote(str(output / 'raw'))}"], check=True)
        subprocess.run(["rsync", "-aL", *[str(output / name) for name in
                        ("lerobot", "lerobot__val", "_generated_mixtures")], f"{target}:{output}/"], check=True)
        subprocess.run(["rsync", "-a", str(output / "provenance.json"), str(output / "quality_report.json"),
                        f"{target}:{output}/"], check=True)
        subprocess.run(["rsync", "-a", str(raw / "metadata.json"), f"{target}:{output / 'raw'}/"], check=True)
    marker = output / "collection.complete.json"
    write_json_atomic(marker, provenance)
    if mirror_host:
        subprocess.run(["rsync", "-a", str(marker), f"ubuntu@{mirror_host}:{output}/"], check=True)
    print(json.dumps({"completed": str(output), "episodes": provenance["episodes"], "train_labels": metadata["train_rows"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SETTINGS, required=True)
    parser.add_argument("--teacher-catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mirror-host")
    parser.add_argument("--stage", choices=("rollout", "convert", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve() / args.task
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"rollout", "all"}:
        catalog = json.loads(args.teacher_catalog.read_text())
        collect_rollouts(args, catalog, catalog["teachers"][args.task], output)
    if args.stage in {"convert", "all"}:
        materialize_rollouts(args.task, output, args.mirror_host)


if __name__ == "__main__":
    main()
