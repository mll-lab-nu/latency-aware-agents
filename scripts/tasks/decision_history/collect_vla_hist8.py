"""Collect one temporal-profile teacher dataset for scratch history VLA training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.parquet as pq

from latency_bench.data.lerobot_export import convert_canonical_rollout_to_lerobot
from latency_bench.data.rollout_probe import write_probe_state, merge_probe_states


ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = ROOT / "outputs/tasks/decision_history_temporal_25m_seed42"
MODEL_REPO = "MLL-Lab/LAGEN-models"
MODEL_REPO_REVISION = "af125bd372c22c5b00e326d9af1af64ea3751b56"
MODEL_LABELS = {"qwenoft": "qwenoft", "qwengr00t": "gr00t", "qwenpi_v3": "pi05"}
TASK_SETTINGS = {
    "flappy": (200, 50, 3600, 256, 256, "gpu"),
    "demon_attack": (200, 50, 7200, 256, 256, "gpu"),
    "deadly_corridor": (2000, 250, 3600, 16, 8, "cpu"),
    "atlantis": (200, 50, 7200, 256, 256, "gpu"),
    "asterix": (200, 50, 7200, 256, 256, "gpu"),
}
HISTORY_DIMS = {"flappy": 24, "demon_attack": 88, "deadly_corridor": 120, "atlantis": 56, "asterix": 136}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_receipt(task: str, model: str) -> dict:
    slug = task.replace("_", "-")
    label = MODEL_LABELS[model]
    entries = [
        item
        for name in ("hf_publish_receipt.json", "hf_publish_receipt_three.json", "hf_publish_receipt_demon_pi05.json")
        for item in json.loads((RECEIPTS / name).read_text())["models"]
        if f"/latency-aware/{slug}/" in "/" + item["path"] + "/"
        and f"sample-factory-{label}-" in item["path"]
    ]
    return entries[0]


def _profile_receipt(task: str, model: str) -> dict:
    label = MODEL_LABELS[model]
    profiles = json.loads((RECEIPTS / "profile_revisions.json").read_text())
    return next(
        item for item in profiles
        if item["job_id"].startswith(f"{task}_decision_history_")
        and f":{label}_rtx3090_temporal_hist8_capacity1" in item["job_id"]
    )


def _export_command(
    *, checkpoint: Path, output_dir: Path, episodes: int, horizon: int,
    probe_envs: int, replay_envs: int, device: str, probe_only: bool,
) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts/rollout_data/export_sf_teacher_dataset.py"),
        "--checkpoint-root", str(checkpoint), "--output-dir", str(output_dir),
        "--max-episodes", str(episodes), "--max-steps-per-episode", str(horizon),
        "--max-attempts", str(episodes),
        "--num-envs", str(probe_envs), "--replay-num-envs", str(replay_envs),
        "--device", device, "--val-fraction", "0.1", "--split-seed", "0",
        "--filter-preset", "none", "--context-window", "1",
        "--image-writer-workers", "32" if device == "gpu" else "8",
    ]
    if probe_only:
        command.append("--probe-only")
    return command


def _quality_report(raw_dir: Path, specs: list[dict], metadata: dict) -> dict:
    action_counts = Counter()
    admitted = 0
    previous_decisions = 0
    for split in ("train", "val"):
        parquet = pq.ParquetFile(raw_dir / f"{split}.parquet")
        for batch in parquet.iter_batches(batch_size=8192, columns=["action_id", "state", "decision_step"]):
            columns = batch.to_pydict()
            action_counts.update(columns["action_id"])
            for state, step in zip(columns["state"], columns["decision_step"]):
                if step > 0:
                    previous_decisions += 1
                    admitted += int(state[-(len(state) // 8)])
    lengths = np.asarray([item["episode_raw_frames"] for item in specs])
    returns = np.asarray([item["episode_raw_return"] for item in specs])
    return {
        "episodes": len(specs),
        "train_episodes": metadata["train_episodes"],
        "val_episodes": metadata["val_episodes"],
        "train_labels": metadata["train_rows"],
        "val_labels": metadata["val_rows"],
        "raw_frames_quantiles": np.quantile(lengths, [0, .1, .5, .9, 1]).tolist(),
        "return_quantiles": np.quantile(returns, [0, .1, .5, .9, 1]).tolist(),
        "termination_reasons": dict(Counter(item["termination_reason"] for item in specs)),
        "action_id_counts": dict(sorted(action_counts.items())),
        "previous_decision_admitted_fraction": admitted / previous_decisions,
    }


def audit_atari_history(raw_dir: Path, task: str) -> dict:
    """Audit exported pre-decision history against each episode's causal action trace."""
    dimension = HISTORY_DIMS[task]
    width = dimension // 8
    action_bits = (width - 1) // 2
    basis = np.eye(action_bits + 1)[:, 1:]
    checked = 0
    episodes = 0
    for split in ("train", "val"):
        previous = {}
        columns = ["episode_idx", "decision_step", "action_id", "state"]
        for batch in pq.ParquetFile(raw_dir / f"{split}.parquet").iter_batches(columns=columns):
            for row in batch.to_pylist():
                state = np.asarray(row["state"])
                assert state.shape == (dimension,)
                assert np.all((state == 0) | (state == 1))
                windows = state.reshape(8, width)
                assert np.all(windows[:, 1:1 + action_bits].sum(axis=1) <= 1)
                assert np.all(windows[:, 1 + action_bits:].sum(axis=1) <= 1)
                if row["decision_step"] == 0:
                    assert not np.any(state)
                    episodes += 1
                else:
                    prior, action, step = previous[row["episode_idx"]]
                    assert step + 1 == row["decision_step"]
                    np.testing.assert_array_equal(state[:-width], prior[width:])
                    np.testing.assert_array_equal(windows[-1, 1:1 + action_bits],
                                                  windows[-1, 0] * basis[action])
                previous[row["episode_idx"]] = (state, row["action_id"], row["decision_step"])
                checked += 1
    return {"task": task, "checked_rows": checked, "episodes": episodes, "status": "passed",
            "history": "binary H8, zero reset, causal window shifts and admitted teacher commands"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SETTINGS, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--teacher-catalog", type=Path,
                        help="Pinned teacher/profile publication catalog for Atlantis and Asterix")
    parser.add_argument("--deadly-probe-shards", type=int, default=0)
    parser.add_argument("--deadly-shard-episodes", type=int, default=2400)
    args = parser.parse_args()
    initial, increment, horizon, probe_envs, replay_envs, device = TASK_SETTINGS[args.task]
    output = args.output_root.resolve() / args.task / args.model
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw"
    if args.task in {"atlantis", "asterix"}:
        catalog = json.loads(args.teacher_catalog.read_text())
        receipt = catalog["teachers"][args.task][args.model]
        repo_revision = catalog["revision"]
        profile_receipt = receipt
        profile_root = args.profile_root.resolve() / "results/profiling"
        snapshot_download("MLL-Lab/LAGEN-profiles", repo_type="dataset",
                          revision=catalog["profile_revision"],
                          allow_patterns=[str(Path(receipt["profile_path"]).parent / "*")],
                          local_dir=profile_root)
        profile_path = profile_root / receipt["profile_path"]
    else:
        receipt = _model_receipt(args.task, args.model)
        profile_receipt = _profile_receipt(args.task, args.model)
        profile_rel = profile_receipt["profile_path"].split("results/profiling/profiles/", 1)[1]
        profile_path = args.profile_root.resolve() / "results/profiling/profiles" / profile_rel
        repo_revision = MODEL_REPO_REVISION
    if _file_sha256(profile_path) != profile_receipt["profile_sha256"]:
        raise RuntimeError(f"Profile content mismatch: {profile_path}")
    snapshot_download(
        MODEL_REPO, revision=repo_revision, allow_patterns=[f"{receipt['path']}/**"],
        local_dir=output / "models",
    )
    bundle = output / "models" / receipt["path"]
    checkpoint = bundle / receipt["checkpoint"]
    if _file_sha256(checkpoint) != receipt["sha256"]:
        raise RuntimeError(f"Teacher checkpoint content mismatch: {checkpoint}")
    teacher_cfg = json.loads((bundle / "config.json").read_text())
    if teacher_cfg["action_history_decisions"] != 8:
        raise RuntimeError("Teacher checkpoint does not use eight-decision history")
    if args.task == "flappy" and teacher_cfg["use_gpu_render"] is not True:
        raise RuntimeError("Flappy teacher does not use GPU rendering")
    if teacher_cfg["latency_profile_worker_slot"] != 0:
        raise RuntimeError("Teacher profile worker slot is not zero")
    if args.task in {"atlantis", "asterix"}:
        # Preserve the published CPU training recipe; this directory records only
        # the explicit profile relocation and data-collection seed.
        shutil.copyfile(bundle / "config.json", output / "teacher_source_config.json")
        runtime_bundle = output / "teacher_runtime"
        runtime_bundle.mkdir(exist_ok=True)
        runtime_checkpoint = runtime_bundle / receipt["checkpoint"]
        if not runtime_checkpoint.exists():
            os.link(checkpoint, runtime_checkpoint)
        checkpoint = runtime_checkpoint
        (runtime_bundle / "config.json").write_text(json.dumps(
            teacher_cfg | {"latency_profile_path": str(profile_path), "seed": 42}, indent=2,
        ) + "\n")

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "third_party/sample-factory"),
        str(ROOT / "third_party/flappy-bird-gymnasium"), str(ROOT / "third_party/cule"),
    ])
    if device == "gpu":
        cuda_home = Path(sys.executable).parents[1]
        compiler = cuda_home / "bin/x86_64-conda-linux-gnu-g++"
        run_env["CUDA_HOME"] = str(cuda_home)
        run_env["CXX"] = str(compiler)
        run_env["NVCC_PREPEND_FLAGS"] = f"-ccbin={compiler}"
    else:
        run_env["OMP_NUM_THREADS"] = "8"
        run_env["MKL_NUM_THREADS"] = "8"
        run_env["OPENBLAS_NUM_THREADS"] = "1"
    if args.deadly_probe_shards:
        existing = json.loads((raw_dir / "probe_state.json").read_text())
        shard_processes = []
        for shard_idx in range(args.deadly_probe_shards):
            shard_dir = raw_dir / "probe_shards" / f"shard_{shard_idx}"
            command = _export_command(
                checkpoint=checkpoint, output_dir=shard_dir,
                episodes=args.deadly_shard_episodes, horizon=horizon,
                probe_envs=probe_envs, replay_envs=replay_envs,
                device=device, probe_only=True,
            )
            command.extend(["--rollout-seed-offset", str((shard_idx + 10) * 1_000_000)])
            log = (output / f"probe-shard-{shard_idx}.log").open("w")
            shard_processes.append((subprocess.Popen(
                command, cwd=args.profile_root.resolve(), env=run_env,
                stdout=log, stderr=subprocess.STDOUT,
            ), log, shard_dir))
        stopped_at_label_target = False
        while any(process.poll() is None for process, _, _ in shard_processes):
            time.sleep(20)
            for process, _, _ in shard_processes:
                if process.poll() not in (None, 0):
                    for other, _, _ in shard_processes:
                        if other.poll() is None:
                            other.terminate()
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            states = [existing] + [
                json.loads((shard_dir / "probe_state.json").read_text())
                for _, _, shard_dir in shard_processes
            ]
            count = sum(len(state["accepted_specs"]) for state in states)
            batch_count = initial + ((count - initial) // increment) * increment
            merged_progress = merge_probe_states(
                states, max_episodes=batch_count, val_fraction=.1, split_seed=0,
            )
            train_labels = sum(
                item["episode_length"] for item in merged_progress["accepted_specs"]
                if item["split"] == "train"
            )
            if train_labels >= 320_000:
                stopped_at_label_target = True
                for process, _, _ in shard_processes:
                    if process.poll() is None:
                        process.terminate()
                break
        for process, log, _ in shard_processes:
            process.wait()
            log.close()
        for process, _, _ in shard_processes:
            if process.returncode and not stopped_at_label_target:
                raise subprocess.CalledProcessError(process.returncode, process.args)
        states = [existing] + [
            json.loads((shard_dir / "probe_state.json").read_text())
            for _, _, shard_dir in shard_processes
        ]
        episode_count = sum(len(state["accepted_specs"]) for state in states)
        for batch_count in range(initial, episode_count + 1, increment):
            candidate = merge_probe_states(
                states, max_episodes=batch_count, val_fraction=.1, split_seed=0,
            )
            train_labels = sum(
                item["episode_length"] for item in candidate["accepted_specs"]
                if item["split"] == "train"
            )
            if train_labels >= 320_000:
                episode_count = batch_count
                break
        write_probe_state(raw_dir, merge_probe_states(
            states, max_episodes=episode_count, val_fraction=.1, split_seed=0,
        ))
    episodes = initial
    if (raw_dir / "probe_state.json").exists():
        episodes = max(episodes, len(json.loads((raw_dir / "probe_state.json").read_text())["accepted_specs"]))
    while True:
        subprocess.run(_export_command(
            checkpoint=checkpoint, output_dir=raw_dir, episodes=episodes, horizon=horizon,
            probe_envs=probe_envs, replay_envs=replay_envs, device=device, probe_only=True,
        ), cwd=args.profile_root.resolve(), env=run_env, check=True)
        state = json.loads((raw_dir / "probe_state.json").read_text())
        if device == "gpu":
            state["accepted_specs"].sort(key=lambda item: item["seed"])
        merged = merge_probe_states([state], max_episodes=episodes, val_fraction=.1, split_seed=0)
        train_labels = sum(item["episode_length"] for item in merged["accepted_specs"] if item["split"] == "train")
        if train_labels >= 320_000:
            break
        episodes += increment
    write_probe_state(raw_dir, merged)
    specs = merged["accepted_specs"]
    subprocess.run(_export_command(
        checkpoint=checkpoint, output_dir=raw_dir, episodes=episodes, horizon=horizon,
        probe_envs=probe_envs, replay_envs=replay_envs, device=device, probe_only=False,
    ), cwd=args.profile_root.resolve(), env=run_env, check=True)
    metadata = json.loads((raw_dir / "metadata.json").read_text())
    if metadata["train_rows"] < 320_000:
        raise RuntimeError("Replay produced fewer labels than the probe budget")
    if args.task in {"atlantis", "asterix"}:
        (output / "history_audit.json").write_text(json.dumps(audit_atari_history(raw_dir, args.task), indent=2) + "\n")
    manifest = convert_canonical_rollout_to_lerobot(raw_dir, output / "lerobot")
    if manifest["state_dim"] != HISTORY_DIMS[args.task]:
        raise RuntimeError("Converted history dimension does not match the teacher")
    report = _quality_report(raw_dir, specs, metadata)
    (output / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "teacher_repo": MODEL_REPO, "teacher_revision": repo_revision,
        "teacher_path": receipt["path"], "teacher_checkpoint": receipt["checkpoint"],
        "teacher_sha256": receipt["sha256"], "profile_path": str(profile_path),
        "profile_sha256": profile_receipt["profile_sha256"],
        **({"profile_repo": "MLL-Lab/LAGEN-profiles", "profile_revision": catalog["profile_revision"],
            "rollout_overrides": {"latency_profile_path": str(profile_path), "seed": 42,
                                  "simulator": device, "probe_envs": probe_envs,
                                  "replay_envs": replay_envs, "policy_inference_batch_size": probe_envs,
                                  "max_episode_raw_frames": horizon}}
           if args.task in {"atlantis", "asterix"} else {}),
        "episodes": episodes, "filter_preset": "none", "history": "decision_v1",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
