"""Collect and publish matched H8 Flappy demonstrations from local teacher bundles."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from latency_bench.utils.io import read_json, read_yaml, write_json_atomic, write_yaml
from latency_bench.utils.paths import REPO_ROOT
from scripts.experiments.temporal_profiles import prepare_collection, teacher


DATASET_REPO = "latency-sensitive-bench/temporal-profile-rollouts"


def prepare(output: Path, bundle: Path, method: str, *, episodes=200, max_steps=7200,
            min_train_rows=320_000) -> None:
    """Select the recorded best checkpoint and keep its profile beside the export."""
    output, bundle = output.resolve(), bundle.resolve()
    if episodes < 10 or max_steps < 1 or min_train_rows < 1:
        raise ValueError("Require at least 10 episodes and positive frame/row budgets")
    (bundle / "DONE").read_bytes()
    completed = read_json(bundle / "training_complete.json")
    flat = read_json(bundle / "config.json")
    training_protocol = read_json(bundle / "protocol.json")
    if (completed["status"] != "training_complete"
            or completed["latest_decision_transitions"] < completed["target_decision_transitions"]):
        raise ValueError("The local teacher bundle does not record completed training")
    expected = {
        "latency_type": {"temporal": "profile_sample", "hmm": "hmm", "block": "block"}[method],
        "seed": 42, "latency_seed": 42, "env_fps": 10, "obs_fps": 10,
        "frame_stack": 4, "action_history_decisions": 8, "use_gpu_render": True,
        "simulated_worker_capacity": 1, "latency_profile_worker_slot": 0,
        "hold_policy": "one_frame_then_noop", "ordering_policy": "latest_ready",
        "add_latency_info": False,
    }
    if training_protocol["settings"]["method"] != method or any(flat.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Local bundle does not match the {method} Flappy 10/10 Hz H8 recipe")
    checkpoint = bundle / completed["selected_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    teacher.validate_profile(bundle / "profile/profile.json", method)
    settings = {
        "method": method, "teacher_bundle": str(bundle), "selected_checkpoint": completed["selected_checkpoint"],
        "initial_episodes": episodes, "episode_increment": 50, "min_train_rows": min_train_rows,
        "max_episode_raw_frames": max_steps, "seed": 42, "profile_seed": 42,
        "env_fps": 10, "obs_fps": 10, "parallel_envs": 256, "replay_num_envs": 256,
        "val_fraction": .1, "split_seed": 0, "filter_preset": "none", "deterministic": True,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path)["settings"] != settings:
            raise ValueError("Collection settings changed; use a new OUTPUT_DIR")
        for filename, original in (("teacher_config.json", flat), ("teacher_training_complete.json", completed),
                                   ("teacher_protocol.json", training_protocol)):
            if read_json(output / filename) != original:
                raise ValueError("Local teacher metadata changed; use a new OUTPUT_DIR")
        for source in (bundle / "profile").glob("*.json"):
            if read_json(source) != read_json(output / "profile" / source.name):
                raise ValueError("Local profile changed; use a new OUTPUT_DIR")
    else:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle / "profile", output / "profile", dirs_exist_ok=True)
        prepare_collection.prepare(checkpoint, output / "profile/profile.json", output,
                                   episodes=episodes, max_steps=max_steps)
        config = read_yaml(output / "collect.yaml")
        config["policy"]["deterministic"] = True
        write_yaml(output / "collect.yaml", config)
        write_json_atomic(output / "teacher_config.json", flat)
        write_json_atomic(output / "teacher_training_complete.json", completed)
        write_json_atomic(output / "teacher_protocol.json", training_protocol)
        write_json_atomic(protocol_path, {
            "settings": settings, "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "reference_collector": "scripts/tasks/decision_history/collect_vla_hist8.py",
            "budget_note": "7200 raw frames is the explicitly chosen comparison cap; current reference source defaults to 3600 for Flappy.",
            "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines(),
        })
    print(f"Local {method} teacher: {checkpoint}\nRollout configuration: {output / 'collect.yaml'}")


def collect(output: Path) -> None:
    from latency_bench.data.rollout import collect_rollouts
    from latency_bench.data.rollout_probe import merge_probe_states, write_probe_state
    from scripts.tasks.decision_history.collect_vla_hist8 import _quality_report, audit_atari_history

    settings = read_json(output / "protocol.json")["settings"]
    raw = output / "raw"
    if not (raw / "metadata.json").exists():
        config = read_yaml(output / "collect.yaml")
        episodes = settings["initial_episodes"]
        if (raw / "probe_state.json").exists():
            episodes = max(episodes, len(read_json(raw / "probe_state.json")["accepted_specs"]))
        while True:
            config["evaluation"]["episodes"] = episodes
            config["collection"].update(max_attempts=episodes, probe_only=True)
            write_yaml(output / "probe.yaml", config)
            collect_rollouts(deepcopy(config))
            state = read_json(raw / "probe_state.json")
            if len(state["accepted_specs"]) < episodes:
                raise RuntimeError("Probe did not complete its episode budget; preserve this run and use a new OUTPUT_DIR")
            # Same ordering and whole-episode split as the H8 VLA collector.
            state["accepted_specs"].sort(key=lambda item: item["seed"])
            merged = merge_probe_states([state], max_episodes=episodes, val_fraction=.1, split_seed=0)
            rows = sum(item["episode_length"] for item in merged["accepted_specs"] if item["split"] == "train")
            print(f"Probe: {episodes} episodes, {rows:,} training rows (target {settings['min_train_rows']:,})", flush=True)
            if rows >= settings["min_train_rows"]:
                break
            episodes += settings["episode_increment"]
        write_probe_state(raw, merged)
        config["collection"]["probe_only"] = False
        write_yaml(output / "collect.yaml", config)
        collect_rollouts(deepcopy(config))
    metadata = read_json(raw / "metadata.json")
    if metadata["train_rows"] < settings["min_train_rows"]:
        raise ValueError("Replay produced fewer training rows than the collection budget")
    if metadata.get("action_history_decisions") != 8 or metadata["env_fps"] != 10 or metadata["obs_fps"] != 10:
        raise ValueError("Collected data does not match the Flappy 10/10 Hz H8 recipe")
    specs = read_json(raw / "probe_state.json")["accepted_specs"]
    write_json_atomic(output / "history_audit.json", audit_atari_history(raw, "flappy"))
    report = _quality_report(raw, specs, metadata)
    write_json_atomic(output / "quality_report.json", report)
    write_json_atomic(output / "collection_complete.json", {
        "status": "collection_complete", "episodes": len(specs),
        "train_rows": metadata["train_rows"], "val_rows": metadata["val_rows"],
    })
    print(f"Collected {len(specs)} episodes: {metadata['train_rows']:,} train / {metadata['val_rows']:,} validation rows")


def export(output: Path) -> None:
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

    completed = read_json(output / "collection_complete.json")
    if (output / "DONE").exists():
        print(f"Reusing completed dataset: {output}")
        return
    # force replaces only derived conversion outputs; raw demonstrations stay intact.
    manifest = convert_canonical_rollout_to_lerobot(output / "raw", output / "lerobot", force=True)
    if (manifest["state_dim"] != 24 or manifest["action_history_decisions"] != 8
            or manifest["frames"] != completed["train_rows"]
            or manifest["validation_frames"] != completed["val_rows"]):
        raise ValueError("LeRobot conversion changed the history shape or row counts")
    protocol = read_json(output / "protocol.json")
    settings = protocol["settings"]
    code = output / "code"
    code.mkdir(exist_ok=True)
    for name in ("rollout.py", "prepare_collection.py", "_collect_teacher.sh", f"collect_{settings['method']}.sh"):
        shutil.copy2(Path(__file__).parent / name, code / name)
    (output / "README.md").write_text(
        f"# Flappy {settings['method']} teacher demonstrations\n\n"
        f"{completed['episodes']} episodes, {completed['train_rows']:,} training rows and "
        f"{completed['val_rows']:,} validation rows. Selected local checkpoint: "
        f"`{settings['selected_checkpoint']}`. Teacher metadata and the exact fitted profile are included.\n\n"
        f"10/10 Hz simulated collection, capacity 1, deterministic teacher, seed 42, "
        f"{settings['max_episode_raw_frames']:,}-raw-frame episode cap, two-pass probe/replay, "
        "256 environments, 90/10 episode split (seed 0), no filtering. Start at "
        f"{settings['initial_episodes']} episodes and add 50 until at least "
        f"{settings['min_train_rows']:,} training rows. The 7,200-frame default is an explicit "
        "comparison setting, not a claim that every archived dataset used that cap.\n\n"
        "`lerobot/` and `lerobot__val/` are the StarVLA training/validation datasets. "
        "Images are embedded in Parquet. Each row preserves the current RGB image, "
        "24 binary values for eight completed decisions, and the teacher-issued action. "
        "Teacher image stacking remains four frames; the VLA uses one image and action horizon 1. "
        "The H8 collector uses the existing task prompt without added latency text.\n\n"
        "Use the existing history-aware Flappy StarVLA recipe with state dimension 24 "
        "and identity state normalization. Point data_root_dir at this downloaded folder, "
        "data_mix at lerobot, and custom_mixtures_path at `_generated_mixtures/lerobot.json`. "
        "Validation uses lerobot__val. Paths in the source manifests/configs record the "
        "collection machine; set local paths after download. No checkpoint weights are uploaded here.\n\n"
        "`raw/` retains canonical embedded-image Parquet and collection metadata. "
        "`quality_report.json` records returns, lengths and action counts; "
        "`history_audit.json` checks binary history, reset, shifting and admitted commands. "
        "These are demonstration statistics, not held-out VLA evaluation scores.\n",
        encoding="utf-8",
    )
    (output / "DONE").touch()
    print(f"VLA-ready datasets: {output / 'lerobot'} and {output / 'lerobot__val'}")


def upload(output: Path, repo_id: str) -> dict:
    from huggingface_hub import HfApi

    (output / "DONE").read_bytes()
    protocol = read_json(output / "protocol.json")
    settings = protocol["settings"]
    destination = f"flappy/{settings['method']}/{teacher.SOURCE_RUN}/seed_42/{protocol['run_id']}"
    receipt_path = output / "upload_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["repo_id"] != repo_id or receipt["path_in_repo"] != destination:
            raise ValueError("This dataset was already uploaded to a different destination")
    else:
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        head = api.repo_info(repo_id, repo_type="dataset").sha
        if api.get_paths_info(repo_id, [destination], repo_type="dataset", revision=head):
            raise FileExistsError(f"HF output exists; refusing to overwrite {repo_id}/{destination}")
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(output), path_in_repo=destination,
            allow_patterns=["lerobot/**", "lerobot__val/**", "_generated_mixtures/**", "profile/**",
                            "raw/*.parquet", "raw/*.json", "raw/*.yaml", "code/**", "*.json", "*.yaml", "README.md", "DONE"],
            commit_message=f"Add Flappy {settings['method']} H8 teacher demonstrations",
            parent_commit=head,
        )
        receipt = {"repo_id": repo_id, "repo_type": "dataset", "path_in_repo": destination,
                   "revision": commit.oid, "commit_url": commit.commit_url,
                   "url": f"https://huggingface.co/datasets/{repo_id}/tree/{commit.oid}/{destination}"}
        write_json_atomic(receipt_path, receipt)
    print(f"Uploaded dataset: {receipt['url']}")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "collect", "export", "upload"):
        sub = commands.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            sub.add_argument("--teacher-bundle", type=Path, required=True)
            sub.add_argument("--method", choices=teacher.METHODS, required=True)
            sub.add_argument("--episodes", type=int, default=200)
            sub.add_argument("--max-steps", type=int, default=7200)
            sub.add_argument("--min-train-rows", type=int, default=320_000)
        elif command == "upload":
            sub.add_argument("--repo-id", default=DATASET_REPO)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.command == "prepare":
        prepare(output, args.teacher_bundle.expanduser(), args.method, episodes=args.episodes,
                max_steps=args.max_steps, min_train_rows=args.min_train_rows)
    elif args.command == "collect":
        collect(output)
    elif args.command == "export":
        export(output)
    else:
        upload(output, args.repo_id)


if __name__ == "__main__":
    main()
