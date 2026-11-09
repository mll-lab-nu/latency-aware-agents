"""Collect and publish matched H8 Flappy demonstrations from local teacher bundles."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import subprocess

from latency_bench.latency.config import latency_type_from_config
from latency_bench.utils.io import read_json, read_yaml, write_json_atomic, write_yaml
from latency_bench.utils.paths import REPO_ROOT
from scripts.experiments.temporal_profiles import prepare_collection, teacher


DATASET_REPO = "latency-sensitive-bench/temporal-profile-rollouts"


def prepare(output: Path, bundle: Path, method: str, *, episodes=200, max_steps=7200,
            min_train_rows=320_000, profile_set="original", episode_return_gt=None,
            max_attempts=None, parallel_envs=256) -> None:
    """Select the recorded best checkpoint and keep its profile beside the export."""
    output, bundle = output.resolve(), bundle.resolve()
    if episodes < 10 or max_steps < 1 or min_train_rows < 1:
        raise ValueError("Require at least 10 episodes and positive frame/row budgets")
    if parallel_envs < 1:
        raise ValueError("Parallel environment count must be positive")
    if episode_return_gt is not None:
        if not math.isfinite(episode_return_gt):
            raise ValueError("Episode return threshold must be finite")
        if max_attempts is None:
            max_attempts = 10 * episodes
    if max_attempts is not None and max_attempts < episodes:
        raise ValueError("MAX_ATTEMPTS must be at least the initial accepted episode target")
    (bundle / "DONE").read_bytes()
    completed = read_json(bundle / "training_complete.json")
    flat = read_json(bundle / "config.json")
    training_protocol = read_json(bundle / "protocol.json")
    if (completed["status"] != "training_complete"
            or completed["latest_decision_transitions"] < completed["target_decision_transitions"]):
        raise ValueError("The local teacher bundle does not record completed training")
    expected = {
        "latency_type": latency_type_from_config({"method": method}),
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
    teacher.validate_profile(bundle / "profile/profile.json", method, profile_set=profile_set)
    settings = {
        "method": method, "teacher_bundle": str(bundle), "selected_checkpoint": completed["selected_checkpoint"],
        "initial_episodes": episodes, "episode_increment": 50, "min_train_rows": min_train_rows,
        "max_episode_raw_frames": max_steps, "seed": 42, "profile_seed": 42,
        "env_fps": 10, "obs_fps": 10, "parallel_envs": parallel_envs, "replay_num_envs": parallel_envs,
        "val_fraction": .1, "split_seed": 0, "filter_preset": "none", "deterministic": True,
    }
    # Omit unset options so completed, unfiltered collections remain resumable.
    if episode_return_gt is not None:
        settings["episode_return_gt"] = episode_return_gt
    if max_attempts is not None:
        settings["max_attempts"] = max_attempts
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
                                   episodes=episodes, max_steps=max_steps, parallel_envs=parallel_envs)
        config = read_yaml(output / "collect.yaml")
        config["policy"]["deterministic"] = True
        for key in ("episode_return_gt", "max_attempts"):
            if key in settings:
                config["collection"][key] = settings[key]
        write_yaml(output / "collect.yaml", config)
        write_json_atomic(output / "teacher_config.json", flat)
        write_json_atomic(output / "teacher_training_complete.json", completed)
        write_json_atomic(output / "teacher_protocol.json", training_protocol)
        write_json_atomic(protocol_path, {
            "settings": settings, "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "profile_set": profile_set,
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
    summaries_path = output / "episode_summaries.json"
    if not (raw / "metadata.json").exists():
        config = read_yaml(output / "collect.yaml")
        episodes = settings["initial_episodes"]
        if (raw / "probe_state.json").exists():
            episodes = max(episodes, len(read_json(raw / "probe_state.json")["accepted_specs"]))
        while True:
            attempt_budget = settings.get("max_attempts", episodes)
            config["evaluation"]["episodes"] = episodes
            config["collection"].update(max_attempts=attempt_budget, probe_only=True)
            write_yaml(output / "probe.yaml", config)
            collect_rollouts(deepcopy(config))
            state = read_json(raw / "probe_state.json")
            if len(state["accepted_specs"]) < episodes:
                raise RuntimeError(
                    f"Probe accepted {len(state['accepted_specs'])}/{episodes} episodes "
                    f"within MAX_ATTEMPTS={attempt_budget} "
                    f"({state['rejected_episodes']} rejected; "
                    f"episode_return_gt={settings.get('episode_return_gt')}). "
                    "Replay was not started. Use a fresh OUTPUT_DIR with a larger "
                    "MAX_ATTEMPTS budget or a revised cutoff."
                )
            # Same ordering and whole-episode split as the H8 VLA collector.
            state["accepted_specs"].sort(key=lambda item: item["seed"])
            merged = merge_probe_states([state], max_episodes=episodes, val_fraction=.1, split_seed=0)
            rows = sum(item["episode_length"] for item in merged["accepted_specs"] if item["split"] == "train")
            print(f"Probe: {episodes} accepted episodes, {state['rejected_episodes']} rejected, "
                  f"{rows:,} training rows (target {settings['min_train_rows']:,})", flush=True)
            if rows >= settings["min_train_rows"]:
                break
            episodes += settings["episode_increment"]
        write_probe_state(raw, merged)
        # Finalizing canonical Parquet removes raw/probe_state.json. Retain the
        # reporting evidence outside that temporary replay directory.
        write_json_atomic(summaries_path, {"source": "probe", "episodes": merged["accepted_specs"]})
        config["collection"]["probe_only"] = False
        write_yaml(output / "collect.yaml", config)
        collect_rollouts(deepcopy(config))
    metadata = read_json(raw / "metadata.json")
    if metadata["train_rows"] < settings["min_train_rows"]:
        raise ValueError("Replay produced fewer training rows than the collection budget")
    if metadata.get("action_history_decisions") != 8 or metadata["env_fps"] != 10 or metadata["obs_fps"] != 10:
        raise ValueError("Collected data does not match the Flappy 10/10 Hz H8 recipe")
    if not summaries_path.exists():
        # Recover runs finalized before summaries were retained. Canonical rows
        # contain the lengths and rewards; no teacher or GPU replay is needed.
        write_json_atomic(summaries_path, {
            "source": "canonical_parquet", "episodes": _episode_summaries_from_parquet(raw, metadata),
            "note": "Recovered from saved rows after probe cleanup; original termination reasons are unavailable.",
        })
    summaries = read_json(summaries_path)
    specs = summaries["episodes"]
    write_json_atomic(output / "history_audit.json", audit_atari_history(raw, "flappy"))
    report = _quality_report(raw, specs, metadata)
    report["episode_summary_source"] = summaries["source"]
    if "note" in summaries:
        report["episode_summary_note"] = summaries["note"]
    write_json_atomic(output / "quality_report.json", report)
    write_json_atomic(output / "collection_complete.json", {
        "status": "collection_complete", "episodes": len(specs),
        "train_rows": metadata["train_rows"], "val_rows": metadata["val_rows"],
    })
    print(f"Collected {len(specs)} episodes: {metadata['train_rows']:,} train / {metadata['val_rows']:,} validation rows")


def _episode_summaries_from_parquet(raw: Path, metadata: dict) -> list[dict]:
    """Recover reporting evidence from canonical splits retaining every episode row."""
    import pyarrow.parquet as pq

    episodes = {}
    reward_field = metadata["reward_field"]
    stride = metadata["obs_stride_raw_frames"]
    for split in ("train", "val"):
        rows = 0
        split_episodes = set()
        for batch in pq.ParquetFile(raw / f"{split}.parquet").iter_batches(
            columns=["episode_idx", "seed", "decision_step", reward_field], batch_size=8192,
        ):
            for row in batch.to_pylist():
                episode_id = row["episode_idx"]
                spec = episodes.setdefault(episode_id, {
                    "episode_idx": episode_id, "seed": row["seed"], "split": split,
                    "episode_length": 0, "episode_raw_frames": 0, "episode_raw_return": 0.,
                    "termination_reason": "not_recorded",
                })
                if (spec["split"] != split or spec["seed"] != row["seed"]
                        or row["decision_step"] != spec["episode_length"]):
                    raise ValueError(f"Cannot recover episode {episode_id}: inconsistent or missing decision rows")
                spec["episode_length"] += 1
                spec["episode_raw_frames"] += stride
                spec["episode_raw_return"] += row[reward_field]
                rows += 1
                split_episodes.add(episode_id)
        if rows != metadata[f"{split}_rows"] or len(split_episodes) != metadata[f"{split}_episodes"]:
            raise ValueError(f"Saved {split} Parquet counts do not match collection metadata")
    return [episodes[key] for key in sorted(episodes)]


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
    profile_source = read_json(output / "teacher_protocol.json")["settings"]["profile_source"]
    launcher_dir = teacher.PROFILE_SETS[protocol.get("profile_set", "original")]["launcher_dir"]
    code = output / "code"
    code.mkdir(exist_ok=True)
    for name in ("rollout.py", "prepare_collection.py", "_collect_teacher.sh"):
        shutil.copy2(Path(__file__).parent / name, code / name)
    launcher = f"collect_{settings['method']}.sh"
    shutil.copy2(Path(__file__).parent.parent / launcher_dir / launcher, code / launcher)
    filtering = (
        f"Keep whole episodes with total raw return strictly > {settings['episode_return_gt']:g}; "
        f"retain all rows from accepted episodes. The total probe attempt cap is {settings['max_attempts']:,}. "
        "Both demonstration splits contain qualifying episodes. "
        if "episode_return_gt" in settings else "No episode or row filtering. "
    )
    (output / "README.md").write_text(
        f"# Flappy {settings['method']} teacher demonstrations\n\n"
        f"{completed['episodes']} episodes, {completed['train_rows']:,} training rows and "
        f"{completed['val_rows']:,} validation rows. Selected local checkpoint: "
        f"`{settings['selected_checkpoint']}`. Teacher metadata and the exact fitted profile are included.\n\n"
        f"Profile source: `{profile_source['repo_id']}` (dataset), revision "
        f"`{profile_source['revision']}`, `{profile_source['path']}`. "
        "Collection uses the local teacher's bundled fit without refitting or downloading replacements.\n\n"
        f"10/10 Hz simulated collection, capacity 1, deterministic teacher, seed 42, "
        f"{settings['max_episode_raw_frames']:,}-raw-frame episode cap, two-pass probe/replay, "
        f"{settings['parallel_envs']} environments, 90/10 episode split (seed 0). "
        f"{filtering}Start at {settings['initial_episodes']} accepted episodes and add 50 until at least "
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
        "`episode_summaries.json` preserves probe statistics, or identifies summaries recovered "
        "from existing Parquet with unavailable termination reasons marked not_recorded. "
        "`quality_report.json` records returns, lengths and action counts; "
        "`raw/filter_report.json` records the acceptance gate and attempted/accepted/rejected counts. "
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
    source_run = read_json(output / "profile/profile.json")["source_run_id"]
    destination = f"flappy/{settings['method']}/{source_run}/seed_42/{protocol['run_id']}"
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
            sub.add_argument("--method", choices=(*teacher.METHODS, "ar1", "arma"), required=True)
            sub.add_argument("--profile-set", choices=teacher.PROFILE_SETS, default="original")
            sub.add_argument("--episodes", type=int, default=200)
            sub.add_argument("--max-steps", type=int, default=7200)
            sub.add_argument("--min-train-rows", type=int, default=320_000)
            sub.add_argument("--parallel-envs", type=int, default=256,
                             help="Environment count shared by probe and image replay")
            sub.add_argument("--episode-return-gt", type=float,
                             help="Keep whole episodes with raw return strictly above this value")
            sub.add_argument("--max-attempts", type=int,
                             help="Total probe attempt cap; default 10x initial episodes when filtering")
        elif command == "upload":
            sub.add_argument("--repo-id", default=DATASET_REPO)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.command == "prepare":
        prepare(output, args.teacher_bundle.expanduser(), args.method, episodes=args.episodes,
                max_steps=args.max_steps, min_train_rows=args.min_train_rows, profile_set=args.profile_set,
                episode_return_gt=args.episode_return_gt, max_attempts=args.max_attempts,
                parallel_envs=args.parallel_envs)
    elif args.command == "collect":
        collect(output)
    elif args.command == "export":
        export(output)
    else:
        upload(output, args.repo_id)


if __name__ == "__main__":
    main()
