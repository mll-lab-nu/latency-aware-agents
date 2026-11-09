"""Prepare, export and publish matched Flappy teachers for the uploaded profiles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile

from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json, read_yaml, write_json_atomic, write_yaml
from latency_bench.utils.paths import REPO_ROOT


METHODS = ("temporal", "hmm", "block")
PROFILE_REPO = "latency-sensitive-bench/temporal-profiles"
PROFILE_REVISION = "1a4bd7bbc14815e0478810457ff204d4cf70636b"
SOURCE_RUN = "20261109T002620386065Z"
PROFILE_ROOT = ("experiments/temporal-alternatives/flappy/openvla/1x-rtx3090/"
                f"{SOURCE_RUN}/flappy_3090_run1")
TEACHER_REPO = "latency-sensitive-bench/temporal-profile-teachers"
SPEC = REPO_ROOT / "configs/experiments/flappy/temporal_alternatives_25m.yaml"
REFERENCE = {
    "repo_id": "MLL-Lab/LAGEN-models",
    "revision": "b17aad08bc4549fdbab5af96e698b3c637a2095a",
    "path": "latency-aware/flappy/small-policy/sample-factory-qwenoft-rtx3090-temporal-hist8-capacity1-s42-v1/config.json",
}


def validate_profile(profile: Path, method: str) -> dict:
    """Reject the wrong run/partition; sample using the actual downstream clock."""
    from latency_bench.core.clock import EnvClock
    from latency_bench.latency.samplers import build_sampler

    manifest = read_json(profile)
    expected = {"workload_id": "flappy", "model_id": "openvla", "env_fps": 30,
                "source_run_id": SOURCE_RUN, "latency_method": method}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Wrong source profile for {method}: {profile}")
    if manifest["provenance"]["session_ids"] != [0, 1, 2]:
        raise ValueError("Teacher profiles must use fitting sessions 0,1,2 only")
    sampler = build_sampler({"method": method, "profile_path": str(profile), "seed": 42},
                            clock=EnvClock(env_fps=10, obs_fps=10))
    sampler.sample({"worker_slot": 0, "episode_id": 0})
    return manifest


def download_profile(output: Path, method: str) -> Path:
    from huggingface_hub import snapshot_download

    remote = f"{PROFILE_ROOT}/profiles/{method}"
    snapshot = Path(snapshot_download(
        repo_id=PROFILE_REPO, repo_type="dataset", revision=PROFILE_REVISION,
        allow_patterns=[f"{remote}/*"],
    ))
    source = snapshot / remote
    destination = output / "profile"
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix="profile-", dir=output) as temporary:
            staged = Path(temporary) / "profile"
            # HF cache entries are symlinks into blobs/. Materialize the bundle
            # before the sampler checks that companions stay inside its folder.
            shutil.copytree(source, staged)
            validate_profile(staged / "profile.json", method)
            staged.rename(destination)
    else:
        # Resume must not silently accept edited profile parameters.
        for path in source.glob("*.json"):
            if read_json(path) != read_json(destination / path.name):
                raise ValueError("Local profile changed; use a new output directory")
    return destination / "profile.json"


def teacher_spec(output: Path, profile: Path, method: str, seed: int) -> dict:
    spec = read_yaml(SPEC)
    spec["__base__"] = str((SPEC.parent / spec["__base__"]).resolve())
    spec.update(generated_root=str(output / "configs"), train_dir=str(output / "checkpoints"),
                output_root=str(output / "teacher_runs"), seeds=[seed])
    spec["env"]["max_episode_steps"] = 0
    # Pin defaults present in the released teacher, alongside the inherited
    # optimizer/rollout settings. The training input includes 24 history bits.
    spec["training"].update(
        encoder_mlp_layers=[512, 512], optimizer="adam", adam_eps=1e-5,
        exploration_loss_coeff=.003, reward_scale=1., reward_clip=1000.,
        save_best_every_sec=5, save_best_after=100000,
    )
    spec["latency_settings"] = {method: {
        "kind": method, "profile_path": str(profile), "profile_worker_slot": 0,
    }}
    return spec


def prepare(output: Path, method: str, seed: int) -> None:
    from scripts.experiment_runner.job_manifest import load_manifest
    from scripts.sweeps.generate_sweep_configs import main as generate_configs

    output.mkdir(parents=True, exist_ok=True)
    profile = download_profile(output, method)
    manifest = validate_profile(profile, method)
    spec = teacher_spec(output, profile, method, seed)
    settings = {"method": method, "seed": seed, "spec": spec,
                "base_config": load_config(spec["__base__"]),
                "profile_source": {"repo_id": PROFILE_REPO, "repo_type": "dataset",
                                   "revision": PROFILE_REVISION,
                                   "path": f"{PROFILE_ROOT}/profiles/{method}"}}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["settings"] != settings:
            raise ValueError("Training settings changed; use a new OUTPUT_DIR or seed")
    else:
        protocol = {
            "settings": settings, "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "reference_teacher": REFERENCE,
            "profile_fit_options": manifest["provenance"]["fit_options"],
            "profile_policy": "Exact uploaded fit, no refitting or offline-selected replacements",
            "repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "working_tree": subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines(),
            "sample_factory_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT / "third_party/sample-factory", text=True).strip(),
        }
        write_json_atomic(protocol_path, protocol)
    write_yaml(output / "sweep.yaml", spec)
    generate_configs(["--spec", str(output / "sweep.yaml"), "--eval-checkpoint-kind", "best"])
    job, = load_manifest(output / "configs/manifest.tsv")
    shutil.copy2(job["train_config"], output / "train.yaml")
    write_json_atomic(output / "job.json", job)
    print(f"Prepared {method}: {manifest['provenance']['fit_options']}")
    print(f"Training config: {output / 'train.yaml'}")
    print(f"Checkpoint directory: {job['checkpoint_dir']}")


def completion_evidence(output: Path) -> tuple[dict, Path]:
    from training.common.checkpoints import find_checkpoint, trusted_torch_load

    config = load_config(output / "train.yaml")
    backend = config["backend"]
    experiment = config["experiment"]["name"]
    latest = find_checkpoint(backend["train_dir"], experiment, kind="latest")
    best = find_checkpoint(backend["train_dir"], experiment, kind="best")
    latest_state = trusted_torch_load(latest, "cpu")
    latest_steps = int(latest_state["env_steps"])
    del latest_state
    target = int(config["training"]["train_for_env_steps"])
    if latest_steps < target:
        raise ValueError(f"Training is incomplete: {latest_steps:,} / {target:,} decision transitions")
    best_state = trusted_torch_load(best, "cpu")
    best_steps = int(best_state["env_steps"])
    del best_state
    root = latest.parent.parent
    flat = read_json(root / "config.json")
    expected = {
        "seed": config["experiment"]["seed"], "env_fps": 10, "obs_fps": 10,
        "frame_stack": 4, "action_history_decisions": 8, "simulated_worker_capacity": 1,
        "train_for_env_steps": target, "use_rnn": False, "add_latency_info": False,
        "save_best_metric": "reward", "max_episode_steps": 0,
        "latency_type": {"temporal": "profile_sample", "hmm": "hmm", "block": "block"}[config["latency"]["method"]],
        "latency_profile_path": config["latency"]["profile_path"],
    }
    if any(flat.get(k) != value for k, value in expected.items()):
        raise ValueError("Saved teacher configuration does not match the frozen training recipe")
    evidence = {
        "status": "training_complete", "target_decision_transitions": target,
        "latest_decision_transitions": latest_steps, "best_decision_transitions": best_steps,
        "selected_checkpoint": str(best.relative_to(root)),
        "latest_checkpoint": str(latest.relative_to(root)),
        "selection_rule": "Sample Factory best rolling training reward over up to 100 completed episodes",
        "selected_checkpoint_evaluation": "Not performed by these training/upload launchers",
    }
    return evidence, root


def export(output: Path) -> None:
    bundle = output / "bundle"
    if (bundle / "DONE").exists():
        write_json_atomic(output / "training_complete.json", read_json(bundle / "training_complete.json"))
        print(f"Reusing completed bundle: {bundle}")
        return
    evidence, root = completion_evidence(output)
    protocol = read_json(output / "protocol.json")
    method = protocol["settings"]["method"]
    with tempfile.TemporaryDirectory(prefix="export-", dir=output) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        shutil.copy2(root / "config.json", staged / "config.json")
        shutil.copytree(root / "checkpoint_p0", staged / "checkpoint_p0",
                        ignore=shutil.ignore_patterns("*.tmp"))
        shutil.copytree(output / "profile", staged / "profile")
        for name in ("protocol.json", "train.yaml", "sweep.yaml", "job.json", "train.log"):
            shutil.copy2(output / name, staged / name)
        for name in ("episode_metrics.jsonl", "summaries"):
            source = root / name
            if source.is_dir():
                shutil.copytree(source, staged / name)
            elif source.is_file():
                shutil.copy2(source, staged / name)
        if (output / "teacher_runs").exists():
            shutil.copytree(output / "teacher_runs", staged / "teacher_runs")
        code = staged / "code"
        code.mkdir()
        for name in ("teacher.py", "_train_teacher.sh", f"train_{method}.sh"):
            shutil.copy2(Path(__file__).parent / name, code / name)
        write_json_atomic(staged / "training_complete.json", evidence)
        (staged / "README.md").write_text(
            f"# Flappy Bird teacher: {method}\n\n"
            "Sample Factory APPO, trained from scratch for 25M decision transitions. "
            "10/10 Hz, four 84x84 RGB frames, eight decision-history entries (24 values), "
            "capacity 1, two actors x 128 environments, seed "
            f"{protocol['settings']['seed']}.\n\n"
            f"Selected checkpoint: `{evidence['selected_checkpoint']}` at "
            f"{evidence['best_decision_transitions']:,} transitions. "
            f"Latest checkpoint reached {evidence['latest_decision_transitions']:,} transitions. "
            "Selection uses rolling training reward, not held-out game scores.\n\n"
            f"Profile: `{PROFILE_REPO}` (dataset), revision `{PROFILE_REVISION}`, "
            f"`{PROFILE_ROOT}/profiles/{method}`. Exact uploaded settings: "
            f"`{protocol['profile_fit_options']}`. No timing refit is performed. "
            "These original uploads differ from the later offline-selected HMM/block profiles.\n\n"
            "Training completion is verified from the latest checkpoint's env_steps. "
            "The selected best checkpoint has not been independently evaluated by this launcher. "
            "Any automatic end-of-training evaluation under teacher_runs refers to the latest "
            "checkpoint, uses 100 episodes and a 3,600-frame cap, and is not the best-checkpoint score.\n\n"
            "The original config.json retains training-machine paths as provenance. "
            "For later collection, pass the explicit downloaded checkpoint and "
            "`--profile-path /downloaded/bundle/profile/profile.json` to "
            "`python -m scripts.experiments.temporal_profiles.prepare_collection`. "
            "That stage defaults to 200 episodes with a 7,200-raw-frame cap; it is not run here.\n\n"
            "The copied training config, source revision, profile and completion evidence "
            "are included beside the retained checkpoints.\n", encoding="utf-8")
        (staged / "DONE").touch()
        staged.rename(bundle)
    write_json_atomic(output / "training_complete.json", evidence)
    print(f"Saved checkpoint bundle: {bundle}\nSelected: {evidence['selected_checkpoint']}")


def upload(output: Path, repo_id: str) -> dict:
    from huggingface_hub import HfApi

    bundle = output / "bundle"
    (bundle / "DONE").read_bytes()
    protocol = read_json(bundle / "protocol.json")
    settings = protocol["settings"]
    destination = f"flappy/{settings['method']}/{SOURCE_RUN}/seed_{settings['seed']}/{protocol['run_id']}"
    receipt_path = output / "upload_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["repo_id"] != repo_id or receipt["path_in_repo"] != destination:
            raise ValueError("This run was already uploaded to a different destination")
    else:
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        head = api.repo_info(repo_id, repo_type="model").sha
        if api.get_paths_info(repo_id, [destination], repo_type="model", revision=head):
            raise FileExistsError(f"HF output exists; refusing to overwrite {repo_id}/{destination}")
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(bundle), path_in_repo=destination,
            commit_message=f"Add Flappy {settings['method']} teacher, seed {settings['seed']}",
            parent_commit=head,
        )
        receipt = {"repo_id": repo_id, "repo_type": "model", "path_in_repo": destination,
                   "revision": commit.oid, "commit_url": commit.commit_url,
                   "url": f"https://huggingface.co/{repo_id}/tree/{commit.oid}/{destination}"}
        write_json_atomic(receipt_path, receipt)
    print(f"Uploaded: {receipt['url']}")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "export", "upload"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--method", choices=METHODS, required=True)
            sub.add_argument("--seed", type=int, default=42)
        elif name == "upload":
            sub.add_argument("--repo-id", default=TEACHER_REPO)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.command == "prepare":
        prepare(output, args.method, args.seed)
    elif args.command == "export":
        export(output)
    else:
        upload(output, args.repo_id)


if __name__ == "__main__":
    main()
