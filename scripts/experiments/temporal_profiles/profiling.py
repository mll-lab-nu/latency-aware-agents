"""Measure Flappy/QwenOFT once and publish comparable fitted timing generators."""
from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import version
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from urllib.parse import quote

from latency_bench.core.config import load_config
from latency_bench.eval.artifacts import read_runtime_snapshot
from latency_bench.policy.model_source import resolve_model_source
from latency_bench.profiling.runner import fit_profile, run_profile
from latency_bench.profiling.runtime_configs import measure_config
from latency_bench.profiling.spec import resolve_profile_config
from latency_bench.utils.io import read_json, write_json_atomic, write_yaml
from latency_bench.utils.paths import REPO_ROOT


RECIPE = REPO_ROOT / "configs/examples/flappy/profile_openvla.yaml"
BACKBONE = {"repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17"}


def code_provenance() -> dict:
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).splitlines(),
        "starvla_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT / "third_party/starVLA", text=True).strip(),
        "fitting_packages": {name: version(name) for name in
                             ("numpy", "scipy", "scikit-learn", "hmmlearn", "matplotlib")},
    }


def check_sessions(source: Path, repeats: int) -> None:
    for session in range(repeats):
        (source / "sessions" / str(session) / "DONE").read_bytes()
        if not (source / "sessions" / str(session) / "observation_attempts.jsonl").is_file():
            raise FileNotFoundError(f"Missing timing records for session {session}: {source}")


def download_model(config: dict, output: Path) -> dict:
    from huggingface_hub import snapshot_download

    config = deepcopy(config)
    policy = config["policy"]
    required = [policy["checkpoint"], policy["model_config"], "config.yaml", "dataset_statistics.json"]
    snapshot = Path(snapshot_download(
        repo_id=policy["hf_repo"], revision=policy["revision"],
        allow_patterns=[f"{policy['root']}/{name}" for name in required],
    ))
    root = snapshot / policy["root"]
    model_config = load_config(root / policy["model_config"])
    if model_config["base_model"]["repo_id"] != BACKBONE["repo_id"]:
        raise ValueError("The zero-latency checkpoint requires a different backbone")
    backbone_root = Path(snapshot_download(**BACKBONE))
    config["provenance"] = {
        "model_artifact": {"source": "hf", "repo_id": policy["hf_repo"],
                           "revision": snapshot.name, "path_in_repo": policy["root"],
                           "checkpoint": policy["checkpoint"], "model_config": policy["model_config"]},
        "backbone_artifact": {"source": "hf", **BACKBONE, "revision": backbone_root.name},
    }
    policy.update(source="local", root=str(root), backbone_path=str(backbone_root))
    metadata = output / "model_metadata"
    metadata.mkdir(exist_ok=True)
    for name in (policy["model_config"], "config.yaml", "dataset_statistics.json"):
        shutil.copyfile(root / name, metadata / name)
    return resolve_model_source(config)


def prepare(args) -> None:
    output = args.output_dir
    if args.existing_run_dir:
        source = args.existing_run_dir.expanduser().absolute()
        contract = read_json(source / "run_contract.json")
        config = read_runtime_snapshot(contract["config"])
        if config["profiling"]["calibration"] is not None:
            config = measure_config(config, contract["deployment"], source,
                                    read_json(source / "calibration.json"))
        check_sessions(source, config["profiling"]["repeats"])
    else:
        source = None
        config = load_config(RECIPE)
        config["experiment"]["name"] = "flappy_zero_qwenoft_temporal_alternatives"
        config["evaluation"].update(episodes=args.episodes, max_env_steps=args.max_env_steps)
        config["profiling"]["repeats"] = args.repeats
        config["logging"]["output_dir"] = str(output / "raw")
    config = resolve_profile_config(config)
    if (config["env"]["name"] != "flappy" or config["policy"]["type"] != "starvla"
            or config["executor"]["mode"] != "realtime"
            or len(config["executor"]["inference_devices"]) != 1):
        raise ValueError("This recipe requires Flappy/StarVLA, realtime, one inference worker")
    repeats = config["profiling"]["repeats"]
    fit_ids = args.fit_session_ids
    if (repeats < 2 or not fit_ids or len(set(fit_ids)) != len(fit_ids)
            or not set(fit_ids) < set(range(repeats))):
        raise ValueError("Choose distinct fitting session IDs and reserve at least one other session")
    if min(config["evaluation"]["episodes"], config["evaluation"]["max_env_steps"]) < 1:
        raise ValueError("Profiling episode and frame budgets must be positive")
    if (len(set(args.methods)) != len(args.methods) or args.hmm_states < 1
            or args.hmm_max_iterations < 2 or args.block_mean_length < 1
            or not math.isfinite(args.block_mean_length)):
        raise ValueError("Use distinct methods, >=1 HMM state, >=2 iterations and finite block length >=1")
    protocol = {
        "schema_version": 1, "measurement_config": config,
        "existing_run_dir": str(source) if source else None,
        "methods": args.methods, "fit_session_ids": fit_ids,
        "held_out_session_ids": sorted(set(range(repeats)) - set(fit_ids)),
        "hmm_states": args.hmm_states, "hmm_max_iterations": args.hmm_max_iterations,
        "block_mean_length": args.block_mean_length,
        "selection": "Explicit settings; this recipe does not tune or evaluate profiles",
    }
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "protocol.json"
    if plan.exists():
        if read_json(plan) != protocol:
            raise ValueError("Protocol changed; use a new OUTPUT_DIR (reuse EXISTING_RUN_DIR to refit)")
    else:
        write_json_atomic(plan, protocol)
    if source is None and not (output / "measurement.yaml").exists():
        resolved = download_model(config, output)
        write_json_atomic(output / "measurement_code.json", code_provenance())
        temporary = output / "measurement.tmp.yaml"
        write_yaml(temporary, resolved)
        temporary.replace(output / "measurement.yaml")
    print(f"Protocol: {plan}\nFit sessions: {fit_ids}; reserved: {protocol['held_out_session_ids']}")


def measure(output: Path) -> None:
    protocol = read_json(output / "protocol.json")
    receipt = output / "measurement.json"
    if receipt.exists():
        check_sessions(Path(read_json(receipt)["RUN_DIR"]),
                       protocol["measurement_config"]["profiling"]["repeats"])
        print(f"Reusing completed measurement: {read_json(receipt)['RUN_DIR']}")
        return
    if protocol["existing_run_dir"]:
        source = Path(protocol["existing_run_dir"])
        result = {"RUN_DIR": str(source), "RUN_ID": source.name, "MEASUREMENT_STATUS": "reused"}
    else:
        contracts = list((output / "raw/runs").rglob("run_contract.json"))
        if len(contracts) > 1:
            raise ValueError("Multiple measurement runs in OUTPUT_DIR; select EXISTING_RUN_DIR explicitly")
        resume_id = contracts[0].parent.name if contracts else None
        result = run_profile(load_config(output / "measurement.yaml"), resume_run_id=resume_id)
        source = Path(result["RUN_DIR"])
    check_sessions(source, protocol["measurement_config"]["profiling"]["repeats"])
    write_json_atomic(receipt, result)
    print(f"RUN_DIR={source}")


def export_profiles(output: Path) -> None:
    protocol = read_json(output / "protocol.json")
    source = Path(read_json(output / "measurement.json")["RUN_DIR"])
    check_sessions(source, protocol["measurement_config"]["profiling"]["repeats"])
    if not (output / "fitting_code.json").exists():
        write_json_atomic(output / "fitting_code.json", code_provenance())
    profiles = output / "profiles"
    profiles.mkdir(exist_ok=True)
    for method in protocol["methods"]:
        target = profiles / method
        options = dict(method=method, session_ids=protocol["fit_session_ids"],
                       hmm_states=protocol["hmm_states"],
                       hmm_max_iterations=protocol["hmm_max_iterations"],
                       block_mean_length=protocol["block_mean_length"])
        # The runner's automatic Temporal fit uses ALL sessions. Refit it here
        # just like the alternatives; only these explicit-split fits are exported.
        if target.exists():
            fit_profile(source, output_dir=target, **options)
        else:
            with tempfile.TemporaryDirectory(prefix=f".fit-{method}-", dir=output) as tmp:
                fitted = Path(tmp) / method
                fit_profile(source, output_dir=fitted, **options)
                fitted.rename(target)
        print(f"{method}: {target / 'profile.json'}")
    bundle = output / "bundle"
    if bundle.exists():
        if read_json(bundle / "protocol.json") != protocol:
            raise ValueError("Exported bundle uses a different protocol")
        print(f"Reusing export: {bundle}")
        return
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as tmp:
        staged = Path(tmp) / "bundle"
        staged.mkdir()
        for name in ("protocol.json", "measurement.json", "measurement.yaml",
                     "measurement_code.json", "fitting_code.json"):
            if (output / name).exists():
                shutil.copyfile(output / name, staged / name)
        if (output / "model_metadata").exists():
            shutil.copytree(output / "model_metadata", staged / "model_metadata")
        raw = staged / "measurement"
        raw.mkdir()
        for name in ("run_contract.json", "hardware.json", "calibration.json", "calibration.DONE"):
            if (source / name).exists():
                shutil.copyfile(source / name, raw / name)
        # Copy recorded evidence only, never model weights or downloaded caches.
        for path in (source / "sessions").rglob("*"):
            if path.is_file() and (path.suffix in {".json", ".jsonl", ".yaml", ".png", ".log"}
                                   or path.name == "DONE"):
                dest = raw / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        shutil.copytree(profiles, staged / "profiles")
        cfg = protocol["measurement_config"]
        (staged / "README.md").write_text(
            "# Flappy/QwenOFT timing profiles\n\n"
            f"Measurement run: `{source.name}`. Measured clocks: "
            f"{cfg['env']['env_fps']}/{cfg['env']['obs_fps']} Hz; "
            f"{cfg['profiling']['repeats']} independent sessions, "
            f"{cfg['evaluation']['episodes']} measured episodes per session, "
            f"{cfg['evaluation']['max_env_steps']} raw-frame episode cap.\n\n"
            f"All exported methods fit sessions {protocol['fit_session_ids']}. "
            f"Reserved sessions: {protocol['held_out_session_ids']}. "
            "The runner's automatic all-session Temporal fit is excluded.\n\n"
            f"HMM states: {protocol['hmm_states']}; iteration limit: {protocol['hmm_max_iterations']}; "
            f"block mean length: {protocol['block_mean_length']} admitted requests. "
            "These are explicit settings, not automatically selected winners. "
            "No timing-fidelity evaluation, teacher training, or VLA evaluation is performed here.\n\n"
            "Keep each profiles/<method>/ directory intact: profile.json references its companion files. "
            "measurement/ contains raw timing records (including reserved sessions), configuration, "
            "warmup records and hardware evidence. run_contract.json and profile provenance identify "
            "the measured model/backbone revisions; code records identify fitting software. "
            "The 7200-frame demonstration-collection budget applies to a later stage.\n",
            encoding="utf-8",
        )
        (staged / "DONE").touch()
        staged.rename(bundle)
    print(f"Exported: {bundle}")


def upload(output: Path, repo_id: str, path_prefix: str) -> dict:
    from huggingface_hub import HfApi

    prefix = PurePosixPath(path_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or str(prefix) == ".":
        raise ValueError("HF path prefix must be a nonempty relative repository path")
    bundle = output / "bundle"
    (bundle / "DONE").read_bytes()
    protocol = read_json(bundle / "protocol.json")
    profile = read_json(bundle / "profiles" / protocol["methods"][0] / "profile.json")
    destination = (prefix / "flappy" / "openvla" / profile["gpu_class"]
                   / profile["source_run_id"] / output.name).as_posix()
    receipt_path = output / "upload_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["repo_id"] != repo_id or receipt["path_in_repo"] != destination:
            raise ValueError("This output already has an upload receipt for a different destination")
    else:
        api = HfApi()
        # Use an existing dataset repo and a new folder, never replace a published profile.
        head = api.repo_info(repo_id, repo_type="dataset").sha
        if api.get_paths_info(repo_id, [destination], repo_type="dataset", revision=head):
            raise FileExistsError(f"HF destination already exists: {repo_id}/{destination}")
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(bundle), path_in_repo=destination,
            commit_message=f"Add Flappy timing profile comparison {profile['source_run_id']}",
            parent_commit=head,
        )
        receipt = {"repo_id": repo_id, "repo_type": "dataset", "path_in_repo": destination,
                   "revision": commit.oid, "commit_url": commit.commit_url,
                   "url": f"https://huggingface.co/datasets/{repo_id}/tree/{commit.oid}/{quote(destination)}"}
        write_json_atomic(receipt_path, receipt)
    print(f"Uploaded: {receipt['url']}\nReceipt: {receipt_path}")
    return receipt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "measure", "export", "upload"):
        sub = commands.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            sub.add_argument("--existing-run-dir", type=Path)
            sub.add_argument("--repeats", type=int, default=5)
            sub.add_argument("--episodes", type=int, default=10)
            sub.add_argument("--max-env-steps", type=int, default=3600)
            sub.add_argument("--methods", nargs="+", choices=("temporal", "hmm", "block"),
                             default=["temporal", "hmm", "block"])
            sub.add_argument("--fit-session-ids", nargs="+", type=int, default=[0, 1, 2])
            sub.add_argument("--hmm-states", type=int, default=2)
            sub.add_argument("--hmm-max-iterations", type=int, default=1000)
            sub.add_argument("--block-mean-length", type=float, default=32)
        elif command == "upload":
            sub.add_argument("--repo-id", default="MLL-Lab/LAGEN-profiles")
            sub.add_argument("--path-prefix", default="experiments/temporal-alternatives")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().absolute()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "measure":
        measure(args.output_dir)
    elif args.command == "export":
        export_profiles(args.output_dir)
    else:
        upload(args.output_dir, args.repo_id, args.path_prefix)


if __name__ == "__main__":
    main()
