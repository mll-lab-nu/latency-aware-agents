"""Profile one published zero-H8 VLA on one visible GPU using the CPU simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from latency_bench.profiling.runner import run_profile
from latency_bench.core.config import load_config
from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.eval_vla_hist8 import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    publication = json.loads(args.publication.read_text())
    task, model = publication["task"], publication["model"]
    config = load_config(f"configs/examples/{task}/profile_{model}.yaml")
    config["policy"].update(source="hf", hf_repo=publication["repo_id"], root=publication["path"],
                                  revision=publication["revision"])
    checkpoint = Path(hf_hub_download(
        publication["repo_id"],
        (Path(publication["path"]) / config["policy"]["checkpoint"]).as_posix(),
        revision=publication["revision"],
    ))
    if file_sha256(checkpoint) != publication["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the model publication")
    result = run_profile(config, resume_run_id=args.resume)
    # This experiment publishes its measured condition for downstream teacher training.
    repository = "MLL-Lab/LAGEN-profiles"
    run_dir = Path(result["RUN_DIR"])
    profile_dir = Path(result["PROFILE_PATH"]).parent
    profile_ref = result["PROFILE_REF"]
    private_files = {"calibration.generated.yaml", "generated.yaml", "resolved_config.yaml",
                     "hardware.json", "run_contract.json"}
    operations = [CommitOperationAdd(
        path_in_repo=f"runs/{profile_ref}/{result['RUN_ID']}/{path.relative_to(run_dir).as_posix()}",
        path_or_fileobj=str(path)) for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in private_files and path.relative_to(run_dir).parts[0] != "profile"]
    operations.extend(CommitOperationAdd(
        path_in_repo=f"profiles/{profile_ref}/{path.name}", path_or_fileobj=str(path))
        for path in sorted(profile_dir.iterdir()) if path.is_file())
    commit = HfApi().create_commit(repo_id=repository, repo_type="dataset", operations=operations,
                                 commit_message=f"Publish Atari H8 measurement {task}/{model}")
    profile_path = "profiles/" + profile_ref + "/profile.json"
    remote = Path(hf_hub_download(repository, profile_path, repo_type="dataset", revision=commit.oid))
    profile_sha = file_sha256(Path(result["PROFILE_PATH"]))
    if file_sha256(remote) != profile_sha:
        raise RuntimeError("Published profile does not match the local measurement")
    write_json_atomic(Path("outputs/tasks/atari_hist8_v1/profile_receipts") / f"{task}-{model}.json",
                      {"task": task, "model": model, "model_publication": publication, "profile": result,
                       "profile_repo": repository, "profile_revision": commit.oid,
                       "profile_path": profile_path, "profile_sha256": profile_sha})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
