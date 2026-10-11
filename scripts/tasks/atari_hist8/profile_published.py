"""Profile one published zero-H8 VLA on one visible GPU using the CPU simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from latency_bench.profiling.runner import publish_profile, run_profile
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
    config = load_config(f"configs/examples/profiling/{task}/{model}.yaml")
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
    result.update(publish_profile(result["RUN_DIR"], repo_id=config["profiling"]["publish_repo"]))
    revision = HfApi().dataset_info(config["profiling"]["publish_repo"]).sha
    profile_path = "profiles/" + result["PROFILE_REF"] + "/profile.json"
    remote = Path(hf_hub_download(config["profiling"]["publish_repo"], profile_path, repo_type="dataset", revision=revision))
    profile_sha = file_sha256(Path(result["PROFILE_PATH"]))
    assert file_sha256(remote) == profile_sha
    write_json_atomic(Path("outputs/tasks/atari_hist8_v1/profile_receipts") / f"{task}-{model}.json",
                      {"task": task, "model": model, "model_publication": publication, "profile": result,
                       "profile_repo": config["profiling"]["publish_repo"], "profile_revision": revision,
                       "profile_path": profile_path, "profile_sha256": profile_sha})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
