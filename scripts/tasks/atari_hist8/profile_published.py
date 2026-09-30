"""Profile one published zero-H8 VLA on one visible GPU using the CPU simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from latency_bench.profiling.profile_resolver import resolve_model_artifacts
from latency_bench.profiling.runner import run_profile
from latency_bench.profiling.spec import load_profile_spec
from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.eval_vla_hist8 import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    publication = json.loads(args.publication.read_text())
    task, model = publication["task"], publication["model"]
    spec = load_profile_spec("configs/profiles.yaml", model, task)
    spec["model_artifact"].update(hf_repo=publication["repo_id"], hf_path=publication["path"],
                                  revision=publication["revision"])
    spec = resolve_model_artifacts(spec, hf_artifacts=True)
    assert file_sha256(Path(spec["checkpoint_path"])) == publication["checkpoint_sha256"]
    result = run_profile(spec, resume_run_id=args.resume, publish=True)
    revision = HfApi().dataset_info(spec["hf_repo"]).sha
    profile_path = "profiles/" + result["PROFILE_REF"] + "/profile.json"
    remote = Path(hf_hub_download(spec["hf_repo"], profile_path, repo_type="dataset", revision=revision))
    profile_sha = file_sha256(Path(result["PROFILE_PATH"]))
    assert file_sha256(remote) == profile_sha
    write_json_atomic(Path("outputs/tasks/atari_hist8_v1/profile_receipts") / f"{task}-{model}.json",
                      {"task": task, "model": model, "model_publication": publication, "profile": result,
                       "profile_repo": spec["hf_repo"], "profile_revision": revision,
                       "profile_path": profile_path, "profile_sha256": profile_sha})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
