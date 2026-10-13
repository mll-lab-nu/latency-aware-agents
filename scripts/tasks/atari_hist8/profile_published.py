"""Profile one published zero-H8 VLA on one visible GPU using the CPU simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from latency_bench.profiling.runner import run_profile
from latency_bench.core.config import load_config
from latency_bench.utils.io import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    source = json.loads(args.model_source.read_text())
    model_source = {key: source[key] for key in ("task", "model", "repo_id", "path", "revision")}
    task, model = model_source["task"], model_source["model"]
    config = load_config(f"configs/examples/{task}/profile_{model}.yaml")
    config["policy"].update(source="hf", hf_repo=model_source["repo_id"], root=model_source["path"],
                                  revision=model_source["revision"])
    result = run_profile(config, resume_run_id=args.resume)
    source_path = Path(result["RUN_DIR"]) / "profile_source.json"
    write_json_atomic(source_path, {
        "task": task, "model": model, "model_source": model_source,
        "profile_path": result["PROFILE_PATH"],
    })
    print(f"PROFILE_SOURCE={source_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
